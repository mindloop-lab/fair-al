
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from .backends import make_backend
from .data import initial_labeled_split, load_dataset, train_val_test_split_indices
from .metrics import distribution, evaluate_predictions, q_bias, q_bias_l1
from .naming import canonical_strategy_name, public_strategy_name
from .strategies import make_strategy
from .utils import ensure_dir, env_report, parse_int_list_arg, parse_list_arg, save_json, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run FAIR-AL active-learning experiments")
    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument("--dataset", type=str, choices=["ABIDE_A", "ABIDE_B"], default="ABIDE_A")
    p.add_argument("--sensitive-index", type=int, default=None)
    p.add_argument("--backend", type=str, choices=["sklearn", "deepasd"], default="sklearn")
    p.add_argument("--classifier", type=str, choices=["svm", "lr", "mlp"], default="svm")
    p.add_argument("--strategies", type=str, default="random,uncertainty,reweighting,heuristic,coreset,badge,fairal_pareto")
    p.add_argument("--seeds", type=str, default="0,1,2,3,4")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--val-size", type=float, default=0.1)
    p.add_argument("--initial-size", type=int, default=40)
    p.add_argument("--budget", type=int, default=120)
    p.add_argument("--batch-size", type=int, default=10)
    p.add_argument("--k-candidates", type=int, default=200)
    p.add_argument("--lambda-weight", type=float, default=0.5)
    p.add_argument("--epsilon", type=float, default=0.03)
    p.add_argument("--pop-perturb", type=float, default=0.0, help="Legacy binary perturbation added to group-0 target probability.")
    p.add_argument(
        "--target-distribution",
        choices=["pool", "balanced", "explicit"],
        default="pool",
        help="Target protected-group distribution pi used by Q-Bias and FAIR-AL.",
    )
    p.add_argument(
        "--target-pi",
        type=str,
        default=None,
        help="Comma-separated target distribution for --target-distribution explicit, e.g. 0.5,0.5.",
    )
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument(
        "--threshold-mode",
        choices=["fixed", "val_bacc", "val_youden"],
        default="fixed",
        help="How to choose decision thresholds for threshold-dependent metrics.",
    )
    p.add_argument("--output", type=str, default="results/experiment.csv")
    p.add_argument("--save-history", action="store_true")
    p.add_argument("--strict-batch", action="store_true", default=True, help="Backfill strategies that return too few samples.")
    # --- revision (stage-2) additions; all default off so historical behaviour is unchanged.
    p.add_argument(
        "--fal-cur-cluster-drop-sensitive",
        action="store_true",
        help="D4 sensitivity: neutralise the protected-attribute columns in the distance "
             "computations used by FAL-CUR (fair-cluster assignment and the within-cluster "
             "representativeness term), by setting them to their column mean. The predictor "
             "input, the group penalty, the splits, the initial set, the budget and the seeds "
             "are unchanged. Off by default, and bit-identical to the reported protocol when off.",
    )
    p.add_argument(
        "--drop-sensitive-inputs",
        action="store_true",
        help="Remove the protected-attribute columns from the model input (audit ablation). "
             "The acquisition strategies still use the attribute; splits and the initial set are unchanged.",
    )
    p.add_argument(
        "--intra-batch",
        action="store_true",
        help="FAIR-AL variants update the cumulative queried distribution after each pick inside a batch "
             "(sequential ablation). Historical behaviour = batch-prior scoring.",
    )
    p.add_argument(
        "--control-start-frac",
        type=float,
        default=0.0,
        help="Fraction of the query budget after which drift control is enabled (0 = control from the start). "
             "Before the switch point, plain uncertainty sampling is used.",
    )
    return p.parse_args()




def parse_target_pi(text: str | None, n_groups: int = 2) -> np.ndarray:
    if text is None:
        raise ValueError("--target-pi is required when --target-distribution explicit")
    vals = np.asarray([float(x.strip()) for x in text.split(",") if x.strip() != ""], dtype=float)
    if len(vals) != n_groups:
        raise ValueError(f"Expected {n_groups} target probabilities, got {len(vals)} from {text!r}")
    if np.any(vals < 0) or float(vals.sum()) <= 0:
        raise ValueError("Target probabilities must be non-negative and sum to a positive value")
    return vals / vals.sum()


def make_target_distribution(args: argparse.Namespace, sensitive_train: np.ndarray, n_groups: int = 2) -> np.ndarray:
    if args.target_distribution == "pool":
        target = distribution(sensitive_train, n_groups)
    elif args.target_distribution == "balanced":
        target = np.ones(n_groups, dtype=float) / n_groups
    elif args.target_distribution == "explicit":
        target = parse_target_pi(args.target_pi, n_groups=n_groups)
    else:  # pragma: no cover
        raise ValueError(f"Unknown target distribution mode: {args.target_distribution}")
    return perturb_distribution(target, args.pop_perturb)


def choose_threshold(y_val: np.ndarray, prob_val: np.ndarray, mode: str, fixed_threshold: float) -> float:
    if mode == "fixed":
        return float(fixed_threshold)
    from sklearn.metrics import balanced_accuracy_score, confusion_matrix

    prob_pos = prob_val[:, 1] if prob_val.ndim == 2 else prob_val
    candidates = np.unique(np.concatenate([[0.0, fixed_threshold, 1.0], prob_pos]))
    best_threshold = float(fixed_threshold)
    best_score = -np.inf
    for th in candidates:
        pred = (prob_pos >= th).astype(int)
        if mode == "val_bacc":
            score = float(balanced_accuracy_score(y_val, pred))
        elif mode == "val_youden":
            tn, fp, fn, tp = confusion_matrix(y_val, pred, labels=[0, 1]).ravel()
            sens = tp / (tp + fn + 1e-12)
            spec = tn / (tn + fp + 1e-12)
            score = float(sens + spec - 1.0)
        else:  # pragma: no cover
            raise ValueError(f"Unknown threshold mode: {mode}")
        if score > best_score or (np.isclose(score, best_score) and abs(float(th) - fixed_threshold) < abs(best_threshold - fixed_threshold)):
            best_score = score
            best_threshold = float(th)
    return best_threshold


def perturb_distribution(p: np.ndarray, amount: float) -> np.ndarray:
    if amount == 0:
        return p
    q = p.copy().astype(float)
    if len(q) != 2:
        return q
    q[0] = np.clip(q[0] + amount, 0.01, 0.99)
    q[1] = 1.0 - q[0]
    return q / q.sum()


def sanitize_selection(selected: np.ndarray, unlabeled_idx: np.ndarray, need: int, seed: int) -> np.ndarray:
    """Keep valid unique selected indices and fill from remaining unlabeled samples."""
    rng = np.random.default_rng(seed)
    unlabeled_set = set(map(int, unlabeled_idx))
    out: list[int] = []
    seen: set[int] = set()
    for idx in np.asarray(selected, dtype=int).ravel():
        ii = int(idx)
        if ii in unlabeled_set and ii not in seen:
            out.append(ii)
            seen.add(ii)
        if len(out) >= need:
            return np.asarray(out, dtype=int)
    remaining = np.array([int(i) for i in unlabeled_idx if int(i) not in seen], dtype=int)
    if len(remaining) > 0 and len(out) < need:
        extra = rng.choice(remaining, size=min(need - len(out), len(remaining)), replace=False)
        out.extend([int(i) for i in extra])
    return np.asarray(out[:need], dtype=int)


def run_one_strategy(args: argparse.Namespace, data, seed: int, strategy_name: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    strategy_code = canonical_strategy_name(strategy_name)
    strategy_public_name = public_strategy_name(strategy_code)
    set_seed(seed)
    X, y, sensitive = data.X, data.y, data.sensitive
    train_idx, val_idx, test_idx = train_val_test_split_indices(y, sensitive, seed=seed, test_size=args.test_size, val_size=args.val_size)
    init_idx, unlabeled_idx = initial_labeled_split(train_idx, y, sensitive, initial_size=args.initial_size, seed=seed)
    labeled_idx = init_idx.copy()
    queried_idx = np.array([], dtype=int)
    target_dist = make_target_distribution(args, sensitive[train_idx], 2)
    strategy = make_strategy(
        strategy_code,
        seed=seed,
        k_candidates=args.k_candidates,
        lambda_weight=args.lambda_weight,
        epsilon=args.epsilon,
        intra_batch=args.intra_batch,
        cluster_drop_cols=(
            list(data.sensitive_feature_cols)
            if getattr(args, "fal_cur_cluster_drop_sensitive", False) else None
        ),
    )
    # Delayed-control ablation: before the switch point the acquisition rule is plain
    # uncertainty sampling; afterwards it is the configured strategy.
    warmup_strategy = make_strategy(
        "uncertainty",
        seed=seed,
        k_candidates=args.k_candidates,
        lambda_weight=args.lambda_weight,
        epsilon=args.epsilon,
    )
    control_start_at = float(args.control_start_frac) * float(args.budget)
    rows: list[dict[str, Any]] = []
    history: list[dict[str, Any]] = []
    n_rounds = int(np.ceil(args.budget / args.batch_size))

    for round_id in tqdm(range(n_rounds + 1), desc=f"{data.name}-{args.backend}-{strategy_code}-seed{seed}", leave=False):
        backend = make_backend(
            backend=args.backend,
            classifier=args.classifier,
            seed=seed + round_id,
            modal_dims=data.modal_dims,
            epochs=args.epochs,
            device=args.device,
            lr=args.lr,
        )
        t0 = time.perf_counter()
        backend.fit(X[labeled_idx], y[labeled_idx])
        train_seconds = time.perf_counter() - t0

        prob_val = backend.predict_proba(X[val_idx])
        threshold_used = choose_threshold(y[val_idx], prob_val, args.threshold_mode, args.threshold)
        prob_test = backend.predict_proba(X[test_idx])
        metrics = evaluate_predictions(y[test_idx], prob_test, sensitive[test_idx], threshold=threshold_used)
        control_active = 1 if len(queried_idx) >= control_start_at else 0
        row: dict[str, Any] = {
            "dataset": data.name,
            "backend": args.backend,
            "classifier": args.classifier if args.backend == "sklearn" else "deepasd_style",
            "strategy": strategy_code,
            "method": strategy_public_name,
            "seed": seed,
            "round": round_id,
            "labeled_size": int(len(labeled_idx)),
            "queried_size": int(len(queried_idx)),
            "unlabeled_size": int(len(unlabeled_idx)),
            # Updated definition: q_bias is Total Variation distance, i.e. 0.5 * L1.
            "q_bias": q_bias(sensitive[queried_idx], target_dist),
            "q_bias_l1": q_bias_l1(sensitive[queried_idx], target_dist),
            "q_bias_definition": "TV=0.5*L1",
            "query_p_group0": float(distribution(sensitive[queried_idx], 2)[0]) if len(queried_idx) else float("nan"),
            "query_p_group1": float(distribution(sensitive[queried_idx], 2)[1]) if len(queried_idx) else float("nan"),
            "target_p_group0": float(target_dist[0]),
            "target_p_group1": float(target_dist[1]),
            "target_distribution_mode": args.target_distribution,
            "target_pi_arg": args.target_pi if args.target_pi is not None else "",
            "threshold_mode": args.threshold_mode,
            "threshold_used": float(threshold_used),
            "train_seconds": float(train_seconds),
            "sensitive_name": data.sensitive_name,
            "data_notes": data.notes,
            "n_features": int(X.shape[1]),
            "drop_sensitive_inputs": bool(args.drop_sensitive_inputs),
            "fal_cur_cluster_drop_cols": ",".join(
                str(c) for c in (data.sensitive_feature_cols
                                 if getattr(args, "fal_cur_cluster_drop_sensitive", False) else [])),
            "intra_batch": bool(args.intra_batch),
            "control_start_frac": float(args.control_start_frac),
            # Accounting: this table has one row per EVALUATION point (n_rounds + 1 = 13
            # rows for budget 120 / batch 10). Only n_rounds of them are followed by a
            # query event. control_state is the acquisition rule in effect at this
            # evaluation point (1 = drift control on, 0 = warm-up uncertainty rule);
            # per-query counters live in the history file, not here.
            "eval_index": int(round_id),
            "query_rounds_done": int(round_id),
            "control_state": int(control_active),
        }
        row.update(metrics)
        rows.append(row)

        if round_id == n_rounds or len(unlabeled_idx) == 0 or len(queried_idx) >= args.budget:
            break

        need = min(args.batch_size, args.budget - len(queried_idx), len(unlabeled_idx))
        t1 = time.perf_counter()
        active_strategy = strategy if control_active == 1 else warmup_strategy
        selected = active_strategy.select(
            backend=backend,
            X=X,
            y=y,
            sensitive=sensitive,
            unlabeled_idx=unlabeled_idx,
            queried_idx=queried_idx,
            batch_size=need,
            target_distribution=target_dist,
        )
        call_stats = getattr(active_strategy, "last_call_stats", None) or {}
        selected = sanitize_selection(selected, unlabeled_idx, need=need, seed=seed * 100000 + round_id)
        select_seconds = time.perf_counter() - t1
        if len(selected) != need:
            raise RuntimeError(f"Strategy {strategy_code} returned {len(selected)} samples but needed {need}.")
        history.append(
            {
                "dataset": data.name,
                "backend": args.backend,
                "classifier": args.classifier,
                "strategy": strategy_code,
                "method": strategy_public_name,
                "seed": seed,
                # One record per QUERY EVENT (n_rounds = 12 for budget 120 / batch 10).
                # The evaluation table has one extra row for the initial state.
                "query_round": int(round_id),
                "control_active": int(control_active),
                "constraint_dropped": int(call_stats.get("constraint_dropped", 0)),
                "feasible_fraction": float(call_stats.get("feasible_fraction", float("nan"))),
                "n_selected": int(len(selected)),
                "selected_indices": selected.tolist(),
                "selected_sensitive": sensitive[selected].tolist(),
                "selected_labels": y[selected].tolist(),
                "select_seconds": float(select_seconds),
            }
        )
        labeled_idx = np.concatenate([labeled_idx, selected])
        queried_idx = np.concatenate([queried_idx, selected])
        unlabeled_idx = np.setdiff1d(unlabeled_idx, selected, assume_unique=False)

    return rows, history


def main() -> None:
    args = parse_args()
    out_path = Path(args.output)
    ensure_dir(out_path.parent)
    if args.drop_sensitive_inputs and args.backend == "deepasd":
        raise SystemExit("--drop-sensitive-inputs is only defined for the sklearn backend")
    if getattr(args, "fal_cur_cluster_drop_sensitive", False) and args.drop_sensitive_inputs:
        raise SystemExit(
            "--fal-cur-cluster-drop-sensitive and --drop-sensitive-inputs are different "
            "interventions; run them separately so the reported ablation stays interpretable")
    data = load_dataset(
        args.data_dir,
        args.dataset,
        sensitive_index=args.sensitive_index,
        drop_sensitive_inputs=args.drop_sensitive_inputs,
    )
    strategies = [canonical_strategy_name(s) for s in parse_list_arg(args.strategies)]
    seeds = parse_int_list_arg(args.seeds)
    all_rows: list[dict[str, Any]] = []
    all_history: list[dict[str, Any]] = []
    for seed in seeds:
        for strategy_name in strategies:
            rows, history = run_one_strategy(args, data, seed, strategy_name)
            all_rows.extend(rows)
            all_history.extend(history)
            pd.DataFrame(all_rows).to_csv(out_path, index=False)
    pd.DataFrame(all_rows).to_csv(out_path, index=False)
    save_json(env_report(include_torch=args.backend == "deepasd"), out_path.with_suffix(".env.json"))
    if args.save_history:
        save_json(all_history, out_path.with_suffix(".history.json"))
    print(f"Saved results: {out_path}")
    print(f"Sensitive attribute: {data.sensitive_name}. {data.notes}")


if __name__ == "__main__":
    main()

