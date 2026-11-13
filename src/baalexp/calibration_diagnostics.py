from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

from .backends import make_backend
from .data import initial_labeled_split, load_dataset, train_val_test_split_indices
from .naming import public_strategy_name


def expected_calibration_error(y_true: np.ndarray, prob_pos: np.ndarray, n_bins: int = 10) -> float:
    y_true = np.asarray(y_true).astype(int)
    prob_pos = np.asarray(prob_pos, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        if hi == 1.0:
            mask = (prob_pos >= lo) & (prob_pos <= hi)
        else:
            mask = (prob_pos >= lo) & (prob_pos < hi)
        if not mask.any():
            continue
        conf = float(prob_pos[mask].mean())
        acc = float(y_true[mask].mean())
        ece += float(mask.mean()) * abs(acc - conf)
    return float(ece)


def reliability_points(y_true: np.ndarray, prob_pos: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    rows = []
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    for b, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
        mask = ((prob_pos >= lo) & (prob_pos <= hi)) if hi == 1.0 else ((prob_pos >= lo) & (prob_pos < hi))
        if not mask.any():
            continue
        rows.append({"bin": b, "lo": lo, "hi": hi, "confidence": float(prob_pos[mask].mean()), "fraction_positive": float(y_true[mask].mean()), "n": int(mask.sum())})
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Replay final labelled sets and compute probability calibration diagnostics")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--inputs", nargs="+", required=True, help="CSV result files or glob patterns")
    p.add_argument("--out-dir", default="results/enhanced_artifacts/calibration")
    p.add_argument("--strategies", default="uncertainty,baal_weighted,baal,quota,stratified_uncertainty")
    p.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    p.add_argument("--n-bins", type=int, default=10)
    return p.parse_args()


def matched_paths(patterns: list[str]) -> list[Path]:
    out = []
    for pat in patterns:
        out.extend(Path(x) for x in sorted(glob.glob(pat)))
    return [p for p in out if p.exists() and p.suffix == ".csv"]


def load_history(path: Path) -> list[dict]:
    hp = path.with_suffix(".history.json")
    if not hp.exists():
        return []
    return json.loads(hp.read_text())


def infer_from_csv(path: Path, df: pd.DataFrame) -> tuple[str, str, int | None]:
    dataset = str(df["dataset"].dropna().iloc[0])
    classifier = str(df["classifier"].dropna().iloc[0]) if "classifier" in df else "lr"
    sensitive_index = 46 if dataset == "ABIDE_B" else None
    return dataset, classifier, sensitive_index


def _display_dataset(x: object) -> str:
    return str(x).replace("_", "-")


def _display_classifier(x: object) -> str:
    return str(x).upper()


def _mean_std_text(mean: object, std: object, digits: int = 3) -> str:
    try:
        m = float(mean)
        s = float(std)
    except Exception:
        return "--"
    if not np.isfinite(m):
        return "--"
    if not np.isfinite(s):
        s = 0.0
    return f"{m:.{digits}f} $\\pm$ {s:.{digits}f}"


def write_calibration_latex_table(summary: pd.DataFrame, out_path: Path) -> None:
    """Write a publication-style calibration table.

    Input summary contains internal columns:
        dataset, classifier, strategy, method,
        brier_mean, brier_std, ece_mean, ece_std, n_seeds

    Output table keeps only:
        Dataset, Classifier, Method, Brier score, ECE, Seeds
    """
    if summary.empty:
        return

    display = summary.copy()

    # Use only the public method name; drop internal strategy names.
    display["Dataset"] = display["dataset"].map(_display_dataset)
    display["Classifier"] = display["classifier"].map(_display_classifier)
    display["Method"] = display["method"].map(str)

    display["Brier score"] = [
        _mean_std_text(m, s)
        for m, s in zip(display["brier_mean"], display["brier_std"])
    ]
    display["ECE"] = [
        _mean_std_text(m, s)
        for m, s in zip(display["ece_mean"], display["ece_std"])
    ]
    display["Seeds"] = display["n_seeds"].astype(int)

    # Optional: keep a stable and readable method order.
    method_order = {
        "Uncertainty": 0,
        "Quota sampling": 1,
        "Stratified uncertainty": 2,
        "FAIR-AL-Pareto": 3,
        "FAIR-AL-Weighted": 4,
    }
    display["method_order"] = display["Method"].map(method_order).fillna(99)

    display = display.sort_values(
        ["Dataset", "Classifier", "method_order", "Method"]
    )

    cols = ["Dataset", "Classifier", "Method", "Brier score", "ECE", "Seeds"]

    tex = r"""
\begin{table}[t]
\centering
\caption{Probability calibration diagnostics at the final active-learning round. Brier score and expected calibration error (ECE) are reported as mean $\pm$ standard deviation across seeds.}
\label{tab:calibration_diagnostics}
\small
\setlength{\tabcolsep}{4pt}
\renewcommand{\arraystretch}{1.10}
\begin{tabular}{lllccc}
\toprule
Dataset & Classifier & Method & Brier score & ECE & Seeds \\
\midrule
""".lstrip()

    for _, r in display[cols].iterrows():
        tex += (
            f"{r['Dataset']} & {r['Classifier']} & {r['Method']} & "
            f"{r['Brier score']} & {r['ECE']} & {r['Seeds']} \\\\\n"
        )

    tex += r"""\bottomrule
\end{tabular}
\end{table}
"""

    out_path.write_text(tex, encoding="utf-8")

def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    keep_strategies = {s.strip() for s in args.strategies.split(",") if s.strip()}
    keep_seeds = {int(s.strip()) for s in args.seeds.split(",") if s.strip()}
    rows = []
    rel_frames = []
    for path in matched_paths(args.inputs):
        df = pd.read_csv(path)
        if df.empty:
            continue
        dataset, classifier, sensitive_index = infer_from_csv(path, df)
        data = load_dataset(args.data_dir, dataset, sensitive_index=sensitive_index)
        hist = pd.DataFrame(load_history(path))
        if hist.empty:
            continue
        for (strategy, seed), gh in hist.groupby(["strategy", "seed"]):
            if strategy not in keep_strategies or int(seed) not in keep_seeds:
                continue
            train_idx, val_idx, test_idx = train_val_test_split_indices(data.y, data.sensitive, seed=int(seed))
            init_idx, _ = initial_labeled_split(train_idx, data.y, data.sensitive, initial_size=40, seed=int(seed))
            selected = []
            for _, r in gh.sort_values("round").iterrows():
                vals = r.get("selected_indices", [])
                if isinstance(vals, str):
                    vals = json.loads(vals)
                selected.extend(int(x) for x in vals)
            labeled = np.unique(np.concatenate([init_idx, np.asarray(selected, dtype=int)]))
            backend = make_backend("sklearn", classifier=classifier, seed=int(seed) + 999, modal_dims=data.modal_dims)
            backend.fit(data.X[labeled], data.y[labeled])
            prob = backend.predict_proba(data.X[test_idx])[:, 1]
            brier = float(brier_score_loss(data.y[test_idx], prob))
            ece = expected_calibration_error(data.y[test_idx], prob, n_bins=args.n_bins)
            rows.append({"source_file": path.name, "dataset": dataset, "classifier": classifier, "strategy": strategy, "method": public_strategy_name(strategy), "seed": int(seed), "n_labeled": int(len(labeled)), "brier": brier, "ece": ece})
            rp = reliability_points(data.y[test_idx], prob, n_bins=args.n_bins)
            if not rp.empty:
                rp["dataset"] = dataset
                rp["classifier"] = classifier
                rp["strategy"] = strategy
                rp["method"] = public_strategy_name(strategy)
                rp["seed"] = int(seed)
                rel_frames.append(rp)
    cal = pd.DataFrame(rows)
    if cal.empty:
        raise SystemExit("No calibration rows generated. Check that inputs have matching .history.json files.")
    cal.to_csv(out_dir / "calibration_by_seed.csv", index=False)
    # summary = cal.groupby(["dataset", "classifier", "strategy", "method"], as_index=False).agg(brier_mean=("brier", "mean"), brier_std=("brier", "std"), ece_mean=("ece", "mean"), ece_std=("ece", "std"), n_seeds=("seed", "nunique"))
    # summary.to_csv(out_dir / "calibration_summary.csv", index=False)
    # tex = summary.to_latex(index=False, escape=True, float_format=lambda x: f"{x:.3f}", caption="Probability calibration diagnostics at the final active-learning round.", label="tab:calibration_diagnostics")
    # (out_dir / "table_calibration_diagnostics.tex").write_text(tex, encoding="utf-8")
    #
    summary = (
        cal.groupby(["dataset", "classifier", "strategy", "method"], as_index=False)
        .agg(
            brier_mean=("brier", "mean"),
            brier_std=("brier", "std"),
            ece_mean=("ece", "mean"),
            ece_std=("ece", "std"),
            n_seeds=("seed", "nunique"),
        )
    )

    summary.to_csv(out_dir / "calibration_summary.csv", index=False)

    write_calibration_latex_table(
        summary,
        out_dir / "table_calibration_diagnostics.tex",
    )

    if rel_frames:
        rel = pd.concat(rel_frames, ignore_index=True)
        rel.to_csv(out_dir / "reliability_points_by_seed.csv", index=False)
        agg = rel.groupby(["dataset", "classifier", "method", "bin"], as_index=False).agg(confidence=("confidence", "mean"), fraction_positive=("fraction_positive", "mean"), n=("n", "sum"))
        for (dataset, classifier), g in agg.groupby(["dataset", "classifier"]):
            plt.figure(figsize=(5.2, 5.0))
            plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1.0)
            for method, sg in g.groupby("method"):
                plt.plot(sg["confidence"], sg["fraction_positive"], marker="o", label=method)
            plt.xlabel("Mean predicted probability")
            plt.ylabel("Observed positive fraction")
            plt.title(f"Reliability diagram: {dataset.replace('_','-')} / {classifier.upper()}")
            plt.legend(fontsize=7)
            plt.tight_layout()
            stem = f"reliability_{dataset}_{classifier}"
            plt.savefig(out_dir / f"{stem}.pdf")
            plt.savefig(out_dir / f"{stem}.png", dpi=240)
            plt.close()


if __name__ == "__main__":
    main()
