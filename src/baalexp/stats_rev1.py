
"""Revision statistics (stage 2).

Two separate jobs, deliberately not mixed:

1. ``historical-bh``: recompute the submitted manuscript's Table 5 exactly as the
   manuscript defines it (Benjamini-Hochberg inside each dataset x classifier x metric
   family). This is an erratum of the published table, not a new analysis.
2. ``new-holm``: pre-specified Holm families for the comparisons *added* during the
   revision (quota / stratified-uncertainty / constrained / dropped-attribute /
   intra-batch / delayed control). Holm is fixed before any new run is inspected.

Every output row carries: family_id, family_size, rank, raw p, adjusted p, and the
adjustment method, so a reader can rebuild the correction by hand. Zero-variance paired
differences are reported as ``degenerate_zero_variance`` with no t-test and no d_z
rather than as an infinite effect size.

Usage
    python -m baalexp.stats_rev1 historical-bh --results-dir <dir> --out <csv>
    python -m baalexp.stats_rev1 new-holm --input <csv> [<csv> ...] --out <csv>
"""
from __future__ import annotations

import argparse
import glob
import os
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats

EPS = 1e-12

# Metrics used for the manuscript's paired comparisons.
METRICS = ["balanced_accuracy", "q_bias", "auc", "eod_abs"]


# --------------------------------------------------------------------------- corrections
def _as_array(p: Iterable[float] | np.ndarray) -> np.ndarray:
    return np.asarray(list(p) if not isinstance(p, np.ndarray) else p, dtype=float)


def bh_fdr(p: Iterable[float] | np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg step-up adjustment (reference implementation)."""
    p = _as_array(p)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(1, n + 1, dtype=float)
    raw = p * n / ranks
    # enforce monotonicity at the sorted p order (step-up)
    sorted_idx = order
    adjusted_sorted = np.minimum.accumulate(raw[sorted_idx][::-1])[::-1]
    adjusted_sorted = np.clip(adjusted_sorted, 0.0, 1.0)
    out = np.empty(n, dtype=float)
    out[sorted_idx] = adjusted_sorted
    return out


def holm(p: Iterable[float] | np.ndarray) -> np.ndarray:
    """Holm step-down adjustment (reference implementation)."""
    p = _as_array(p)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(1, n + 1, dtype=float)
    raw = (n - ranks + 1.0) * p
    adjusted_sorted = np.maximum.accumulate(raw[order])
    adjusted_sorted = np.clip(adjusted_sorted, 0.0, 1.0)
    out = np.empty(n, dtype=float)
    out[order] = adjusted_sorted
    return out


def cross_check(p: Iterable[float] | np.ndarray, method: str) -> tuple[np.ndarray, str]:
    """Recompute with an independent implementation when statsmodels is available."""
    p_arr = _as_array(p)
    main = bh_fdr(p_arr) if method == "bh" else holm(p_arr)
    try:  # optional independent implementation
        from statsmodels.stats.multitest import multipletests  # type: ignore

        sm = multipletests(p_arr, method="fdr_bh" if method == "bh" else "holm")[1]
        worst = float(np.max(np.abs(sm - main)))
        return main, f"statsmodels max_abs_diff={worst:.3e}"
    except Exception as exc:  # pragma: no cover - optional dependency
        return main, f"statsmodels unavailable ({type(exc).__name__})"


# --------------------------------------------------------------------------- paired stats
@dataclass
class PairedResult:
    n: int
    mean_a: float
    mean_b: float
    delta: float
    sd: float
    t: float | None
    p: float | None
    dz: float | None
    note: str


def paired(a: np.ndarray, b: np.ndarray) -> PairedResult:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    n = len(a)
    if n == 0:
        return PairedResult(0, np.nan, np.nan, np.nan, np.nan, None, None, None, "no_pairs")
    diff = a - b
    delta = float(diff.mean())
    sd = float(diff.std(ddof=1)) if n > 1 else float("nan")
    if n < 2:
        return PairedResult(n, float(a.mean()), float(b.mean()), delta, sd, None, None, None, "too_few_pairs")
    if sd <= EPS:
        # Identical per-seed differences: a t-test is undefined and an "infinite" effect
        # size would be misleading. Report the constant difference instead.
        note = "degenerate_zero_variance" if abs(delta) > EPS else "identical_arms"
        return PairedResult(n, float(a.mean()), float(b.mean()), delta, sd, None, None, None, note)
    t, p = stats.ttest_rel(a, b)
    dz = delta / sd
    return PairedResult(n, float(a.mean()), float(b.mean()), delta, sd, float(t), float(p), float(dz), "ok")


# --------------------------------------------------------------------------- process Q-Bias
def process_qbias_stats(df: pd.DataFrame, strategy: str, budget: float = 120.0) -> dict:
    """Budget-axis integral of the process Q-Bias, with an explicit definition.

    Recorded points are EVALUATION points (u_i = queried_size_i / budget, Q_i = q_bias_i).
    The initial evaluation has an empty queried set, where q_bias is defined as 0 by
    convention only (no query has happened yet, so there is no composition to measure).

    Main definition (reported in the manuscript):
        I = (1 / (u_end - u_start)) * integral_{u_start}^{u_end} Q(u) du
    with trapezoidal integration over the recorded points, u_start = first point with
    queried_size > 0, u_end = last point. The normalisation makes the value a
    budget-averaged Q-Bias on the queried range, so it is comparable across arms.

    Sensitivity: I_from_zero includes the u = 0 convention point (Q = 0). The two values
    bracket the effect of the start convention and both are reported.
    """
    d = df[df["strategy"] == strategy].sort_values("round")
    u = d["queried_size"].to_numpy(dtype=float) / budget
    q = d["q_bias"].to_numpy(dtype=float)
    q = np.where(np.isfinite(q), q, 0.0)
    if len(u) < 2:
        return dict(strategy=strategy, i_main=float("nan"), i_from_zero=float("nan"),
                    u_start=float("nan"), u_end=float("nan"), n_points=len(u))
    # main: from the first non-empty queried set
    start = int(np.argmax(u > 0))
    seg_u, seg_q = u[start:], q[start:]
    area = float(np.trapezoid(seg_q, seg_u)) if len(seg_u) > 1 else 0.0
    span = float(seg_u[-1] - seg_u[0])
    i_main = area / span if span > 0 else float("nan")
    # sensitivity: include the u = 0 convention point
    area0 = float(np.trapezoid(q, u))
    i_from_zero = area0 / float(u[-1]) if u[-1] > 0 else float("nan")
    return dict(strategy=strategy, i_main=i_main, i_from_zero=i_from_zero,
                u_start=float(seg_u[0]), u_end=float(seg_u[-1]), n_points=len(u))


# --------------------------------------------------------------------------- data access
def final_round(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["round"] == df.groupby(["strategy", "seed"])["round"].transform("max")]


def wide(df: pd.DataFrame, value: str) -> pd.DataFrame:
    f = final_round(df)
    return f.pivot_table(index="seed", columns="strategy", values=value)


def build_rows(df: pd.DataFrame, comparisons: list[tuple[str, str]], family_prefix: str) -> list[dict]:
    """Comparisons are evaluated inside dataset x classifier, as the manuscript states."""
    rows: list[dict] = []
    for (dataset, classifier), g in df.groupby(["dataset", "classifier"]):
        for metric in METRICS:
            if metric not in g.columns:
                continue
            w = wide(g, metric)
            for a, b in comparisons:
                if a not in w.columns or b not in w.columns:
                    continue
                res = paired(w[a].to_numpy(), w[b].to_numpy())
                rows.append(
                    dict(
                        family_id=f"{family_prefix}|{dataset}|{classifier}|{metric}",
                        dataset=dataset,
                        classifier=classifier,
                        metric=metric,
                        contrast=f"{a} vs {b}",
                        strategy=a,
                        baseline=b,
                        n_pairs=res.n,
                        mean_strategy=res.mean_a,
                        mean_baseline=res.mean_b,
                        delta=res.delta,
                        sd_diff=res.sd,
                        cohen_dz=res.dz,
                        paired_t=res.t,
                        p_raw=res.p,
                        note=res.note,
                    )
                )
    return rows


def adjust(rows: list[dict], method: str) -> list[dict]:
    out: list[dict] = []
    df = pd.DataFrame(rows)
    if df.empty:
        return rows
    for fam, g in df.groupby("family_id", sort=True):
        valid = g["p_raw"].notna()
        p_vals = g.loc[valid, "p_raw"].to_numpy(dtype=float)
        adjusted, check = cross_check(p_vals, method) if len(p_vals) else (np.array([]), "empty")
        ranks = np.empty(len(p_vals), dtype=int)
        if len(p_vals):
            order = np.argsort(p_vals)
            ranks[order] = np.arange(1, len(p_vals) + 1)
        adj_map = dict(zip(g.loc[valid].index, adjusted))
        rank_map = dict(zip(g.loc[valid].index, ranks))
        for idx, row in g.iterrows():
            row = dict(row)
            row["family_size"] = int(len(p_vals))
            row["rank_within_family"] = int(rank_map[idx]) if idx in rank_map else 0
            row["p_adjusted"] = float(adj_map[idx]) if idx in adj_map else float("nan")
            row["adjust_method"] = method
            row["adjust_crosscheck"] = check
            out.append(row)
    return out


# --------------------------------------------------------------------------- CLI
def cmd_historical(args) -> None:
    files = sorted(glob.glob(os.path.join(args.results_dir, "*_core_*.csv")))
    if not files:
        raise SystemExit(f"no core result CSVs under {args.results_dir}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    # The family is "every strategy compared with uncertainty inside one
    # dataset x classifier x metric cell", derived from the data rather than hard-coded,
    # so the family size can be checked against the manuscript's own definition.
    strategies = sorted(set(df["strategy"].unique()) - {"uncertainty"})
    comparisons = [(s, "uncertainty") for s in strategies]
    rows = adjust(build_rows(df, comparisons, "hist"), "bh")
    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False)
    print(f"[historical-bh] wrote {args.out}  rows={len(out)}  families={out['family_id'].nunique()}")
    focus = out[(out.metric == "q_bias") & (out.contrast.isin(["baal_weighted vs uncertainty"])) & (out.classifier.isin(["lr", "svm"]))]
    cols = ["dataset", "classifier", "delta", "cohen_dz", "p_raw", "p_adjusted", "rank_within_family", "family_size", "note"]
    print(focus[cols].to_string(index=False))


def cmd_new(args) -> None:
    frames = []
    for path in args.input:
        d = pd.read_csv(path)
        d["_source"] = os.path.basename(path)
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    comparisons = [(s, "uncertainty") for s in ["baal", "baal_weighted", "baal_hard", "quota", "stratified_uncertainty"]]
    rows = adjust(build_rows(df, comparisons, "rev1"), "holm")
    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False)
    print(f"[new-holm] wrote {args.out}  rows={len(out)}  families={out['family_id'].nunique()}")


def cmd_manifest(args) -> None:
    """Evaluate exactly the comparisons declared in comparisons_manifest.csv."""
    man = pd.read_csv(args.manifest)
    frames = []
    for path in args.input:
        d = pd.read_csv(path)
        d["_source"] = os.path.basename(path)
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    rows: list[dict] = []
    for _, m in man.iterrows():
        sub = df[(df["dataset"] == m["dataset"]) & (df["classifier"] == m["classifier"])]
        if sub.empty:
            rows.append({**m.to_dict(), "status": "missing_setting"})
            continue
        metric = m["metric"]
        a, b = m["strategy_a"], m["strategy_b"]
        src_a = m["source_a"] if "source_a" in man.columns else None
        src_b = m["source_b"] if "source_b" in man.columns else None
        # Each side may live in a different results file (kept-vs-dropped, batch-prior vs
        # intra-batch, control fraction 0 vs 0.5 ...). Align strictly by seed.
        def side_series(strategy: str, src):
            s = sub if (src is None or pd.isna(src)) else sub[sub["_source"] == src]
            if s.empty or strategy not in set(s["strategy"]):
                return None
            return final_round(s)[final_round(s)["strategy"] == strategy].set_index("seed")[metric]

        sa, sb = side_series(a, src_a), side_series(b, src_b)
        if sa is None or sb is None:
            rows.append({**m.to_dict(), "status": f"missing_arm({a}|{b})"})
            continue
        common = sa.index.intersection(sb.index)
        if len(common) == 0:
            rows.append({**m.to_dict(), "status": "no_common_seeds"})
            continue
        res = paired(sa.loc[common].to_numpy(), sb.loc[common].to_numpy())
        rows.append({**m.to_dict(), "status": "ok", "n_pairs": res.n,
                     "mean_a": res.mean_a, "mean_b": res.mean_b, "delta": res.delta,
                     "sd_diff": res.sd, "cohen_dz": res.dz, "paired_t": res.t,
                     "p_raw": res.p, "note": res.note})
    out = pd.DataFrame(rows)
    # adjust inside each declared family, method taken from the manifest
    out["p_adjusted"] = np.nan
    out["family_size_evaluated"] = 0
    out["rank_within_family"] = 0
    for fam, g in out.groupby("family_id"):
        method = str(g["adjustment"].iloc[0]).lower()
        valid = g["p_raw"].notna()
        p = g.loc[valid, "p_raw"].to_numpy(dtype=float)
        if not len(p):
            continue
        adj = bh_fdr(p) if method == "bh" else holm(p)
        order = np.argsort(p)
        ranks = np.empty(len(p), dtype=int)
        ranks[order] = np.arange(1, len(p) + 1)
        out.loc[g.loc[valid].index, "p_adjusted"] = adj
        out.loc[g.loc[valid].index, "rank_within_family"] = ranks
        out.loc[g.loc[valid].index, "family_size_evaluated"] = len(p)
    out.to_csv(args.out, index=False)
    print(f"[manifest] wrote {args.out} rows={len(out)} families={out['family_id'].nunique()}")
    print(out.groupby(["family_id", "metric"]).size().to_string())


def main() -> None:
    p = argparse.ArgumentParser(description="Revision statistics (BH erratum + pre-specified Holm)")
    sub = p.add_subparsers(dest="cmd", required=True)

    h = sub.add_parser("historical-bh", help="recompute submitted Table 5 under its own BH definition")
    h.add_argument("--results-dir", required=True)
    h.add_argument("--out", required=True)
    h.set_defaults(func=cmd_historical)

    n = sub.add_parser("new-holm", help="Holm families for revision-added comparisons")
    n.add_argument("--input", nargs="+", required=True)
    n.add_argument("--out", required=True)
    n.set_defaults(func=cmd_new)

    m = sub.add_parser("manifest", help="evaluate the comparisons declared in comparisons_manifest.csv")
    m.add_argument("--manifest", required=True)
    m.add_argument("--input", nargs="+", required=True)
    m.add_argument("--out", required=True)
    m.set_defaults(func=cmd_manifest)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
