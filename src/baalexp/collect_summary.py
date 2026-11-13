
from __future__ import annotations

import argparse
from pathlib import Path
import glob
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon

KEYS = ["dataset", "backend", "classifier", "strategy"]
DEFAULT_METRICS = [
    "accuracy",
    "balanced_accuracy",
    "auc",
    "q_bias",
    "eod_abs",
    "fpr_gap_abs",
    "fnr_gap_abs",
    "group0_sensitivity",
    "group1_sensitivity",
    "group0_fpr",
    "group1_fpr",
    "group0_fnr",
    "group1_fnr",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Collect raw result CSVs into manuscript-ready summary tables")
    p.add_argument("--inputs", nargs="+", required=True, help="Raw fairal-run CSV files, e.g. results/*_fixed.csv")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--baseline", default="uncertainty")
    p.add_argument("--metrics", default=",".join(DEFAULT_METRICS))
    return p.parse_args()


def read_inputs(paths: list[str]) -> pd.DataFrame:
    frames = []
    for pattern in paths:
        matched = [Path(x) for x in sorted(glob.glob(pattern))] if any(ch in pattern for ch in "*?[") else [Path(pattern)]
        for p in matched:
            if p.exists() and p.suffix == ".csv":
                df = pd.read_csv(p)
                df["source_file"] = str(p)
                frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No CSV files matched: {paths}")
    return pd.concat(frames, ignore_index=True)


def final_rows(df: pd.DataFrame) -> pd.DataFrame:
    idx = df.groupby(["dataset", "backend", "classifier", "strategy", "seed", "source_file"])["round"].idxmax()
    return df.loc[idx].reset_index(drop=True)


def mean_std_table(final: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, g in final.groupby(KEYS):
        row = dict(zip(KEYS, keys))
        row["n_seeds"] = int(g["seed"].nunique())
        for m in metrics:
            if m in g:
                row[f"{m}_mean"] = float(g[m].mean())
                row[f"{m}_std"] = float(g[m].std(ddof=1)) if len(g) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows).sort_values(KEYS).reset_index(drop=True)


def safe_wilcoxon(x: np.ndarray, y: np.ndarray) -> float:
    diff = np.asarray(x) - np.asarray(y)
    if len(diff) < 2 or np.allclose(diff, 0):
        return float("nan")
    try:
        return float(wilcoxon(diff, zero_method="wilcox", alternative="two-sided").pvalue)
    except Exception:
        return float("nan")


def tests_vs_baseline(final: pd.DataFrame, baseline: str, metrics: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, g in final.groupby(["dataset", "backend", "classifier"]):
        b = g[g["strategy"] == baseline].set_index("seed")
        if b.empty:
            continue
        for strategy, sg0 in g.groupby("strategy"):
            if strategy == baseline:
                continue
            sg = sg0.set_index("seed")
            common = sorted(set(b.index) & set(sg.index))
            if len(common) < 2:
                continue
            for m in metrics:
                if m not in g:
                    continue
                x = sg.loc[common, m].to_numpy(dtype=float)
                y = b.loc[common, m].to_numpy(dtype=float)
                try:
                    t_p = float(ttest_rel(x, y, nan_policy="omit").pvalue)
                except Exception:
                    t_p = float("nan")
                w_p = safe_wilcoxon(x, y)
                rows.append(
                    {
                        "dataset": keys[0],
                        "backend": keys[1],
                        "classifier": keys[2],
                        "strategy": strategy,
                        "baseline": baseline,
                        "metric": m,
                        "n_pairs": len(common),
                        "mean_strategy": float(np.nanmean(x)),
                        "mean_baseline": float(np.nanmean(y)),
                        "delta_strategy_minus_baseline": float(np.nanmean(x - y)),
                        "paired_t_p": t_p,
                        "wilcoxon_p": w_p,
                    }
                )
    return pd.DataFrame(rows).sort_values(["dataset", "backend", "classifier", "strategy", "metric"]).reset_index(drop=True)


def wide_pvalue_table(tests: pd.DataFrame) -> pd.DataFrame:
    if tests.empty:
        return tests
    keep = ["balanced_accuracy", "q_bias", "eod_abs", "group0_sensitivity", "group1_sensitivity"]
    t = tests[tests["metric"].isin(keep)].copy()
    pieces = []
    for value in ["delta_strategy_minus_baseline", "paired_t_p", "wilcoxon_p"]:
        w = t.pivot_table(
            index=["dataset", "backend", "classifier", "strategy", "baseline"],
            columns="metric",
            values=value,
            aggfunc="first",
        )
        w.columns = [f"{c}_{value}" for c in w.columns]
        pieces.append(w)
    return pd.concat(pieces, axis=1).reset_index()


def main() -> None:
    args = parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    df = read_inputs(args.inputs)
    final = final_rows(df)
    summary = mean_std_table(final, metrics)
    tests = tests_vs_baseline(final, args.baseline, metrics)
    wide = wide_pvalue_table(tests)

    df.to_csv(out / "all_iterations.csv", index=False)
    final.to_csv(out / "all_final_rows.csv", index=False)
    summary.to_csv(out / "all_summary_mean_std.csv", index=False)
    tests.to_csv(out / "all_tests_vs_baseline.csv", index=False)
    wide.to_csv(out / "all_tests_vs_baseline_wide.csv", index=False)
    print(f"Saved collected summary to {out}")


if __name__ == "__main__":
    main()

