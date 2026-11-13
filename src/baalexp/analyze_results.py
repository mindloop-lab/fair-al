
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import ttest_rel

from .metrics import clinical_impact_per_1000
from .utils import ensure_dir


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze FAIR-AL experiment CSV files")
    p.add_argument("--input", type=str, required=True, help="CSV file or comma-separated CSV files")
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--baseline", type=str, default="uncertainty")
    return p.parse_args()


def load_inputs(paths: str) -> pd.DataFrame:
    frames = [pd.read_csv(p.strip()) for p in paths.split(",") if p.strip()]
    return pd.concat(frames, ignore_index=True)




def benjamini_hochberg(p_values: pd.Series) -> pd.Series:
    p = pd.to_numeric(p_values, errors="coerce")
    out = pd.Series(np.nan, index=p_values.index, dtype=float)
    valid = p.dropna().sort_values()
    m = len(valid)
    if m == 0:
        return out
    adjusted = valid * m / np.arange(1, m + 1)
    adjusted = np.minimum.accumulate(adjusted.iloc[::-1]).iloc[::-1].clip(upper=1.0)
    out.loc[adjusted.index] = adjusted.values
    return out


def paired_cohen_dz(delta: pd.Series) -> float:
    d = pd.to_numeric(delta, errors="coerce").dropna()
    if len(d) < 2:
        return float("nan")
    sd = float(d.std(ddof=1))
    if sd <= 1e-12:
        return 0.0
    return float(d.mean() / sd)


def final_rows(df: pd.DataFrame) -> pd.DataFrame:
    idx = df.groupby(["dataset", "backend", "classifier", "strategy", "seed"])["round"].idxmax()
    return df.loc[idx].reset_index(drop=True)


def mean_std_table(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    rows = []
    for keys, g in df.groupby(["dataset", "backend", "classifier", "strategy"]):
        row = dict(zip(["dataset", "backend", "classifier", "strategy"], keys))
        for m in metrics:
            if m in g:
                row[f"{m}_mean"] = g[m].mean()
                row[f"{m}_std"] = g[m].std()
        rows.append(row)
    return pd.DataFrame(rows)


def paired_tests(final: pd.DataFrame, baseline: str, metrics: list[str]) -> pd.DataFrame:
    rows = []
    for keys, g in final.groupby(["dataset", "backend", "classifier"]):
        base = g[g["strategy"] == baseline].set_index("seed")
        if base.empty:
            continue
        for strategy, sg0 in g.groupby("strategy"):
            if strategy == baseline:
                continue
            sg = sg0.set_index("seed")
            common = sorted(set(base.index) & set(sg.index))
            if len(common) < 2:
                continue
            for m in metrics:
                if m not in g:
                    continue
                stat, p = ttest_rel(sg.loc[common, m], base.loc[common, m], nan_policy="omit")
                rows.append(
                    {
                        "dataset": keys[0],
                        "backend": keys[1],
                        "classifier": keys[2],
                        "strategy": strategy,
                        "baseline": baseline,
                        "metric": m,
                        "mean_strategy": sg.loc[common, m].mean(),
                        "mean_baseline": base.loc[common, m].mean(),
                        "delta_strategy_minus_baseline": sg.loc[common, m].mean() - base.loc[common, m].mean(),
                        "p_value": p,
                    }
                )
    return pd.DataFrame(rows)


def plot_curves(df: pd.DataFrame, out_dir: Path, metric: str) -> None:
    if metric not in df:
        return
    for keys, g in df.groupby(["dataset", "backend", "classifier"]):
        plt.figure(figsize=(7, 5))
        for strategy, sg in g.groupby("strategy"):
            curve = sg.groupby("queried_size")[metric].agg(["mean", "std"]).reset_index()
            plt.plot(curve["queried_size"], curve["mean"], marker="o", label=strategy)
            if len(curve) > 1:
                lo = curve["mean"] - curve["std"].fillna(0)
                hi = curve["mean"] + curve["std"].fillna(0)
                plt.fill_between(curve["queried_size"], lo, hi, alpha=0.15)
        plt.xlabel("Queried labels")
        plt.ylabel(metric)
        plt.title(" / ".join(map(str, keys)))
        plt.legend(fontsize=8)
        plt.tight_layout()
        filename = f"curve_{keys[0]}_{keys[1]}_{keys[2]}_{metric}.pdf".replace("/", "-")
        plt.savefig(out_dir / filename)
        plt.close()


def clinical_impact(final: pd.DataFrame, baseline: str) -> pd.DataFrame:
    rows = []
    for keys, g in final.groupby(["dataset", "backend", "classifier", "seed"]):
        base_rows = g[g["strategy"] == baseline]
        if base_rows.empty:
            continue
        base = base_rows.iloc[0].to_dict()
        for _, row in g.iterrows():
            if row["strategy"] == baseline:
                continue
            impact = clinical_impact_per_1000(row.to_dict(), base)
            impact.update({"dataset": keys[0], "backend": keys[1], "classifier": keys[2], "seed": keys[3], "strategy": row["strategy"], "baseline": baseline})
            rows.append(impact)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(args.out_dir)
    df = load_inputs(args.input)
    final = final_rows(df)
    metrics = [
        "accuracy", "balanced_accuracy", "auc", "sensitivity", "specificity", "q_bias", "eod_abs", "fpr_gap_abs", "fnr_gap_abs",
        "group0_sensitivity", "group1_sensitivity", "group0_fpr", "group1_fpr", "group0_fnr", "group1_fnr",
    ]
    summary = mean_std_table(final, metrics)
    tests = paired_tests(final, args.baseline, ["balanced_accuracy", "q_bias", "eod_abs", "group0_sensitivity", "group1_sensitivity", "fpr_gap_abs", "fnr_gap_abs"])
    impact = clinical_impact(final, args.baseline)
    final.to_csv(out_dir / "final_rows.csv", index=False)
    summary.to_csv(out_dir / "summary_mean_std.csv", index=False)
    tests.to_csv(out_dir / "paired_tests_vs_baseline.csv", index=False)
    impact.to_csv(out_dir / "clinical_impact_per_1000_vs_baseline.csv", index=False)
    for metric in ["balanced_accuracy", "q_bias", "eod_abs", "group0_sensitivity", "group1_sensitivity"]:
        plot_curves(df, out_dir, metric)
    print(f"Saved analysis to {out_dir}")


if __name__ == "__main__":
    main()

