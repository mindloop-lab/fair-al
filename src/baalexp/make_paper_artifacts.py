
from __future__ import annotations

import argparse
from pathlib import Path
import glob

import matplotlib.pyplot as plt
import pandas as pd

from .naming import PUBLIC_ORDER, public_strategy_name

PREFERRED_INTERNAL = [
    "random",
    "uncertainty",
    "coreset",
    "badge",
    "baal",
    "baal_weighted",
    "baal_hard",
    "baal_random_pareto",
    "heuristic",
    "reweighting",
]
PAPER_METRICS = ["balanced_accuracy", "auc", "q_bias", "eod_abs", "group0_sensitivity", "group1_sensitivity"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create manuscript tables, figure data, and PDF figures from fairal-run CSV files")
    p.add_argument("--inputs", nargs="+", required=True, help="Raw fairal-run CSV files or glob patterns")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--baseline", default="uncertainty")
    p.add_argument("--prefix", default="fairal")
    return p.parse_args()


def read_inputs(paths: list[str]) -> pd.DataFrame:
    frames = []
    for pat in paths:
        matched = [Path(x) for x in sorted(glob.glob(pat))] if any(ch in pat for ch in "*?[") else [Path(pat)]
        for p in matched:
            if p.exists() and p.suffix == ".csv":
                df = pd.read_csv(p)
                df["source_file"] = str(p)
                frames.append(df)
    if not frames:
        raise FileNotFoundError(paths)
    out = pd.concat(frames, ignore_index=True)
    out["strategy_public"] = out["strategy"].map(public_strategy_name)
    return out


def final_rows(df: pd.DataFrame) -> pd.DataFrame:
    idx = df.groupby(["dataset", "backend", "classifier", "strategy", "seed", "source_file"])["round"].idxmax()
    return df.loc[idx].reset_index(drop=True)


def fmt_mean_std(mean: float, std: float, digits: int = 3) -> str:
    if pd.isna(mean):
        return ""
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def make_main_table(final: pd.DataFrame, out: Path, prefix: str) -> pd.DataFrame:
    rows = []
    for keys, g in final.groupby(["dataset", "backend", "classifier", "strategy", "strategy_public"]):
        row = dict(zip(["dataset", "backend", "classifier", "strategy", "strategy_public"], keys))
        row["n_seeds"] = int(g["seed"].nunique())
        for m in PAPER_METRICS:
            if m in g:
                row[m] = fmt_mean_std(g[m].mean(), g[m].std(ddof=1))
                row[f"{m}_mean"] = g[m].mean()
                row[f"{m}_std"] = g[m].std(ddof=1)
        rows.append(row)
    table = pd.DataFrame(rows)
    table["strategy_public"] = pd.Categorical(table["strategy_public"], PUBLIC_ORDER, ordered=True)
    table = table.sort_values(["dataset", "classifier", "strategy_public"]).reset_index(drop=True)
    table.to_csv(out / f"{prefix}_main_table.csv", index=False)
    display_cols = ["dataset", "classifier", "strategy_public", "n_seeds"] + PAPER_METRICS
    table[display_cols].rename(columns={"strategy_public": "method"}).to_markdown(out / f"{prefix}_main_table.md", index=False)
    with (out / f"{prefix}_main_table.tex").open("w", encoding="utf-8") as f:
        f.write(table[display_cols].rename(columns={"strategy_public": "method"}).to_latex(index=False, escape=False))
    return table


def aggregate_curves(df: pd.DataFrame, out: Path, prefix: str) -> pd.DataFrame:
    metrics = ["balanced_accuracy", "q_bias", "eod_abs", "group0_sensitivity", "group1_sensitivity"]
    rows = []
    for keys, g in df.groupby(["dataset", "backend", "classifier", "strategy", "strategy_public", "queried_size"]):
        row = dict(zip(["dataset", "backend", "classifier", "strategy", "strategy_public", "queried_size"], keys))
        for m in metrics:
            if m in g:
                row[f"{m}_mean"] = g[m].mean()
                row[f"{m}_std"] = g[m].std(ddof=1)
        rows.append(row)
    curves = pd.DataFrame(rows).sort_values(["dataset", "classifier", "strategy_public", "queried_size"])
    curves.to_csv(out / f"{prefix}_curve_data.csv", index=False)
    return curves


def plot_metric_curves(curves: pd.DataFrame, out: Path, prefix: str, metric: str) -> None:
    col = f"{metric}_mean"
    if col not in curves:
        return
    for keys, g in curves.groupby(["dataset", "backend", "classifier"]):
        plt.figure(figsize=(7.2, 4.8))
        for method in PUBLIC_ORDER:
            sg = g[g["strategy_public"] == method]
            if sg.empty:
                continue
            plt.plot(sg["queried_size"], sg[col], marker="o", linewidth=1.5, label=method)
            std_col = f"{metric}_std"
            if std_col in sg and len(sg) > 1:
                lo = sg[col] - sg[std_col].fillna(0)
                hi = sg[col] + sg[std_col].fillna(0)
                plt.fill_between(sg["queried_size"], lo, hi, alpha=0.12)
        plt.xlabel("Queried labels")
        plt.ylabel(metric.replace("_", " "))
        plt.title(" / ".join(map(str, keys)))
        plt.legend(fontsize=7, ncol=2)
        plt.tight_layout()
        stem = f"{prefix}_curve_{keys[0]}_{keys[2]}_{metric}".replace("/", "-")
        plt.savefig(out / f"{stem}.pdf")
        plt.savefig(out / f"{stem}.png", dpi=220)
        plt.close()


def plot_final_tradeoff(final: pd.DataFrame, out: Path, prefix: str) -> None:
    if "balanced_accuracy" not in final or "q_bias" not in final:
        return
    agg = final.groupby(["dataset", "backend", "classifier", "strategy_public"])[["balanced_accuracy", "q_bias", "eod_abs"]].mean().reset_index()
    agg.to_csv(out / f"{prefix}_final_tradeoff_data.csv", index=False)
    for keys, g in agg.groupby(["dataset", "backend", "classifier"]):
        plt.figure(figsize=(6.4, 4.8))
        for _, row in g.iterrows():
            plt.scatter(row["q_bias"], row["balanced_accuracy"], s=45)
            plt.text(row["q_bias"], row["balanced_accuracy"], str(row["strategy_public"]), fontsize=7)
        plt.xlabel("Final Q-Bias ↓")
        plt.ylabel("Final Balanced Accuracy ↑")
        plt.title("Final accuracy–query-bias trade-off: " + " / ".join(map(str, keys)))
        plt.tight_layout()
        stem = f"{prefix}_tradeoff_{keys[0]}_{keys[2]}".replace("/", "-")
        plt.savefig(out / f"{stem}.pdf")
        plt.savefig(out / f"{stem}.png", dpi=220)
        plt.close()


def make_clinical_impact_table(final: pd.DataFrame, out: Path, prefix: str, baseline: str) -> None:
    rows = []
    for keys, g in final.groupby(["dataset", "backend", "classifier", "seed"]):
        base = g[g["strategy"] == baseline]
        if base.empty:
            continue
        b = base.iloc[0]
        for _, r in g.iterrows():
            if r["strategy"] == baseline:
                continue
            row = dict(zip(["dataset", "backend", "classifier", "seed"], keys))
            row["strategy"] = r["strategy"]
            row["method"] = public_strategy_name(r["strategy"])
            row["baseline"] = public_strategy_name(baseline)
            for group in ["group0", "group1"]:
                pos = r.get(f"{group}_tp", 0) + r.get(f"{group}_fn", 0)
                neg = r.get(f"{group}_tn", 0) + r.get(f"{group}_fp", 0)
                n = max(float(r.get(f"{group}_n", 1)), 1.0)
                pos_rate = pos / n
                neg_rate = neg / n
                row[f"{group}_delta_fn_per_1000"] = 1000 * pos_rate * (r[f"{group}_fnr"] - b[f"{group}_fnr"])
                row[f"{group}_delta_fp_per_1000"] = 1000 * neg_rate * (r[f"{group}_fpr"] - b[f"{group}_fpr"])
            rows.append(row)
    impact = pd.DataFrame(rows)
    if impact.empty:
        return
    impact.to_csv(out / f"{prefix}_clinical_impact_by_seed.csv", index=False)
    # summary = impact.groupby(["dataset", "backend", "classifier", "method", "baseline"]).agg(["mean", "std"])


    group_cols = ["dataset", "backend", "classifier", "method", "baseline"]

    # Only aggregate numeric columns. Pandas 2.x no longer silently ignores
    # non-numeric columns for mean/std in groupby aggregation.
    numeric_cols = impact.select_dtypes(include="number").columns.tolist()

    # Exclude identifier-like numeric columns if present.
    exclude_cols = {
        "seed",
        "round",
        "labeled_size",
        "queried_size",
        "unlabeled_size",
    }
    numeric_cols = [c for c in numeric_cols if c not in exclude_cols and c not in group_cols]

    if not numeric_cols:
        print("[WARN] No numeric columns found for clinical impact summary; skipping.")
        return

    summary = (
        impact.groupby(group_cols, dropna=False)[numeric_cols]
        .agg(["mean", "std"])
        .reset_index()
    )

    summary.columns = [
        "_".join([str(x) for x in col if str(x) != ""]).rstrip("_")
        if isinstance(col, tuple)
        else col
        for col in summary.columns
    ]





    
    summary.columns = ["_".join(c).strip("_") for c in summary.columns.to_flat_index()]
    summary = summary.reset_index()
    summary.to_csv(out / f"{prefix}_clinical_impact_summary.csv", index=False)


def main() -> None:
    args = parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = read_inputs(args.inputs)
    final = final_rows(df)
    final.to_csv(out / f"{args.prefix}_final_rows.csv", index=False)
    make_main_table(final, out, args.prefix)
    curves = aggregate_curves(df, out, args.prefix)
    for metric in ["balanced_accuracy", "q_bias", "eod_abs", "group0_sensitivity", "group1_sensitivity"]:
        plot_metric_curves(curves, out, args.prefix, metric)
    plot_final_tradeoff(final, out, args.prefix)
    make_clinical_impact_table(final, out, args.prefix, args.baseline)
    print(f"Saved paper artifacts to {out}")


if __name__ == "__main__":
    main()

