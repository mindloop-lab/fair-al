
from __future__ import annotations

r"""Generate paper-ready tables and figures for FAIR-AL experiments.

This module is designed to be copied into ``fairal_deepasd_tv_project_v8_1/src/baalexp``.
It reads the existing CSV files under ``results/`` and creates:

- paper-ready CSV summaries;
- LaTeX tables that can be inserted with ``\input{...}``;
- PDF/PNG figures for Q-Bias curves, BA curves, trade-off plots, and sensitivity;
- a short Markdown report explaining the generated artifacts.

The script intentionally keeps internal code names such as ``baal`` compatible
with existing result CSV files, but displays them in the paper as FAIR-AL names.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping
import argparse
import math
import re
import warnings

import numpy as np
import pandas as pd

try:
    from scipy.stats import ttest_rel, wilcoxon, mannwhitneyu
except Exception:  # pragma: no cover - only used in minimal environments
    ttest_rel = None
    wilcoxon = None
    mannwhitneyu = None

import matplotlib.pyplot as plt

EPS = 1e-12

STRATEGY_DISPLAY: dict[str, str] = {
    "random": "Random",
    "uncertainty": "Uncertainty",
    "reweighting": "Reweighting",
    "quota": "Quota sampling",
    "stratified_uncertainty": "Stratified uncertainty",
    "heuristic": "Heuristic",
    "coreset": "CoreSet",
    "badge": "BADGE",
    "baal": "FAIR-AL-Pareto",
    "baal_weighted": "FAIR-AL-Weighted",
    "baal_hard": "FAIR-AL-Constrained",
    "baal_random_pareto": "FAIR-AL-RandomPareto",
}

STRATEGY_ORDER: list[str] = [
    "random",
    "uncertainty",
    "reweighting",
    "heuristic",
    "coreset",
    "badge",
    "baal",
    "baal_weighted",
    "baal_hard",
    "baal_random_pareto",
]

METRIC_DISPLAY: dict[str, str] = {
    "balanced_accuracy": "BA",
    "auc": "AUC",
    "q_bias": "Q-Bias",
    "eod_abs": "EOD",
    "group0_sensitivity": "G0 Sens.",
    "group1_sensitivity": "G1 Sens.",
    "group0_fpr": "G0 FPR",
    "group1_fpr": "G1 FPR",
    "group0_fnr": "G0 FNR",
    "group1_fnr": "G1 FNR",
    "accuracy": "Acc.",
}

CORE_METRICS = [
    "balanced_accuracy",
    "auc",
    "q_bias",
    "eod_abs",
    "group0_sensitivity",
    "group1_sensitivity",
]

EXTENDED_METRICS = CORE_METRICS + [
    "group0_fpr",
    "group1_fpr",
    "group0_fnr",
    "group1_fnr",
]

MAIN_DATASETS = ["ABIDE_A", "ABIDE_B"]
MAIN_CLASSIFIERS = ["svm", "lr", "mlp"]


@dataclass
class ArtifactPaths:
    results_dir: Path
    out_dir: Path
    tables_dir: Path
    figures_dir: Path
    data_dir: Path


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_")


def tex_escape(value: object) -> str:
    s = str(value)
    repl = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
        "\\": r"\textbackslash{}",
    }
    return "".join(repl.get(ch, ch) for ch in s)


def fmt_float(x: object, ndigits: int = 3) -> str:
    try:
        val = float(x)
    except Exception:
        return "--"
    if not np.isfinite(val):
        return "--"
    return f"{val:.{ndigits}f}"


def fmt_mean_std(mean: object, std: object, ndigits: int = 3) -> str:
    return f"{fmt_float(mean, ndigits)} $\\pm$ {fmt_float(std, ndigits)}"




def benjamini_hochberg(p_values: pd.Series) -> pd.Series:
    """Benjamini-Hochberg FDR adjustment, preserving NaNs."""
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
    """Cohen's dz for paired differences."""
    d = pd.to_numeric(delta, errors="coerce").dropna()
    if len(d) < 2:
        return float("nan")
    sd = float(d.std(ddof=1))
    if sd <= EPS:
        return 0.0
    return float(d.mean() / sd)


def load_csv_if_exists(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:
        warnings.warn(f"Failed to read {path}: {exc}")
        return None


def infer_classifier_from_path(path: Path) -> str:
    name = path.name
    for clf in MAIN_CLASSIFIERS:
        if f"_{clf}_" in name:
            return clf
    return "unknown"


def load_main_experiment_results(results_dir: Path) -> pd.DataFrame:
    """Load validated main FAIR-AL experiment CSV files.

    Formal paper results should use:
    - ABIDE_A_*_core_v8.csv
    - ABIDE_B_*_core_sex46_v8.csv
    """
    primary_patterns = [
        "ABIDE_A_*_core_v8.csv",
        "ABIDE_B_*_core_sex46_v8.csv",
    ]
    fallback_patterns = [
        "ABIDE_A_*_core_fixed.csv",
        "ABIDE_B_*_core_sex46_v4.csv",
    ]
    primary_files = [p for pat in primary_patterns for p in sorted(results_dir.glob(pat))]
    patterns = primary_patterns if primary_files else fallback_patterns

    frames: list[pd.DataFrame] = []
    for pat in patterns:
        for path in sorted(results_dir.glob(pat)):
            if path.name.endswith(".env.json") or path.name.endswith(".history.json"):
                continue
            df = load_csv_if_exists(path)
            if df is None or df.empty:
                continue
            if "classifier" not in df.columns or df["classifier"].isna().all():
                df["classifier"] = infer_classifier_from_path(path)
            df["source_file"] = path.name
            frames.append(df)
    if not frames:
        raise FileNotFoundError(
            f"No main experiment CSV files found in {results_dir}. Expected primary patterns: {primary_patterns}; fallback patterns: {fallback_patterns}"
        )
    out = pd.concat(frames, ignore_index=True)
    out["method"] = out["strategy"].map(STRATEGY_DISPLAY).fillna(out["strategy"])
    out["strategy_order"] = out["strategy"].map({s: i for i, s in enumerate(STRATEGY_ORDER)}).fillna(999)
    return out


def final_rows(df: pd.DataFrame) -> pd.DataFrame:
    key_cols = ["dataset", "backend", "classifier", "strategy", "seed"]
    missing = [c for c in key_cols + ["round"] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for final_rows: {missing}")
    idx = df.groupby(key_cols)["round"].idxmax()
    out = df.loc[idx].copy().reset_index(drop=True)
    out["method"] = out["strategy"].map(STRATEGY_DISPLAY).fillna(out["strategy"])
    out["strategy_order"] = out["strategy"].map({s: i for i, s in enumerate(STRATEGY_ORDER)}).fillna(999)
    return out


def summarize_final_rows(final: pd.DataFrame, metrics: list[str] = EXTENDED_METRICS) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    group_cols = ["dataset", "backend", "classifier", "strategy", "method", "strategy_order"]
    for keys, g in final.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, keys))
        row["n_seeds"] = int(g["seed"].nunique()) if "seed" in g.columns else len(g)
        for m in metrics:
            if m in g.columns:
                row[f"{m}_mean"] = float(g[m].mean())
                row[f"{m}_std"] = float(g[m].std(ddof=1)) if len(g[m].dropna()) > 1 else 0.0
        rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["dataset", "classifier", "strategy_order"]).reset_index(drop=True)
    return out


def paired_tests(final: pd.DataFrame, baseline: str = "uncertainty", metrics: list[str] = CORE_METRICS) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, g in final.groupby(["dataset", "backend", "classifier"], dropna=False):
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
            for metric in metrics:
                if metric not in g.columns:
                    continue
                x = sg.loc[common, metric].astype(float)
                y = base.loc[common, metric].astype(float)
                delta = x - y
                paired_t_p = np.nan
                wilcoxon_p = np.nan
                if ttest_rel is not None:
                    try:
                        paired_t_p = float(ttest_rel(x, y, nan_policy="omit").pvalue)
                    except Exception:
                        paired_t_p = np.nan
                if wilcoxon is not None:
                    try:
                        if np.nanmax(np.abs(delta)) > 0:
                            wilcoxon_p = float(wilcoxon(delta.dropna()).pvalue)
                    except Exception:
                        wilcoxon_p = np.nan
                rows.append(
                    {
                        "dataset": keys[0],
                        "backend": keys[1],
                        "classifier": keys[2],
                        "strategy": strategy,
                        "method": STRATEGY_DISPLAY.get(strategy, strategy),
                        "baseline": baseline,
                        "metric": metric,
                        "metric_display": METRIC_DISPLAY.get(metric, metric),
                        "n_pairs": len(common),
                        "mean_strategy": float(x.mean()),
                        "mean_baseline": float(y.mean()),
                        "delta_strategy_minus_baseline": float(delta.mean()),
                        "cohen_dz": paired_cohen_dz(delta),
                        "paired_t_p": paired_t_p,
                        "wilcoxon_p": wilcoxon_p,
                    }
                )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["paired_t_p_fdr"] = benjamini_hochberg(out["paired_t_p"])
        out["wilcoxon_p_fdr"] = benjamini_hochberg(out["wilcoxon_p"])
        out["strategy_order"] = out["strategy"].map({s: i for i, s in enumerate(STRATEGY_ORDER)}).fillna(999)
        out = out.sort_values(["dataset", "classifier", "strategy_order", "metric"]).reset_index(drop=True)
    return out


def write_latex_table(path: Path, table_body: str, caption: str, label: str, note: str | None = None, star: bool = True) -> None:
    env = "table*" if star else "table"
    text = [f"\\begin{{{env}}}[htbp]", "\\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}", table_body]
    if note:
        text.extend([
            "\\vspace{0.5em}",
            "\\begin{minipage}{0.96\\textwidth}",
            "\\footnotesize",
            f"\\textit{{Notes.}} {note}",
            "\\end{minipage}",
        ])
    text.append(f"\\end{{{env}}}")
    path.write_text("\n".join(text) + "\n", encoding="utf-8")


def make_main_results_table(summary: pd.DataFrame, out_tex: Path, out_csv: Path) -> pd.DataFrame:
    keep_strategies = [
        "random",
        "uncertainty",
        "reweighting",
        "heuristic",
        "coreset",
        "badge",
        "baal",
        "baal_weighted",
        "baal_hard",
        "baal_random_pareto",
    ]
    tab = summary[summary["strategy"].isin(keep_strategies)].copy()
    tab = tab.sort_values(["dataset", "classifier", "strategy_order"])
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    tab.to_csv(out_csv, index=False)

    headers = ["Dataset", "Classifier", "Method", "BA", "AUC", "Q-Bias", "EOD", "G0 Sens.", "G1 Sens."]
    lines = ["\\resizebox{\\textwidth}{!}{%", "\\begin{tabular}{lllcccccc}", "\\toprule", " & ".join(headers) + r" \\", "\\midrule"]
    for _, r in tab.iterrows():
        values = [
            tex_escape(r["dataset"].replace("_", "-")),
            tex_escape(str(r["classifier"]).upper()),
            tex_escape(r["method"]),
            fmt_mean_std(r.get("balanced_accuracy_mean"), r.get("balanced_accuracy_std")),
            fmt_mean_std(r.get("auc_mean"), r.get("auc_std")),
            fmt_mean_std(r.get("q_bias_mean"), r.get("q_bias_std")),
            fmt_mean_std(r.get("eod_abs_mean"), r.get("eod_abs_std")),
            fmt_mean_std(r.get("group0_sensitivity_mean"), r.get("group0_sensitivity_std")),
            fmt_mean_std(r.get("group1_sensitivity_mean"), r.get("group1_sensitivity_std")),
        ]
        lines.append(" & ".join(values) + r" \\")
    lines.extend(["\\bottomrule", "\\end{tabular}%", "}"])
    note = (
        "Results are averaged over random seeds. BA denotes balanced accuracy. "
        "Group~0 denotes the minority sex group and Group~1 denotes the majority sex group. "
        "For ABIDE-B, the sensitive attribute is PHENO feature~46, used as the sex proxy."
    )
    write_latex_table(
        out_tex,
        "\n".join(lines),
        "Main active-learning results on ABIDE-derived ASD screening features.",
        "tab:fairal_main_results",
        note,
    )
    return tab


def make_compact_highlight_table(summary: pd.DataFrame, out_tex: Path, out_csv: Path) -> pd.DataFrame:
    """A shorter table for the main paper; full table can go to supplement."""
    keep = ["uncertainty", "reweighting", "baal", "baal_weighted", "baal_hard"]
    tab = summary[summary["strategy"].isin(keep)].copy()
    tab = tab.sort_values(["dataset", "classifier", "strategy_order"])
    tab.to_csv(out_csv, index=False)
    headers = ["Dataset", "Clf.", "Method", "BA", "AUC", "Q-Bias", "EOD"]
    lines = ["\\resizebox{\\textwidth}{!}{%", "\\begin{tabular}{lllcccc}", "\\toprule", " & ".join(headers) + r" \\", "\\midrule"]
    for _, r in tab.iterrows():
        values = [
            tex_escape(r["dataset"].replace("_", "-")),
            tex_escape(str(r["classifier"]).upper()),
            tex_escape(r["method"]),
            fmt_mean_std(r.get("balanced_accuracy_mean"), r.get("balanced_accuracy_std")),
            fmt_mean_std(r.get("auc_mean"), r.get("auc_std")),
            fmt_mean_std(r.get("q_bias_mean"), r.get("q_bias_std")),
            fmt_mean_std(r.get("eod_abs_mean"), r.get("eod_abs_std")),
        ]
        lines.append(" & ".join(values) + r" \\")
    lines.extend(["\\bottomrule", "\\end{tabular}%", "}"])
    note = (
        "This compact table highlights the uncertainty baseline, two naive fairness interventions, "
        "and FAIR-AL variants. Full results including Random, CoreSet, BADGE, and RandomPareto "
        "are provided in the supplementary table."
    )
    write_latex_table(out_tex, "\n".join(lines), "Compact summary of FAIR-AL results.", "tab:fairal_compact_results", note)
    return tab


def make_tests_table(tests: pd.DataFrame, out_tex: Path, out_csv: Path) -> pd.DataFrame:
    if tests.empty:
        out_csv.write_text("", encoding="utf-8")
        return tests
    metrics = ["balanced_accuracy", "q_bias", "eod_abs"]
    keep = tests[tests["metric"].isin(metrics)].copy()
    keep = keep[keep["strategy"].isin(["baal", "baal_weighted", "baal_hard", "reweighting", "heuristic"])]
    keep.to_csv(out_csv, index=False)
    headers = ["Dataset", "Clf.", "Method", "Metric", "$\\Delta$ vs Unc.", "dz", "$p_t$", "$q_t$", "$p_W$", "$q_W$"]
    lines = ["\\resizebox{\\textwidth}{!}{%", "\\begin{tabular}{lllccccccc}", "\\toprule", " & ".join(headers) + r" \\", "\\midrule"]
    for _, r in keep.iterrows():
        vals = [
            tex_escape(str(r["dataset"]).replace("_", "-")),
            tex_escape(str(r["classifier"]).upper()),
            tex_escape(r["method"]),
            tex_escape(r["metric_display"]),
            fmt_float(r["delta_strategy_minus_baseline"]),
            fmt_float(r.get("cohen_dz")),
            fmt_float(r["paired_t_p"]),
            fmt_float(r.get("paired_t_p_fdr")),
            fmt_float(r["wilcoxon_p"]),
            fmt_float(r.get("wilcoxon_p_fdr")),
        ]
        lines.append(" & ".join(vals) + r" \\")
    lines.extend(["\\bottomrule", "\\end{tabular}%", "}"])
    note = "$\\Delta$ is method minus Uncertainty. dz is paired Cohen's effect size. $p_t$/$p_W$ are raw paired $t$-test/Wilcoxon p-values; $q_t$/$q_W$ are Benjamini--Hochberg FDR-adjusted values."
    write_latex_table(out_tex, "\n".join(lines), "Paired comparisons against uncertainty sampling.", "tab:fairal_tests_vs_uncertainty", note)
    return keep


def plot_metric_curve(df: pd.DataFrame, dataset: str, classifier: str, metric: str, out_base: Path, strategies: list[str] | None = None) -> None:
    sub = df[(df["dataset"] == dataset) & (df["classifier"] == classifier)]
    if sub.empty or metric not in sub.columns:
        return
    if strategies is None:
        strategies = ["uncertainty", "reweighting", "heuristic", "baal", "baal_weighted", "baal_hard"]
    plt.figure(figsize=(7.0, 4.8))
    for strategy in strategies:
        sg = sub[sub["strategy"] == strategy]
        if sg.empty:
            continue
        curve = sg.groupby("queried_size")[metric].agg(["mean", "std"]).reset_index()
        label = STRATEGY_DISPLAY.get(strategy, strategy)
        plt.plot(curve["queried_size"], curve["mean"], marker="o", linewidth=1.8, label=label)
        if len(curve) > 1:
            lo = curve["mean"] - curve["std"].fillna(0)
            hi = curve["mean"] + curve["std"].fillna(0)
            plt.fill_between(curve["queried_size"], lo, hi, alpha=0.12)
    plt.xlabel("Queried labels")
    plt.ylabel(METRIC_DISPLAY.get(metric, metric))
    plt.title(f"{dataset.replace('_', '-')} / {classifier.upper()}")
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    for ext in ["pdf", "png"]:
        plt.savefig(out_base.with_suffix(f".{ext}"), dpi=300)
    plt.close()


def plot_tradeoff(final: pd.DataFrame, dataset: str, classifier: str, out_base: Path) -> None:
    sub = final[(final["dataset"] == dataset) & (final["classifier"] == classifier)]
    if sub.empty:
        return
    agg = sub.groupby(["strategy", "method", "strategy_order"])[["balanced_accuracy", "q_bias"]].agg(["mean", "std"]).reset_index()
    agg.columns = ["_".join([str(x) for x in c if x]) for c in agg.columns.to_flat_index()]
    agg = agg.sort_values("strategy_order")
    plt.figure(figsize=(6.5, 5.2))
    for _, r in agg.iterrows():
        x = r["q_bias_mean"]
        y = r["balanced_accuracy_mean"]
        plt.errorbar(x, y, xerr=r.get("q_bias_std", 0), yerr=r.get("balanced_accuracy_std", 0), fmt="o", capsize=3)
        plt.text(x, y, " " + str(r["method"]), fontsize=8, va="center")
    plt.xlabel("Q-Bias (lower is better)")
    plt.ylabel("Balanced accuracy (higher is better)")
    plt.title(f"Final accuracy--query-bias trade-off: {dataset.replace('_', '-')} / {classifier.upper()}")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    for ext in ["pdf", "png"]:
        plt.savefig(out_base.with_suffix(f".{ext}"), dpi=300)
    plt.close()


def make_core_figures(df: pd.DataFrame, final: pd.DataFrame, figures_dir: Path) -> None:
    ensure_dir(figures_dir)
    for dataset in MAIN_DATASETS:
        for classifier in MAIN_CLASSIFIERS:
            for metric in ["q_bias", "balanced_accuracy", "eod_abs", "group0_sensitivity", "group1_sensitivity"]:
                out_base = figures_dir / f"fairal_curve_{safe_name(dataset)}_{classifier}_{metric}"
                plot_metric_curve(df, dataset, classifier, metric, out_base)
            out_base = figures_dir / f"fairal_tradeoff_{safe_name(dataset)}_{classifier}"
            plot_tradeoff(final, dataset, classifier, out_base)


def parse_sensitivity_kind(path: Path) -> tuple[str, str, float | int | str] | None:
    name = path.stem
    # ABIDE_B_svm_sex46_sens_kcand_100_fixed
    # sens_kcand_100_fixed
    m = re.search(r"sens_(kcand|batch|init|pop)_([A-Za-z0-9mp]+)_fixed", name)
    if not m:
        return None
    kind, raw = m.group(1), m.group(2)
    if kind in {"kcand", "batch", "init"}:
        try:
            val: float | int | str = int(raw)
        except ValueError:
            val = raw
    else:
        s = raw.replace("m", "-").replace("p", ".")
        try:
            val = float(s)
        except ValueError:
            val = raw
    dataset = "ABIDE_B" if name.startswith("ABIDE_B") else "ABIDE_A"
    return dataset, kind, val


def load_sensitivity_results(results_dir: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in sorted(results_dir.glob("*sens_*_fixed.csv")):
        parsed = parse_sensitivity_kind(path)
        if parsed is None:
            continue
        dataset, kind, val = parsed
        df = load_csv_if_exists(path)
        if df is None or df.empty:
            continue
        df["sens_dataset"] = dataset
        df["sens_kind"] = kind
        df["sens_value"] = val
        df["source_file"] = path.name
        if "classifier" not in df.columns or df["classifier"].isna().all():
            df["classifier"] = "svm"
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["method"] = out["strategy"].map(STRATEGY_DISPLAY).fillna(out["strategy"])
    return out


def make_sensitivity_tables_and_figures(sens: pd.DataFrame, tables_dir: Path, figures_dir: Path, data_dir: Path) -> None:
    if sens.empty:
        return
    fin = final_rows(sens.rename(columns={"sens_dataset": "dataset"}) if "dataset" not in sens.columns else sens)
    # Preserve sensitivity metadata by merging from source columns after final_rows if possible.
    if not {"sens_kind", "sens_value"}.issubset(fin.columns):
        # fallback: recompute final by richer group
        idx = sens.groupby(["source_file", "strategy", "seed"])["round"].idxmax()
        fin = sens.loc[idx].copy().reset_index(drop=True)
    summary_rows = []
    for keys, g in fin.groupby(["sens_dataset", "sens_kind", "sens_value", "strategy", "method"], dropna=False):
        row = dict(zip(["dataset", "kind", "value", "strategy", "method"], keys))
        for m in ["balanced_accuracy", "q_bias", "eod_abs"]:
            if m in g.columns:
                row[f"{m}_mean"] = float(g[m].mean())
                row[f"{m}_std"] = float(g[m].std(ddof=1)) if len(g) > 1 else 0.0
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    if summary.empty:
        return
    summary.to_csv(data_dir / "fairal_sensitivity_summary.csv", index=False)

    # LaTeX tables for k_candidates and p_pop; these are the most relevant for the manuscript.
    for kind, label, caption in [
        ("kcand", "tab:fairal_sensitivity_kcand", "Sensitivity to the candidate-pool size."),
        ("pop", "tab:fairal_sensitivity_pop", "Sensitivity to target protected-group distribution perturbation."),
    ]:
        sub = summary[summary["kind"] == kind].copy()
        if sub.empty:
            continue
        sub = sub[sub["strategy"].isin(["uncertainty", "baal", "baal_weighted", "baal_hard"])]
        sub = sub.sort_values(["dataset", "value", "strategy"])
        headers = ["Dataset", "Value", "Method", "BA", "Q-Bias", "EOD"]
        lines = ["\\resizebox{\\textwidth}{!}{%", "\\begin{tabular}{llcccc}", "\\toprule", " & ".join(headers) + r" \\", "\\midrule"]
        for _, r in sub.iterrows():
            vals = [
                tex_escape(str(r["dataset"]).replace("_", "-")),
                tex_escape(r["value"]),
                tex_escape(r["method"]),
                fmt_mean_std(r.get("balanced_accuracy_mean"), r.get("balanced_accuracy_std")),
                fmt_mean_std(r.get("q_bias_mean"), r.get("q_bias_std")),
                fmt_mean_std(r.get("eod_abs_mean"), r.get("eod_abs_std")),
            ]
            lines.append(" & ".join(vals) + r" \\")
        lines.extend(["\\bottomrule", "\\end{tabular}%", "}"])
        note = "Only SVM sensitivity runs are summarized here. Full sensitivity CSV files are provided in the results directory."
        write_latex_table(tables_dir / f"fairal_sensitivity_{kind}.tex", "\n".join(lines), caption, label, note)

    # Figures: plot q_bias vs sensitivity value for key methods.
    for kind in ["kcand", "pop", "batch", "init"]:
        sub = summary[summary["kind"] == kind].copy()
        if sub.empty:
            continue
        for dataset in sorted(sub["dataset"].unique()):
            ds = sub[sub["dataset"] == dataset]
            plt.figure(figsize=(6.5, 4.5))
            for strategy in ["uncertainty", "baal", "baal_weighted", "baal_hard"]:
                sg = ds[ds["strategy"] == strategy].sort_values("value")
                if sg.empty:
                    continue
                plt.errorbar(sg["value"], sg["q_bias_mean"], yerr=sg["q_bias_std"], marker="o", capsize=3, label=STRATEGY_DISPLAY.get(strategy, strategy))
            plt.xlabel(kind)
            plt.ylabel("Q-Bias")
            plt.title(f"Sensitivity of Q-Bias: {dataset.replace('_', '-')}, {kind}")
            plt.grid(alpha=0.25)
            plt.legend(fontsize=8)
            plt.tight_layout()
            out_base = figures_dir / f"fairal_sensitivity_{safe_name(dataset)}_{kind}_q_bias"
            for ext in ["pdf", "png"]:
                plt.savefig(out_base.with_suffix(f".{ext}"), dpi=300)
            plt.close()


def make_deepasd_tables(results_dir: Path, tables_dir: Path, data_dir: Path) -> None:
    static_files = [
        results_dir / "deepasd_static_sex_ABIDE_A_fixed.summary.csv",
        results_dir / "deepasd_static_sex_ABIDE_B_sex46_fixed.summary.csv",
    ]
    static_frames = [load_csv_if_exists(p) for p in static_files]
    static_frames = [x for x in static_frames if x is not None and not x.empty]
    if static_frames:
        static = pd.concat(static_frames, ignore_index=True)
        static.to_csv(data_dir / "deepasd_static_sex_summary.csv", index=False)
        headers = ["Dataset", "BA", "AUC", "G0 Sens.", "G1 Sens.", "G0 FNR", "G1 FNR", "G0 FPR", "G1 FPR", "Sens. Gap", "FPR Gap"]
        lines = ["\\resizebox{\\textwidth}{!}{%", "\\begin{tabular}{lcccccccccc}", "\\toprule", " & ".join(headers) + r" \\", "\\midrule"]
        for _, r in static.iterrows():
            dataset = str(r["dataset"]).replace("_", "-")
            if dataset == "ABIDE-B":
                dataset = "ABIDE-B sex46"
            vals = [
                tex_escape(dataset),
                fmt_float(r.get("balanced_accuracy_mean")),
                fmt_float(r.get("auc_mean")),
                fmt_float(r.get("group0_sensitivity_mean")),
                fmt_float(r.get("group1_sensitivity_mean")),
                fmt_float(r.get("group0_fnr_mean")),
                fmt_float(r.get("group1_fnr_mean")),
                fmt_float(r.get("group0_fpr_mean")),
                fmt_float(r.get("group1_fpr_mean")),
                fmt_float(r.get("sens_g0_minus_g1_mean")),
                fmt_float(r.get("fpr_g0_minus_g1_mean")),
            ]
            lines.append(" & ".join(vals) + r" \\")
        lines.extend(["\\bottomrule", "\\end{tabular}%", "}"])
        note = (
            "Group~0 denotes the minority sex group and Group~1 denotes the majority sex group. "
            "Sensitivity gap is Group~0 minus Group~1. These results show sex-stratified outcome trade-offs, not a simple pattern of minority-group underperformance."
        )
        write_latex_table(
            tables_dir / "deepasd_static_sex_table.tex",
            "\n".join(lines),
            "Sex-stratified performance of the original DeepASD model.",
            "tab:deepasd_static_sex",
            note,
        )

    unc_files = [
        results_dir / "deepasd_uncertainty_sex_ABIDE_A_fixed.summary.csv",
        results_dir / "deepasd_uncertainty_sex_ABIDE_B_sex46_fixed.summary.csv",
    ]
    unc_frames = [load_csv_if_exists(p) for p in unc_files]
    unc_frames = [x for x in unc_frames if x is not None and not x.empty]
    if unc_frames:
        unc = pd.concat(unc_frames, ignore_index=True)
        unc.to_csv(data_dir / "deepasd_uncertainty_sex_summary.csv", index=False)
        # summary has one entropy row and multiple topk rows per dataset.
        rows = []
        for dataset, g in unc.groupby("dataset"):
            ent = g[g["topk"].isna()].iloc[0] if g["topk"].isna().any() else None
            top = g[g["topk"].notna()].copy()
            def get_top(k: int, col: str) -> float:
                r = top[np.isclose(top["topk"].astype(float), float(k))]
                return float(r[col].iloc[0]) if not r.empty else np.nan
            rows.append(
                {
                    "dataset": dataset,
                    "entropy_g0_minus_g1": ent.get("entropy_mean_g0_minus_g1_mean", np.nan) if ent is not None else np.nan,
                    "mw_p": ent.get("entropy_mannwhitney_p_mean", np.nan) if ent is not None else np.nan,
                    "target_g0": get_top(40, "target_p_group0_mean"),
                    "top40_g0": get_top(40, "topk_p_group0_mean"),
                    "top40_q_bias": get_top(40, "topk_q_bias_mean"),
                    "top120_q_bias": get_top(120, "topk_q_bias_mean"),
                    "top200_q_bias": get_top(200, "topk_q_bias_mean"),
                }
            )
        deep_unc = pd.DataFrame(rows)
        deep_unc.to_csv(data_dir / "deepasd_uncertainty_by_sex_table_data.csv", index=False)
        headers = ["Dataset", "Entropy G0--G1", "MW $p$", "Target G0", "Top-40 G0", "Top-40 Q-Bias", "Top-120 Q-Bias", "Top-200 Q-Bias"]
        lines = ["\\resizebox{\\textwidth}{!}{%", "\\begin{tabular}{lccccccc}", "\\toprule", " & ".join(headers) + r" \\", "\\midrule"]
        for _, r in deep_unc.iterrows():
            dataset = str(r["dataset"]).replace("_", "-")
            if dataset == "ABIDE-B":
                dataset = "ABIDE-B sex46"
            vals = [
                tex_escape(dataset),
                fmt_float(r["entropy_g0_minus_g1"]),
                fmt_float(r["mw_p"]),
                fmt_float(r["target_g0"]),
                fmt_float(r["top40_g0"]),
                fmt_float(r["top40_q_bias"]),
                fmt_float(r["top120_q_bias"]),
                fmt_float(r["top200_q_bias"]),
            ]
            lines.append(" & ".join(vals) + r" \\")
        lines.extend(["\\bottomrule", "\\end{tabular}%", "}"])
        note = (
            "Entropy G0--G1 is the mean entropy of Group~0 minus Group~1. MW $p$ is the Mann--Whitney test p-value. "
            "The results indicate that DeepASD uncertainty-based acquisition can underrepresent the minority sex group, especially at small query budgets."
        )
        write_latex_table(
            tables_dir / "deepasd_uncertainty_by_sex_table.tex",
            "\n".join(lines),
            "Uncertainty-by-sex analysis of the original DeepASD model.",
            "tab:deepasd_uncertainty_sex",
            note,
        )


def make_clinical_impact_table(results_dir: Path, tables_dir: Path, data_dir: Path) -> None:
    path = results_dir / "paper_artifacts" / "baal_clinical_impact_summary.csv"
    if not path.exists():
        path = results_dir / "paper_artifacts" / "fairal_clinical_impact_summary.csv"
    df = load_csv_if_exists(path)
    if df is None or df.empty:
        return
    df.to_csv(data_dir / "fairal_clinical_impact_summary.csv", index=False)
    # Keep a concise table if recognizable columns exist.
    cols = [c for c in df.columns if c in {"dataset", "classifier", "strategy", "method", "baseline"} or "delta" in c]
    if not cols:
        return
    show = df[cols].copy().head(40)
    headers = [tex_escape(c.replace("_", " ")) for c in show.columns]
    lines = ["\\resizebox{\\textwidth}{!}{%", f"\\begin{{tabular}}{{{'l' * len(headers)}}}", "\\toprule", " & ".join(headers) + r" \\", "\\midrule"]
    for _, r in show.iterrows():
        vals = []
        for c in show.columns:
            if pd.api.types.is_numeric_dtype(show[c]):
                vals.append(fmt_float(r[c]))
            else:
                vals.append(tex_escape(r[c]))
        lines.append(" & ".join(vals) + r" \\")
    lines.extend(["\\bottomrule", "\\end{tabular}%", "}"])
    note = "Clinical impact is reported as changes in false positives/false negatives per 1000 patients relative to the baseline when available."
    write_latex_table(tables_dir / "fairal_clinical_impact_table.tex", "\n".join(lines), "Clinical impact summary.", "tab:fairal_clinical_impact", note)


def make_master_latex_file(tables_dir: Path) -> None:
    tables = [
        "fairal_main_results_compact.tex",
        "fairal_main_results_full.tex",
        "fairal_tests_vs_uncertainty.tex",
        "deepasd_static_sex_table.tex",
        "deepasd_uncertainty_by_sex_table.tex",
        "fairal_sensitivity_kcand.tex",
        "fairal_sensitivity_pop.tex",
        "fairal_clinical_impact_table.tex",
    ]
    lines = ["% Auto-generated by baalexp.paper_figures_tables", "% Include selected tables in your manuscript with \\input{tables/generated/<file>.tex}"]
    for t in tables:
        if (tables_dir / t).exists():
            lines.append(f"\\input{{tables/generated/{t}}}")
    (tables_dir / "all_tables_input.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_figures_readme(figures_dir: Path) -> None:
    lines = [
        "# FAIR-AL generated figures",
        "",
        "Recommended main-text figures:",
        "",
        "1. `fairal_curve_ABIDE_A_svm_q_bias.pdf` and `fairal_curve_ABIDE_B_lr_q_bias.pdf` for Q-Bias trajectories.",
        "2. `fairal_tradeoff_ABIDE_A_svm.pdf` and `fairal_tradeoff_ABIDE_B_lr.pdf` for final accuracy--query-bias trade-offs.",
        "3. `fairal_sensitivity_ABIDE_A_pop_q_bias.pdf` and `fairal_sensitivity_ABIDE_B_pop_q_bias.pdf` for target-distribution perturbation sensitivity, if included in the main text.",
        "",
        "Other figures can be moved to supplementary materials.",
    ]
    (figures_dir / "README_figures.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_artifact_report(paths: ArtifactPaths, main_summary: pd.DataFrame, tests: pd.DataFrame) -> None:
    lines = [
        "# FAIR-AL paper artifact generation report",
        "",
        "## Core interpretation",
        "",
        "The generated artifacts support the manuscript narrative that FAIR-AL controls acquisition-level demographic query drift in ASD screening. The tables keep internal CSV strategy names compatible with the existing code, but display FAIR-AL variants.",
        "",
        "## Generated directories",
        "",
        f"- Tables: `{paths.tables_dir}`",
        f"- Figures: `{paths.figures_dir}`",
        f"- Processed data: `{paths.data_dir}`",
        "",
        "## Recommended manuscript placement",
        "",
        "- Introduction / motivating evidence: `deepasd_static_sex_table.tex`, `deepasd_uncertainty_by_sex_table.tex`.",
        "- Main results: `fairal_main_results_compact.tex` plus Q-Bias and trade-off figures.",
        "- Supplementary: `fairal_main_results_full.tex`, sensitivity tables, clinical-impact table, all additional curves.",
        "",
        "## Available LaTeX tables",
        "",
    ]
    for p in sorted(paths.tables_dir.glob("*.tex")):
        lines.append(f"- `{p.name}`")
    lines.extend(["", "## Available figures", ""])
    for p in sorted(paths.figures_dir.glob("*.pdf")):
        lines.append(f"- `{p.name}`")
    lines.extend(["", "## Summary snapshot", ""])
    if not main_summary.empty:
        compact = main_summary[main_summary["strategy"].isin(["uncertainty", "baal", "baal_weighted", "baal_hard"])]
        cols = ["dataset", "classifier", "method", "balanced_accuracy_mean", "auc_mean", "q_bias_mean", "eod_abs_mean"]
        lines.append(compact[[c for c in cols if c in compact.columns]].to_markdown(index=False))
    (paths.out_dir / "artifact_generation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_generation(results_dir: Path, out_dir: Path) -> None:
    paths = ArtifactPaths(
        results_dir=results_dir,
        out_dir=ensure_dir(out_dir),
        tables_dir=ensure_dir(out_dir / "tables" / "generated"),
        figures_dir=ensure_dir(out_dir / "fig"),
        data_dir=ensure_dir(out_dir / "data"),
    )

    df = load_main_experiment_results(results_dir)
    df.to_csv(paths.data_dir / "fairal_all_iterations.csv", index=False)
    fin = final_rows(df)
    fin.to_csv(paths.data_dir / "fairal_final_rows.csv", index=False)
    summary = summarize_final_rows(fin)
    summary.to_csv(paths.data_dir / "fairal_summary_mean_std.csv", index=False)
    tests = paired_tests(fin)
    tests.to_csv(paths.data_dir / "fairal_tests_vs_uncertainty.csv", index=False)

    make_main_results_table(summary, paths.tables_dir / "fairal_main_results_full.tex", paths.data_dir / "fairal_main_results_full.csv")
    make_compact_highlight_table(summary, paths.tables_dir / "fairal_main_results_compact.tex", paths.data_dir / "fairal_main_results_compact.csv")
    make_tests_table(tests, paths.tables_dir / "fairal_tests_vs_uncertainty.tex", paths.data_dir / "fairal_tests_vs_uncertainty_compact.csv")
    make_core_figures(df, fin, paths.figures_dir)

    sens = load_sensitivity_results(results_dir)
    if not sens.empty:
        make_sensitivity_tables_and_figures(sens, paths.tables_dir, paths.figures_dir, paths.data_dir)

    make_deepasd_tables(results_dir, paths.tables_dir, paths.data_dir)
    make_clinical_impact_table(results_dir, paths.tables_dir, paths.data_dir)
    make_master_latex_file(paths.tables_dir)
    make_figures_readme(paths.figures_dir)
    make_artifact_report(paths, summary, tests)

    print(f"Saved paper artifacts to: {paths.out_dir}")
    print(f"Tables: {paths.tables_dir}")
    print(f"Figures: {paths.figures_dir}")
    print(f"Data: {paths.data_dir}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate paper-ready FAIR-AL tables and figures.")
    p.add_argument("--results-dir", type=Path, default=Path("results"), help="Directory containing experiment CSV files.")
    p.add_argument("--out-dir", type=Path, default=Path("paper_artifacts_v2"), help="Output directory for generated artifacts.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_generation(args.results_dir, args.out_dir)


if __name__ == "__main__":
    main()

