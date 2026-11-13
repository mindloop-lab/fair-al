from __future__ import annotations

"""Enhanced manuscript artifacts for the FAIR-AL revision.

This module adds reviewer-requested diagnostics without changing the validated
core v8 CSV files.  It can read the existing main results plus optional enhanced
baseline/sensitivity CSV files, then generates paper-ready figures and LaTeX
snippets under ``results/enhanced_artifacts``.
"""

import argparse
import glob
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data import load_dataset, train_val_test_split_indices
from .naming import PUBLIC_ORDER, public_strategy_name

CORE_PATTERNS = [
    "ABIDE_A_*_core_v8.csv",
    "ABIDE_B_*_core_sex46_v8.csv",
]
ENHANCED_PATTERNS = [
    "enhanced/*quota*.csv",
    "enhanced/*pi_sensitivity*.csv",
    "*_quota_baselines*.csv",
    "*_pi_sensitivity*.csv",
]
SELECTED_METHODS = [
    "Random",
    "Uncertainty",
    "Quota sampling",
    "Stratified uncertainty",
    "FAIR-AL-Pareto",
    "FAIR-AL-Weighted",
    "FAIR-AL-Constrained",
    "Minority heuristic",
    "Reweighting",
]


def safe_name(text: object) -> str:
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


def fmt(x: object, digits: int = 3) -> str:
    try:
        v = float(x)
    except Exception:
        return "--"
    if not np.isfinite(v):
        return "--"
    return f"{v:.{digits}f}"


def mean_std_text(mean: object, std: object, digits: int = 3) -> str:
    return f"{fmt(mean, digits)} $\\pm$ {fmt(std, digits)}"


def read_csvs(results_dir: Path, patterns: list[str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for pat in patterns:
        for path in sorted(results_dir.glob(pat)):
            if path.name.endswith(".env.json") or path.name.endswith(".history.json"):
                continue
            try:
                df = pd.read_csv(path)
            except Exception:
                continue
            if df.empty or "round" not in df.columns:
                continue
            df["source_file"] = str(path.relative_to(results_dir))
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["strategy_public"] = out["strategy"].map(public_strategy_name)
    out["strategy_public"] = out["strategy_public"].fillna(out["strategy"])
    return out


def final_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    key = ["dataset", "backend", "classifier", "strategy", "seed", "source_file"]
    key = [c for c in key if c in df.columns]
    idx = df.groupby(key, dropna=False)["round"].idxmax()
    out = df.loc[idx].copy().reset_index(drop=True)
    out["strategy_public"] = out["strategy"].map(public_strategy_name).fillna(out["strategy"])
    return out


def make_dataset_statistics(data_dir: Path, out_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for dataset, sens_idx, protected_label in [
        ("ABIDE_A", None, "sex column inferred from ABIDE-A table"),
        ("ABIDE_B", 46, "PHENO feature 46 binary proxy"),
    ]:
        data = load_dataset(data_dir, dataset, sensitive_index=sens_idx)
        y = data.y
        a = data.sensitive
        n_sites: object = "--"
        site_note = "not available"
        if dataset == "ABIDE_A" and data.feature_names is not None:
            site_cols = [i for i, name in enumerate(data.feature_names) if str(name).startswith("SITE_ID_")]
            if site_cols:
                site_mat = data.X[:, site_cols]
                assigned = site_mat.argmax(axis=1)
                # Count only sites with at least one assigned subject.
                n_sites = int(len(np.unique(assigned[site_mat.max(axis=1) > 0])))
                site_note = "one-hot SITE_ID columns"
        rows.append(
            {
                "dataset": dataset.replace("_", "-"),
                "n_total": int(len(y)),
                "control_n": int((y == 0).sum()),
                "asd_n": int((y == 1).sum()),
                "group0_n": int((a == 0).sum()),
                "group1_n": int((a == 1).sum()),
                "group1_pct": float((a == 1).mean() * 100),
                "n_sites": n_sites,
                "feature_dim": int(data.X.shape[1]),
                "protected_attribute": protected_label,
                "site_note": site_note,
            }
        )
    stats = pd.DataFrame(rows)
    stats.to_csv(out_dir / "table2a_dataset_statistics.csv", index=False)
    tex_rows = []
    for _, r in stats.iterrows():
        tex_rows.append(
            " & ".join(
                [
                    tex_escape(r["dataset"]),
                    str(r["n_total"]),
                    f"{r['asd_n']} / {r['control_n']}",
                    f"{r['group0_n']} / {r['group1_n']}",
                    str(r["n_sites"]),
                    str(r["feature_dim"]),
                    tex_escape(r["protected_attribute"]),
                ]
            )
            + r" \\"
        )
    tex = r"""
\begin{table}[t]
\centering
\caption{Dataset statistics for ABIDE-derived feature-based ASD screening experiments. ABIDE-B group labels are treated as a binary proxy group rather than externally validated biological sex labels.}
\label{tab:dataset_statistics}
\small
\begin{tabular}{lrrrrrl}
\toprule
Dataset & $N$ & ASD / Control & Group 0 / Group 1 & Sites & Features & Protected-group variable \\
\midrule
""".lstrip()
    tex += "\n".join(tex_rows)
    tex += r"""
\bottomrule
\end{tabular}
\end{table}
"""
    (out_dir / "table2a_dataset_statistics.tex").write_text(tex, encoding="utf-8")
    return stats


def make_site_distribution(data_dir: Path, out_dir: Path) -> None:
    data = load_dataset(data_dir, "ABIDE_A")
    if data.feature_names is None:
        return
    site_cols = [(i, str(n).replace("SITE_ID_", "")) for i, n in enumerate(data.feature_names) if str(n).startswith("SITE_ID_")]
    if not site_cols:
        return
    rows: list[dict[str, object]] = []
    for seed in range(10):
        train_idx, val_idx, test_idx = train_val_test_split_indices(data.y, data.sensitive, seed=seed)
        for split, idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
            mat = data.X[idx][:, [i for i, _ in site_cols]]
            assigned = mat.argmax(axis=1)
            for j, (_, site) in enumerate(site_cols):
                rows.append({"seed": seed, "split": split, "site": site, "n": int((assigned == j).sum())})
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "abide_a_site_distribution_by_split.csv", index=False)
    summary = df.groupby(["split", "site"], as_index=False)["n"].mean()
    pivot = summary.pivot(index="site", columns="split", values="n").fillna(0)
    pivot = pivot[[c for c in ["train", "val", "test"] if c in pivot.columns]]
    pivot.to_csv(out_dir / "abide_a_site_distribution_mean.csv")
    tex = pivot.round(1).reset_index().to_latex(index=False, escape=True, caption="Mean ABIDE-A site counts across the ten random seeds used by the main experiments.", label="tab:site_distribution_abide_a")
    (out_dir / "table_site_distribution_abide_a.tex").write_text(tex, encoding="utf-8")


def aggregate_final(final: pd.DataFrame) -> pd.DataFrame:
    metrics = ["balanced_accuracy", "auc", "q_bias", "eod_abs", "group0_sensitivity", "group1_sensitivity"]
    rows: list[dict[str, object]] = []
    for keys, g in final.groupby(["dataset", "classifier", "strategy", "strategy_public"], dropna=False):
        row = dict(zip(["dataset", "classifier", "strategy", "strategy_public"], keys))
        row["n_seeds"] = int(g["seed"].nunique()) if "seed" in g else len(g)
        for m in metrics:
            if m in g.columns:
                row[f"{m}_mean"] = float(g[m].mean())
                row[f"{m}_std"] = float(g[m].std(ddof=1)) if len(g[m].dropna()) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def make_delta_heatmaps(final: pd.DataFrame, out_dir: Path, baseline: str = "uncertainty") -> pd.DataFrame:
    if final.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    key_cols = ["dataset", "classifier", "source_file"] if "source_file" in final.columns else ["dataset", "classifier"]
    # Use seed-paired deltas when possible, across each source file to avoid mixing sensitivity and core runs.
    for keys, g in final.groupby(key_cols, dropna=False):
        base = g[g["strategy"] == baseline][["seed", "balanced_accuracy", "q_bias"]].rename(
            columns={"balanced_accuracy": "base_ba", "q_bias": "base_qbias"}
        )
        if base.empty:
            continue
        for strategy, sg in g.groupby("strategy"):
            if strategy == baseline:
                continue
            merged = sg.merge(base, on="seed", how="inner")
            if merged.empty:
                continue
            rows.append(
                {
                    **dict(zip(key_cols, keys if isinstance(keys, tuple) else (keys,))),
                    "strategy": strategy,
                    "method": public_strategy_name(strategy),
                    "n_seeds": int(merged["seed"].nunique()),
                    "delta_q_bias_mean": float((merged["q_bias"] - merged["base_qbias"]).mean()),
                    "delta_balanced_accuracy_mean": float((merged["balanced_accuracy"] - merged["base_ba"]).mean()),
                    "delta_q_bias_std": float((merged["q_bias"] - merged["base_qbias"]).std(ddof=1)) if len(merged) > 1 else 0.0,
                    "delta_balanced_accuracy_std": float((merged["balanced_accuracy"] - merged["base_ba"]).std(ddof=1)) if len(merged) > 1 else 0.0,
                }
            )
    delta = pd.DataFrame(rows)
    if delta.empty:
        return delta
    delta.to_csv(out_dir / "all_setting_delta_vs_uncertainty.csv", index=False)

    # Collapse source files for the manuscript heatmaps.
    collapsed = delta.groupby(["dataset", "classifier", "strategy", "method"], as_index=False).agg(
        delta_q_bias_mean=("delta_q_bias_mean", "mean"),
        delta_balanced_accuracy_mean=("delta_balanced_accuracy_mean", "mean"),
        n_seeds=("n_seeds", "max"),
    )
    collapsed.to_csv(out_dir / "all_setting_delta_vs_uncertainty_collapsed.csv", index=False)
    for metric, label, fname in [
        ("delta_q_bias_mean", r"$\Delta$ Q-Bias vs. uncertainty", "heatmap_delta_qbias"),
        ("delta_balanced_accuracy_mean", r"$\Delta$ Balanced accuracy vs. uncertainty", "heatmap_delta_ba"),
    ]:
        plot_df = collapsed.copy()
        plot_df["setting"] = plot_df["dataset"].str.replace("_", "-") + " / " + plot_df["classifier"].str.upper()
        methods = [m for m in SELECTED_METHODS if m in set(plot_df["method"])]
        settings = sorted(plot_df["setting"].unique())
        if not methods or not settings:
            continue
        mat = np.full((len(methods), len(settings)), np.nan)
        for i, method in enumerate(methods):
            for j, setting in enumerate(settings):
                vals = plot_df[(plot_df["method"] == method) & (plot_df["setting"] == setting)][metric]
                if not vals.empty:
                    mat[i, j] = vals.iloc[0]
        fig_w = max(7.0, 0.8 * len(settings) + 3)
        fig_h = max(4.8, 0.35 * len(methods) + 1.5)
        plt.figure(figsize=(fig_w, fig_h))
        im = plt.imshow(mat, aspect="auto")
        plt.colorbar(im, label=label)
        plt.xticks(range(len(settings)), settings, rotation=45, ha="right")
        plt.yticks(range(len(methods)), methods)
        for i in range(len(methods)):
            for j in range(len(settings)):
                if np.isfinite(mat[i, j]):
                    plt.text(j, i, fmt(mat[i, j], 3), ha="center", va="center", fontsize=7)
        plt.title(label)
        plt.tight_layout()
        plt.savefig(out_dir / f"{fname}.pdf")
        plt.savefig(out_dir / f"{fname}.png", dpi=240)
        plt.close()
    tex = r"""
\begin{figure}[t]
\centering
\includegraphics[width=.98\linewidth]{fig/supplement/heatmap_delta_qbias.pdf}
\caption{All-setting change in final Q-Bias relative to uncertainty sampling. Negative values indicate lower demographic query drift.}
\label{fig:delta_qbias_heatmap}
\end{figure}

\begin{figure}[t]
\centering
\includegraphics[width=.98\linewidth]{fig/supplement/heatmap_delta_ba.pdf}
\caption{All-setting change in final balanced accuracy relative to uncertainty sampling.}
\label{fig:delta_ba_heatmap}
\end{figure}
""".lstrip()
    (out_dir / "figure_blocks_delta_heatmaps.tex").write_text(tex, encoding="utf-8")
    return delta


def make_formal_curves(df: pd.DataFrame, out_dir: Path) -> None:
    if df.empty:
        return
    methods = ["Uncertainty", "Quota sampling", "Stratified uncertainty", "FAIR-AL-Weighted", "FAIR-AL-Pareto", "Random"]
    for dataset, clf, metric, ylabel, label_suffix in [
        ("ABIDE_A", "lr", "q_bias", "Q-Bias (TV distance)", "q_bias"),
        ("ABIDE_A", "lr", "balanced_accuracy", "Balanced accuracy", "ba"),
        ("ABIDE_B", "svm", "q_bias", "Q-Bias (TV distance)", "q_bias_abideb_svm"),
    ]:
        g = df[(df["dataset"] == dataset) & (df["classifier"] == clf)]
        if g.empty or metric not in g.columns:
            continue
        agg = g.groupby(["strategy_public", "queried_size"], as_index=False).agg(
            mean=(metric, "mean"), std=(metric, "std")
        )
        plt.figure(figsize=(7.2, 4.8))
        for method in methods:
            sg = agg[agg["strategy_public"] == method].sort_values("queried_size")
            if sg.empty:
                continue
            plt.plot(sg["queried_size"], sg["mean"], marker="o", linewidth=1.6, label=method)
            lo = sg["mean"] - sg["std"].fillna(0)
            hi = sg["mean"] + sg["std"].fillna(0)
            plt.fill_between(sg["queried_size"], lo, hi, alpha=0.12)
        plt.xlabel("Queried labels")
        plt.ylabel(ylabel)
        title_ds = dataset.replace("_", "-")
        clf_name = {"lr": "logistic regression", "svm": "support vector machine", "mlp": "multilayer perceptron"}.get(clf, clf)
        plt.title(f"{title_ds} with {clf_name}")
        plt.legend(fontsize=8, ncol=2)
        plt.tight_layout()
        stem = f"formal_curve_{safe_name(dataset)}_{clf}_{label_suffix}"
        plt.savefig(out_dir / f"{stem}.pdf")
        plt.savefig(out_dir / f"{stem}.png", dpi=240)
        plt.close()

    tex = r"""
\begin{figure}[t]
\centering
\includegraphics[width=.49\linewidth]{fig/formal_curve_ABIDE_A_lr_q_bias.pdf}
\includegraphics[width=.49\linewidth]{fig/formal_curve_ABIDE_A_lr_ba.pdf}
\caption{Active-learning trajectories on ABIDE-A with logistic regression. The left panel shows Q-Bias as TV distance; the right panel shows balanced accuracy. Curves show mean values over seeds with shaded one-standard-deviation bands.}
\label{fig:formal_abidea_lr_trajectories}
\end{figure}
""".lstrip()
    (out_dir / "figure2_formal_curves.tex").write_text(tex, encoding="utf-8")


# def make_formal_tradeoff(final: pd.DataFrame, out_dir: Path) -> None:
#     if final.empty:
#         return
#     methods = ["Random", "Uncertainty", "Quota sampling", "Stratified uncertainty", "FAIR-AL-Pareto", "FAIR-AL-Weighted", "FAIR-AL-Constrained"]
#     for dataset, clf in [("ABIDE_A", "lr"), ("ABIDE_B", "svm")]:
#         g = final[(final["dataset"] == dataset) & (final["classifier"] == clf) & (final["strategy_public"].isin(methods))]
#         if g.empty:
#             continue
#         agg = g.groupby("strategy_public", as_index=False).agg(
#             q_mean=("q_bias", "mean"),
#             q_std=("q_bias", "std"),
#             ba_mean=("balanced_accuracy", "mean"),
#             ba_std=("balanced_accuracy", "std"),
#         )
#         plt.figure(figsize=(6.8, 5.2))
#         for _, r in agg.iterrows():
#             plt.errorbar(r["q_mean"], r["ba_mean"], xerr=0 if math.isnan(r["q_std"]) else r["q_std"], yerr=0 if math.isnan(r["ba_std"]) else r["ba_std"], marker="o", capsize=3)
#             plt.text(r["q_mean"], r["ba_mean"], str(r["strategy_public"]), fontsize=8)
#         plt.xlabel("Final Q-Bias (TV distance) ↓")
#         plt.ylabel("Final balanced accuracy ↑")
#         plt.title(f"Accuracy--query-drift trade-off: {dataset.replace('_','-')} / {clf.upper()}")
#         plt.annotate("Preferred direction", xy=(0.05, 0.95), xycoords="axes fraction", fontsize=8)
#         plt.tight_layout()
#         stem = f"formal_tradeoff_{safe_name(dataset)}_{clf}"
#         plt.savefig(out_dir / f"{stem}.pdf")
#         plt.savefig(out_dir / f"{stem}.png", dpi=240)
#         plt.close()
#     tex = r"""
# \begin{figure}[t]
# \centering
# \includegraphics[width=.49\linewidth]{fig/formal_tradeoff_ABIDE_A_lr.pdf}
# \includegraphics[width=.49\linewidth]{fig/formal_tradeoff_ABIDE_B_svm.pdf}
# \caption{Final accuracy--query-drift trade-off with one-standard-deviation error bars. Lower Q-Bias and higher balanced accuracy are preferred.}
# \label{fig:formal_tradeoff}
# \end{figure}
# """.lstrip()
#     (out_dir / "figure3_formal_tradeoff.tex").write_text(tex, encoding="utf-8")

def filter_main_tradeoff_rows(final: pd.DataFrame) -> pd.DataFrame:
    """Keep rows appropriate for the main-text trade-off figure.

    The main trade-off figure should not average over pi-sensitivity runs,
    because Q-Bias is defined relative to the target distribution pi.
    We keep validated core v8 rows and quota-baseline rows only.
    """
    if final.empty:
        return final.copy()

    out = final.copy()

    if "source_file" in out.columns:
        src = out["source_file"].astype(str)

        keep_core = (
            src.str.contains("core_v8", na=False)
            | src.str.contains("core_sex46_v8", na=False)
        )

        keep_quota = src.str.contains("quota", case=False, na=False)

        # Exclude pi-sensitivity outputs from the main trade-off figure.
        exclude_pi = src.str.contains("pi_sensitivity", case=False, na=False)

        out = out[(keep_core | keep_quota) & (~exclude_pi)].copy()

    if "target_distribution_mode" in out.columns:
        mode = out["target_distribution_mode"].fillna("pool").astype(str)
        out = out[mode.isin(["", "pool", "nan"]) | out["target_distribution_mode"].isna()].copy()

    return out


def make_formal_tradeoff(final: pd.DataFrame, out_dir: Path) -> None:
    """Generate paper-ready trade-off plots without overlapping point labels.

    The old version placed a text label next to every point, which caused
    severe overlap in dense low-Q-Bias regions. This version uses a legend
    instead and keeps only the preferred-direction annotation.
    """
    if final.empty:
        return

    final = filter_main_tradeoff_rows(final)

    if final.empty:
        return


    methods = [
        "Random",
        "Uncertainty",
        "Quota sampling",
        "Stratified uncertainty",
        "FAIR-AL-Pareto",
        "FAIR-AL-Weighted",
        "FAIR-AL-Constrained",
    ]

    # Keep a stable order in the legend.
    method_order = {m: i for i, m in enumerate(methods)}

    for dataset, clf in [("ABIDE_A", "lr"), ("ABIDE_B", "svm")]:
        g = final[
            (final["dataset"] == dataset)
            & (final["classifier"] == clf)
            & (final["strategy_public"].isin(methods))
        ].copy()

        if g.empty:
            continue

        agg = (
            g.groupby("strategy_public", as_index=False)
            .agg(
                q_mean=("q_bias", "mean"),
                q_std=("q_bias", "std"),
                ba_mean=("balanced_accuracy", "mean"),
                ba_std=("balanced_accuracy", "std"),
            )
            .copy()
        )

        agg["order"] = agg["strategy_public"].map(method_order)
        agg = agg.sort_values("order")

        fig, ax = plt.subplots(figsize=(7.2, 5.0))

        for _, r in agg.iterrows():
            q_mean = float(r["q_mean"])
            ba_mean = float(r["ba_mean"])

            q_std = 0.0 if pd.isna(r["q_std"]) else float(r["q_std"])
            ba_std = 0.0 if pd.isna(r["ba_std"]) else float(r["ba_std"])

            # q_std = 0.0 if pd.isna(r["q_std"]) else float(r["q_std"])
            ## Q-Bias is non-negative; avoid drawing error bars below zero.
            xerr = np.array([[min(q_std, q_mean)], [q_std]])

            ax.errorbar(
                float(r["q_mean"]),
                float(r["ba_mean"]),
                # xerr=q_std,
                xerr=xerr,
                yerr=ba_std,
                marker="o",
                # markeredgecolor='none',   # 或设为与 facecolor 相同
                # markeredgewidth=0,        # 边缘宽度设为 0
                linestyle="none",
                capsize=3,
                markersize=5,
                linewidth=1.0,
                label=str(r["strategy_public"]),
            )

        ax.set_xlim(left=0.0)

        ax.set_xlabel("Final Q-Bias (TV distance)")
        ax.set_ylabel("Final balanced accuracy")
        ax.set_title(
            f"Accuracy--query-drift trade-off: "
            f"{dataset.replace('_', '-')} / {clf.upper()}"
        )

        # Add preferred direction as an arrow in axes coordinates.
        ax.annotate(
            "Preferred direction",
            xy=(0.12, 0.88),
            xytext=(0.42, 0.72),
            xycoords="axes fraction",
            textcoords="axes fraction",
            arrowprops={"arrowstyle": "->", "linewidth": 1.0},
            fontsize=9,
        )

        # handles, labels = ax.get_legend_handles_labels()
        # for h in handles:
        #     # errorbar 的图例句柄是 Line2D 对象
        #     if hasattr(h, 'set_markeredgecolor'):
        #         h.set_markeredgecolor('none')
        #     if hasattr(h, 'set_markeredgewidth'):
        #         h.set_markeredgewidth(0)
        #     if hasattr(h, 'set_markerfacecolor'):
        #         # 确保填充色与原始数据点一致
        #         pass  # 通常已正确继承，无需修改

        # ax.legend(
        #     handles=handles,
        #     labels=labels,
        #     loc="center left",
        #     bbox_to_anchor=(1.02, 0.5),
        #     frameon=False,
        #     fontsize=8,
        # )
        # Put legend outside the plotting area to avoid point-label overlap.
        ax.legend(
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=False,
            fontsize=8,
        )

        fig.tight_layout()
        stem = f"formal_tradeoff_{safe_name(dataset)}_{clf}"
        fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
        fig.savefig(out_dir / f"{stem}.png", dpi=240, bbox_inches="tight")
        plt.close(fig)

    tex = r"""
\begin{figure}[t]
\centering
\includegraphics[width=.49\linewidth]{fig/formal_tradeoff_ABIDE_A_lr.pdf}
\includegraphics[width=.49\linewidth]{fig/formal_tradeoff_ABIDE_B_svm.pdf}
\caption{Final balanced-accuracy--query-drift trade-off with one-standard-deviation error bars. Lower Q-Bias and higher balanced accuracy are preferred. Method labels are shown in the legend to avoid overlap. ABIDE-B results are interpreted as proxy-group acquisition-fairness analyses.}
\label{fig:formal_tradeoff}
\end{figure}
""".lstrip()

    (out_dir / "figure3_formal_tradeoff.tex").write_text(tex, encoding="utf-8")


def make_pi_sensitivity(final: pd.DataFrame, out_dir: Path) -> None:
    if final.empty or "target_distribution_mode" not in final.columns:
        return
    g = final[final["source_file"].astype(str).str.contains("pi_sensitivity", na=False)].copy()
    if g.empty:
        return
    g["pi_setting"] = g["target_distribution_mode"].fillna("pool")
    if "target_pi_arg" in g.columns:
        explicit = g["pi_setting"].eq("explicit")
        g.loc[explicit, "pi_setting"] = "explicit=" + g.loc[explicit, "target_pi_arg"].astype(str)
    summary = g.groupby(["dataset", "classifier", "strategy_public", "pi_setting"], as_index=False).agg(
        q_bias_mean=("q_bias", "mean"),
        q_bias_std=("q_bias", "std"),
        balanced_accuracy_mean=("balanced_accuracy", "mean"),
        balanced_accuracy_std=("balanced_accuracy", "std"),
        n_seeds=("seed", "nunique"),
    )
    summary.to_csv(out_dir / "pi_sensitivity_summary.csv", index=False)
    # compact tex table
    display = summary.copy()
    display["Q-Bias"] = [mean_std_text(m, s) for m, s in zip(display["q_bias_mean"], display["q_bias_std"])]
    display["BA"] = [mean_std_text(m, s) for m, s in zip(display["balanced_accuracy_mean"], display["balanced_accuracy_std"])]
    tex = display[["dataset", "classifier", "strategy_public", "pi_setting", "n_seeds", "Q-Bias", "BA"]].rename(
        columns={"dataset": "Dataset", "classifier": "Classifier", "strategy_public": "Method", "pi_setting": "$\\pi$", "n_seeds": "Seeds"}
    ).to_latex(index=False, escape=False, caption="Sensitivity of FAIR-AL results to the target protected-group distribution $\\pi$.", label="tab:pi_sensitivity")
    (out_dir / "table_pi_sensitivity.tex").write_text(tex, encoding="utf-8")
    for dataset, clf in summary[["dataset", "classifier"]].drop_duplicates().itertuples(index=False):
        sg = summary[(summary["dataset"] == dataset) & (summary["classifier"] == clf)]
        if sg.empty:
            continue
        labels = sg["strategy_public"] + "\n" + sg["pi_setting"]
        x = np.arange(len(sg))
        plt.figure(figsize=(max(7, 0.55 * len(sg) + 2), 4.8))
        plt.errorbar(x, sg["q_bias_mean"], yerr=sg["q_bias_std"].fillna(0), marker="o",
          # markeredgecolor='none',   # 或设为与 facecolor 相同
          # markeredgewidth=0,        # 边缘宽度设为 0
          capsize=3)
        plt.xticks(x, labels, rotation=45, ha="right")
        plt.ylabel("Final Q-Bias (TV distance)")
        plt.title(f"Target-distribution sensitivity: {dataset.replace('_','-')} / {clf.upper()}")
        plt.tight_layout()
        stem = f"pi_sensitivity_{safe_name(dataset)}_{clf}"
        plt.savefig(out_dir / f"{stem}.pdf")
        plt.savefig(out_dir / f"{stem}.png", dpi=240)
        plt.close()


def make_threshold_note(out_dir: Path) -> None:
    text = r"""
% Auto-generated threshold clarification snippet.
Unless otherwise stated, the validated v8 main experiments report threshold-dependent metrics at a fixed probability threshold of 0.5.  The enhanced code additionally supports validation-selected thresholds via \texttt{--threshold-mode val\_bacc} or \texttt{--threshold-mode val\_youden}; in those analyses, the threshold is selected on the validation split and then fixed before held-out test evaluation.  AUC and Q-Bias are threshold-independent.
""".lstrip()
    (out_dir / "threshold_rule_clarification.tex").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate enhanced FAIR-AL revision artifacts")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out-dir", default="results/enhanced_artifacts")
    args = parser.parse_args()
    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    main_df = read_csvs(results_dir, CORE_PATTERNS)
    enhanced_df = read_csvs(results_dir, ENHANCED_PATTERNS)
    combined = pd.concat([x for x in [main_df, enhanced_df] if not x.empty], ignore_index=True) if (not main_df.empty or not enhanced_df.empty) else pd.DataFrame()
    combined_final = final_rows(combined)
    combined_final.to_csv(out_dir / "combined_final_rows.csv", index=False)
    if not combined.empty:
        combined.to_csv(out_dir / "combined_curve_rows.csv", index=False)

    make_dataset_statistics(Path(args.data_dir), out_dir)
    make_site_distribution(Path(args.data_dir), out_dir)
    make_delta_heatmaps(combined_final, out_dir)
    make_formal_curves(combined, out_dir)
    make_formal_tradeoff(combined_final, out_dir)
    make_pi_sensitivity(combined_final, out_dir)
    make_threshold_note(out_dir)
    aggregate_final(combined_final).to_csv(out_dir / "combined_final_summary.csv", index=False)
    print(f"Saved enhanced artifacts to {out_dir}")


if __name__ == "__main__":
    main()
