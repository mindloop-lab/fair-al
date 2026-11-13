"""R2-6: regenerate the main-text figures from FROZEN data with larger effective fonts.

The submitted figures were drawn at 7.2 x 4.8 in with legend font 8-9 pt and inserted at
0.49\\linewidth (about 3.2 in), so the effective font size on the page was ~4-5 pt.
Here the same frozen numbers are redrawn at single-column geometry (3.5 x 2.6 in) with
9-10 pt fonts, so the effective size stays at 9-10 pt at the same insertion width.

No number is computed here: inputs are the frozen CSVs only.
Outputs: manuscript/src-rev1/fig_rev1/*.pdf
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CODE_ROOT = Path(os.environ.get("FAIRAL_CODE_ROOT", Path(__file__).resolve().parents[1]))
ROOT = Path(os.environ.get("FAIRAL_PROJECT_ROOT",
                           CODE_ROOT.parent if CODE_ROOT.name == "code" else CODE_ROOT.parents[1]))
ITER = ROOT / "results_as_submitted/paper_artifacts/data/fairal_all_iterations.csv"
FINAL = ROOT / "results_as_submitted/paper_artifacts/data/fairal_main_results_full.csv"
OUT = Path(os.environ.get("FAIRAL_FIG_DIR", ROOT / "manuscript/src-rev1/fig_rev1"))

plt.rcParams.update({
    "font.size": 10, "axes.labelsize": 10, "axes.titlesize": 10,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 8.5,
    "axes.linewidth": 0.8, "lines.linewidth": 1.2, "pdf.fonttype": 42,
    "figure.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})

SERIES = [("uncertainty", "Uncertainty"), ("baal", "FAIR-AL-Pareto"),
          ("baal_weighted", "FAIR-AL-Weighted"), ("quota", "Quota sampling")]


def curve(ds: str, clf: str, metric: str, ylabel: str, fname: str) -> None:
    d = pd.read_csv(ITER)
    d = d[(d.dataset == ds) & (d.classifier == clf)]
    if d.empty:
        print(f"  skip {fname}: no rows")
        return
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    for code, label in SERIES:
        s = d[d.strategy == code]
        if s.empty:
            continue
        g = s.groupby("queried_size")[metric].agg(["mean", "std"]).reset_index()
        ax.plot(g.queried_size, g["mean"], label=label)
        ax.fill_between(g.queried_size, g["mean"] - g["std"], g["mean"] + g["std"], alpha=0.18)
    ax.set_xlabel("Queried samples")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25, linewidth=0.5)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=2)
    fig.savefig(os.path.join(OUT, fname))
    plt.close(fig)
    print("  wrote", fname)


def tradeoff(ds: str, clf: str, fname: str) -> None:
    # The revision figure must show the strong composition baselines, which only exist in
    # the v9 enhanced files (the v8 main-results artifact predates them).
    suffix = "_sex46proxy" if ds == "ABIDE_B" else ""
    src = f"{ROOT}/results_as_submitted/enhanced/{ds}_{clf}_quota_baselines{suffix}_v9.csv"
    raw = pd.read_csv(src)
    raw = raw[raw["round"] == raw["round"].max()]
    d = (raw.groupby("strategy")
            .agg(q_bias_mean=("q_bias", "mean"), q_bias_std=("q_bias", "std"),
                 balanced_accuracy_mean=("balanced_accuracy", "mean"),
                 balanced_accuracy_std=("balanced_accuracy", "std"))
            .reset_index())
    if d.empty:
        print(f"  skip {fname}: no rows")
        return
    fig, ax = plt.subplots(figsize=(3.5, 3.15))
    for code, label in [("uncertainty", "Uncertainty"), ("baal", "FAIR-AL-Pareto"),
                        ("baal_weighted", "FAIR-AL-Weighted"), ("baal_hard", "FAIR-AL-Constrained"),
                        ("quota", "Quota sampling"), ("stratified_uncertainty", "Stratified unc.")]:
        s = d[d.strategy == code]
        if s.empty:
            continue
        # the artifact CSV stores per-setting means/stds (one row per strategy)
        qb, qbs = float(s["q_bias_mean"].iloc[0]), float(s["q_bias_std"].iloc[0])
        ba, bas = float(s["balanced_accuracy_mean"].iloc[0]), float(s["balanced_accuracy_std"].iloc[0])
        ax.errorbar(qb, ba, xerr=qbs, yerr=bas,
                    fmt="o", ms=4, capsize=1.5, label=label)
    ax.set_xlabel("Q-Bias (TV distance, lower is better)")
    ax.set_ylabel("Balanced accuracy")
    ax.grid(alpha=0.25, linewidth=0.5)
    # legend under the axes so it never covers the points
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.28),
              ncol=3, handletextpad=0.4, columnspacing=0.9)
    fig.savefig(os.path.join(OUT, fname))
    plt.close(fig)
    print("  wrote", fname)


def main() -> None:
    global OUT
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default=str(OUT),
                    help="figure output directory (default: $FAIRAL_FIG_DIR or "
                         "<project>/manuscript/src-rev1/fig_rev1)")
    args = ap.parse_args()
    OUT = Path(args.out_dir)
    os.makedirs(OUT, exist_ok=True)
    print("regenerating figures from frozen CSVs")
    curve("ABIDE_A", "lr", "q_bias", "Q-Bias (TV)", "fig_rev1_curve_ABIDE_A_lr_q_bias.pdf")
    curve("ABIDE_A", "lr", "balanced_accuracy", "Balanced accuracy",
          "fig_rev1_curve_ABIDE_A_lr_balanced_accuracy.pdf")
    tradeoff("ABIDE_A", "lr", "fig_rev1_tradeoff_ABIDE_A_lr.pdf")
    tradeoff("ABIDE_B", "svm", "fig_rev1_tradeoff_ABIDE_B_svm.pdf")


if __name__ == "__main__":
    main()
