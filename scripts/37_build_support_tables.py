#!/usr/bin/env python3
"""37: the remaining deliverable tables, all from frozen results (no re-run).

  1. R1-A readable summary + the declared Pareto / Weighted / Constrained contrasts,
     with the ABIDE-B/SVM Pareto-vs-quota per-seed detail.
  2. R2-3 process drift: process_drift.csv (budget-axis integral from the first non-empty
     queried set, plus the zero-point sensitivity) for continuous / 50% / 75% control,
     under both batch implementations.
  3. Historical Table 5: independent BH (m = 9) recomputation from the frozen core CSVs,
     compared with the printed table and with the archived erratum.
  4. Seven-level target sensitivity: 14 underlying CSVs copied + the 42-cell table.
  5. Fallback events for all 20 settings with source hashes and explicit denominators.

Usage: python scripts/37_build_support_tables.py --out <dir>
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CODE_ROOT = Path(os.environ.get("FAIRAL_CODE_ROOT", Path(__file__).resolve().parents[1]))
PROJECT_ROOT = Path(os.environ.get("FAIRAL_PROJECT_ROOT", CODE_ROOT.parents[1]))
FROZEN = PROJECT_ROOT / "results_as_submitted"
REV = CODE_ROOT / "results_rev1"
sys.path.insert(0, str(CODE_ROOT / "src"))
from baalexp.pairing import bh_fdr, paired_stats  # noqa: E402

BUDGET = 120.0
LEVELS = {"m0p20": -0.20, "m0p10": -0.10, "m0p05": -0.05, "0p00": 0.0,
          "0p05": 0.05, "0p10": 0.10, "0p20": 0.20}


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def process_integral(df: pd.DataFrame, strategy: str) -> dict:
    """Budget-axis integral of Q-Bias.

    I_main: trapezoid over the recorded points from the first non-empty queried set,
            normalised by (u_end - u_start) so it is a budget-averaged Q-Bias.
    I_zero: the same integral but starting at the u = 0 convention point (Q = 0), which
            brackets the effect of the start convention.
    """
    d = df[df["strategy"] == strategy].sort_values("round")
    if d.empty:
        return {}
    u = d["queried_size"].to_numpy(float) / BUDGET
    q = np.where(np.isfinite(d["q_bias"].to_numpy(float)), d["q_bias"].to_numpy(float), 0.0)
    if len(u) < 2:
        return {"i_main": np.nan, "i_zero": np.nan, "n_points": len(u),
                "u_start": np.nan, "u_end": np.nan}
    start = int(np.argmax(u > 0))
    area = float(np.trapezoid(q[start:], u[start:]))
    span = float(u[-1] - u[start])
    area0 = float(np.trapezoid(q, u))
    return {"i_main": area / span if span > 0 else np.nan,
            "i_zero": area0 / u[-1] if u[-1] > 0 else np.nan,
            "u_start": float(u[start]), "u_end": float(u[-1]), "n_points": int(len(u))}


def r1a_tables(out: Path, contrasts: pd.DataFrame) -> list[str]:
    made = []
    sub = contrasts[(contrasts.status == "ok") &
                    contrasts.family_id.str.startswith(("FAM_R1A|", "FAM_R1Ai|"))].copy()
    sub["family"] = sub.family_id.str.split("|").str[0]
    sub["comparison"] = sub.strategy_a + " - " + sub.strategy_b
    cols = ["family", "dataset", "classifier", "metric", "comparison", "n_pairs",
            "mean_a", "mean_b", "delta", "ci95_low", "ci95_high", "cohen_dz", "p_raw",
            "p_adjusted", "note", "family_planned", "source_a", "source_b"]
    sub[cols].sort_values(["family", "dataset", "classifier", "metric", "comparison"]) \
        .to_csv(out / "R1A_strong_baselines.csv", index=False)
    made.append("R1A_strong_baselines.csv")

    lines = ["# R1-A strong composition baselines (10 seeds, endpoint queried_size = 120)", "",
             "Delta = arm_a - arm_b; negative Q-Bias delta means less query drift. "
             "Holm is applied inside each declared (dataset, classifier, metric) family.", "",
             "| setting | metric | comparison | mean_a | mean_b | delta | 95% CI | p_raw | Holm | n | note |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in sub.sort_values(["family", "dataset", "classifier", "metric", "comparison"]).itertuples():
        lines.append(f"| {r.dataset}/{r.classifier} | {r.metric} | {r.comparison} | "
                     f"{r.mean_a:.4f} | {r.mean_b:.4f} | {r.delta:+.4f} | "
                     f"[{r.ci95_low:+.4f}, {r.ci95_high:+.4f}] | "
                     f"{'' if pd.isna(r.p_raw) else f'{r.p_raw:.4g}'} | "
                     f"{'' if pd.isna(r.p_adjusted) else f'{r.p_adjusted:.4g}'} | "
                     f"{int(r.n_pairs)} | {r.note} |")
    (out / "R1A_strong_baselines.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    made.append("R1A_strong_baselines.md")

    # the one limited advantage the reviewer accepted
    key = contrasts[(contrasts.analysis_id.astype(str) ==
                     "R1A|ABIDE_B|svm|balanced_accuracy|baal-quota")]
    if len(key) == 1:
        k = key.iloc[0]
        pd.DataFrame([{
            "analysis_id": k.analysis_id, "delta": k.delta, "ci95_low": k.ci95_low,
            "ci95_high": k.ci95_high, "p_raw": k.p_raw, "holm_m9": k.p_adjusted,
            "family_planned": k.family_planned, "seeds": k.seeds_paired,
            "source_a": k.source_a, "source_a_sha256": k.source_a_sha256,
            "source_b": k.source_b, "source_b_sha256": k.source_b_sha256,
            "note": "the only comparison whose advantage survives the nine-item family",
        }]).to_csv(out / "R1A_B_SVM_pareto_quota.csv", index=False)
        made.append("R1A_B_SVM_pareto_quota.csv")
    return made


def r23_process(out: Path, cache: dict) -> list[str]:
    rows = []
    import re
    pat = re.compile(r"^(ABIDE_[AB])_(lr|svm|mlp)_(batch_prior|intra_batch)_frac_([0-9.]+)\.csv$")
    for path in sorted(glob.glob(str(REV / "formal/delayed_control/*.csv"))):
        base = os.path.basename(path)
        m = pat.match(base)
        if not m:
            continue
        ds, clf, impl, frac = m.groups()
        df = cache.setdefault(path, pd.read_csv(path))
        for strat in sorted(df["strategy"].unique()):
            for seed, g in df[df.strategy == strat].groupby("seed"):
                r = process_integral(g, strat)
                if not r:
                    continue
                rows.append({"dataset": ds, "classifier": clf, "batch_impl": impl,
                             "control_start_frac": float(frac),
                             "arm": strat, "seed": int(seed), **r,
                             "source": os.path.relpath(path, PROJECT_ROOT),
                             "source_sha256": sha(Path(path))})
    # continuous reference: the frozen core runs (control from the first query)
    for name in ("ABIDE_A_lr_core_v8", "ABIDE_A_svm_core_v8",
                 "ABIDE_B_lr_core_sex46_v8", "ABIDE_B_svm_core_sex46_v8"):
        p = FROZEN / f"{name}.csv"
        if not p.exists():
            continue
        df = cache.setdefault(str(p), pd.read_csv(p))
        ds = "ABIDE_A" if name.startswith("ABIDE_A") else "ABIDE_B"
        clf = str(df["classifier"].unique().item()).lower()
        for strat in ("baal_hard", "baal_weighted", "baal", "uncertainty"):
            if strat not in set(df.strategy):
                continue
            for seed, g in df[df.strategy == strat].groupby("seed"):
                r = process_integral(g, strat)
                if not r:
                    continue
                rows.append({"dataset": ds, "classifier": clf, "batch_impl": "continuous",
                             "control_start_frac": 0.0, "arm": strat, "seed": int(seed), **r,
                             "source": os.path.relpath(p, PROJECT_ROOT),
                             "source_sha256": sha(p)})
    pf = pd.DataFrame(rows)
    pf.to_csv(out / "process_drift.csv", index=False)
    summ = (pf.groupby(["dataset", "classifier", "batch_impl", "control_start_frac", "arm"])
              .agg(n_seeds=("seed", "nunique"), i_main_mean=("i_main", "mean"),
                   i_main_sd=("i_main", "std"), i_zero_mean=("i_zero", "mean"),
                   u_end=("u_end", "max")).reset_index())
    summ.to_csv(out / "process_drift_summary.csv", index=False)
    lines = ["# R2-3 process drift (budget-averaged Q-Bias from the first non-empty queried set)", "",
             "I_main normalises the trapezoid integral over [u_start, u_end]; I_zero includes the "
             "u = 0 convention point (Q = 0) and brackets the start-convention effect.", "",
             "| setting | batch impl | control start | arm | n | I_main | I_zero | u_end |",
             "|---|---|---|---|---|---|---|---|"]
    for r in summ.sort_values(["dataset", "classifier", "batch_impl", "control_start_frac",
                               "arm"]).itertuples():
        lines.append(f"| {r.dataset}/{r.classifier} | {r.batch_impl} | {r.control_start_frac:.2f} | "
                     f"{r.arm} | {int(r.n_seeds)} | {r.i_main_mean:.4f} | "
                     f"{r.i_zero_mean:.4f} | {r.u_end:.2f} |")
    (out / "process_drift.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return ["process_drift.csv", "process_drift_summary.csv", "process_drift.md"]


def table5(out: Path) -> list[str]:
    """Independent BH (m = 9) recomputation of the submitted Table 5."""
    rows = []
    for ds in ("ABIDE_A", "ABIDE_B"):
        for clf in ("lr", "svm", "mlp"):
            name = (f"{ds}_{clf}_core_v8" if ds == "ABIDE_A"
                    else f"{ds}_{clf}_core_sex46_v8")
            p = FROZEN / f"{name}.csv"
            if not p.exists():
                continue
            df = pd.read_csv(p)
            if "uncertainty" not in set(df.strategy):
                continue
            base = df[(df.strategy == "uncertainty") &
                      np.isclose(df.queried_size, BUDGET)].set_index("seed")
            for metric in ("q_bias", "balanced_accuracy", "auc"):
                if metric not in df.columns:
                    continue
                fam = []
                for strat in sorted(set(df.strategy) - {"uncertainty"}):
                    s = df[(df.strategy == strat) &
                           np.isclose(df.queried_size, BUDGET)].set_index("seed")
                    common = sorted(set(base.index) & set(s.index))
                    st = paired_stats(s.loc[common, metric].to_numpy(),
                                      base.loc[common, metric].to_numpy())
                    fam.append(dict(dataset=ds, classifier=clf, metric=metric, arm=strat,
                                    n_pairs=st["n"], delta=st["delta"], cohen_dz=st["dz"],
                                    p_raw=st["p"], note=st["note"], seeds=common,
                                    source=os.path.relpath(p, PROJECT_ROOT),
                                    source_sha256=sha(p)))
                if not fam:
                    continue
                fam_df = pd.DataFrame(fam)
                testable = fam_df.p_raw.notna()
                fam_df["p_adj_bh"] = np.nan
                if testable.any():
                    fam_df.loc[testable, "p_adj_bh"] = bh_fdr(
                        fam_df.loc[testable, "p_raw"].to_numpy(float))
                fam_df["family_size"] = int(len(fam_df))
                rows.append(fam_df)
    bh = pd.concat(rows, ignore_index=True)
    bh["seeds"] = bh.seeds.apply(lambda s: ",".join(map(str, s)))
    bh.to_csv(out / "table5_independent_bh.csv", index=False)

    # compare with the archived erratum (printed vs recomputed)
    er = pd.read_csv(REV / "stat/table5_erratum.csv")
    name_map = {"FAIR-AL-Pareto": "baal", "FAIR-AL-Weighted": "baal_weighted",
                "FAIR-AL-Constrained": "baal_hard", "FAIR-AL-RandomPareto": "baal_random_pareto",
                "Minority heuristic": "heuristic", "Random": "random", "CoreSet": "coreset",
                "BADGE": "badge", "Reweighting": "reweighting"}
    metric_map = {"Q-Bias": "q_bias", "Bal. Acc.": "balanced_accuracy", "AUC": "auc"}
    out_rows = []
    for r in er.itertuples():
        arm = name_map.get(r.method)
        metric = metric_map.get(r.metric)
        hit = bh[(bh.dataset.str.replace("_", "-") == r.dataset) &
                 (bh.classifier == str(r.classifier).lower()) & (bh.metric == metric) &
                 (bh.arm == arm)]
        mine = hit.iloc[0] if len(hit) else None
        sig_printed = bool(pd.notna(r.padj_printed) and float(r.padj_printed) < 0.05)
        sig_new = bool(mine is not None and pd.notna(mine.p_adj_bh) and float(mine.p_adj_bh) < 0.05)
        out_rows.append({
            "dataset": r.dataset, "classifier": r.classifier, "method": r.method, "metric": r.metric,
            "delta_printed": r.delta_printed, "dz_printed": r.dz_printed,
            "padj_printed": r.padj_printed,
            "delta_recomputed": None if mine is None else mine.delta,
            "dz_recomputed": None if mine is None else mine.cohen_dz,
            "p_raw_recomputed": None if mine is None else mine.p_raw,
            "p_adj_bh_recomputed": None if mine is None else mine.p_adj_bh,
            "family_size": None if mine is None else int(mine.family_size),
            "significant_printed": sig_printed, "significant_recomputed": sig_new,
            "significance_changed": sig_printed != sig_new,
            "agrees_with_archived_erratum": bool(
                mine is not None and pd.notna(r.p_adj_bh_recomputed)
                and abs(float(mine.p_adj_bh) - float(r.p_adj_bh_recomputed)) < 1e-6),
            "source": None if mine is None else mine.source,
            "source_sha256": None if mine is None else mine.source_sha256,
            "seeds": None if mine is None else mine.seeds,
        })
    t5 = pd.DataFrame(out_rows)
    t5.to_csv(out / "table5_correction.csv", index=False)
    return ["table5_independent_bh.csv", "table5_correction.csv"]


def seven_tier(out: Path) -> list[str]:
    rows, copies = [], []
    tdir = out / "seven_tier_sources"
    tdir.mkdir(exist_ok=True)
    for p in sorted(glob.glob(str(FROZEN / "*_sens_pop_*_v8.csv"))):
        shutil.copy2(p, tdir / os.path.basename(p))
        copies.append(os.path.basename(p))
        base = os.path.basename(p)
        parts = base.replace(".csv", "").split("_sens_pop_")
        setting, lvl = parts[0], parts[1].replace("_v8", "")
        d = pd.read_csv(p)
        ds = str(d["dataset"].unique().item())
        clf = str(d["classifier"].unique().item()).lower()
        ep = d[np.isclose(d.queried_size, BUDGET)]
        for strat, g in ep.groupby("strategy"):
            rows.append({"dataset": ds, "classifier": clf,
                         "sensitive_key": "sex46proxy" if "sex46" in setting else "SEX_1",
                         "delta_pool_pi": LEVELS.get(lvl, np.nan), "level_tag": lvl,
                         "strategy": strat,
                         "target_p_group0": float(g.target_p_group0.mean()),
                         "realized_p_group0_mean": float(g.query_p_group0.mean()),
                         "realized_p_group0_sd": float(g.query_p_group0.std(ddof=1)),
                         "q_bias_mean": float(g.q_bias.mean()),
                         "q_bias_sd": float(g.q_bias.std(ddof=1)),
                         "balanced_accuracy_mean": float(g.balanced_accuracy.mean()),
                         "balanced_accuracy_sd": float(g.balanced_accuracy.std(ddof=1)),
                         "n_seeds": int(g.seed.nunique()), "expected_seeds": "0,1,2,3,4",
                         "endpoint": int(BUDGET), "source_file": base,
                         "source_sha256": sha(Path(p))})
    st = pd.DataFrame(rows).sort_values(["dataset", "strategy", "delta_pool_pi"])
    st.to_csv(out / "seven_tier_table.csv", index=False)
    return ["seven_tier_table.csv", f"seven_tier_sources/ ({len(copies)} CSVs)"]


def fallback(out: Path) -> list[str]:
    rows = []
    settings = []
    for p in sorted(glob.glob(str(REV / "formal/lambda_tau/*.history.json"))) + \
            sorted(glob.glob(str(REV / "formal/fal_cur*/*.history.json"))) + \
            sorted(glob.glob(str(REV / "formal/delayed_control/*.history.json"))) + \
            sorted(glob.glob(str(FROZEN / "*.history.json"))):
        hist = json.load(open(p))
        dfh = pd.DataFrame(hist)
        if dfh.empty or "strategy" not in dfh:
            continue
        rel = os.path.relpath(p, PROJECT_ROOT)
        digest = sha(Path(p))
        arms = sorted(dfh.strategy.unique())
        for (ds, strat), g in dfh.groupby(["dataset", "strategy"]):
            if "constraint_dropped" not in g:
                continue
            dropped = int(g["constraint_dropped"].sum())
            frac = g["feasible_fraction"].astype(float)
            nonfinite = int((~np.isfinite(frac)).sum())
            feasible = frac.fillna(0.0)
            n_nonempty_partial = int(((feasible > 0) & (feasible < 1.0)).sum())
            rows.append({"source": rel, "source_sha256": digest, "dataset": ds, "arm": strat,
                         "n_query_events": int(len(g)),
                         "n_unique_keys": int(g.drop_duplicates(["seed", "query_round"]).shape[0]),
                         "n_seeds": int(g.seed.nunique()), "n_rounds": int(g.query_round.nunique()),
                         "n_arms_in_file": len(arms),
                         "fallback_events": dropped,
                         "rate_over_arm_events": dropped / len(g) if len(g) else np.nan,
                         "n_partial_feasible": n_nonempty_partial,
                         "n_nonfinite_feasible_fraction": nonfinite,
                         "feasible_frac_mean": float(feasible.mean())})
    fb = pd.DataFrame(rows)
    fb.to_csv(out / "fallback_events_by_setting.csv", index=False)
    return ["fallback_events_by_setting.csv"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    contrasts = pd.read_csv(out / "05_CONTRASTS_FINAL.csv")
    cache: dict = {}
    made = []
    made += r1a_tables(out, contrasts)
    made += r23_process(out, cache)
    made += table5(out)
    made += seven_tier(out)
    made += fallback(out)
    (out / "tables_manifest.json").write_text(json.dumps(made, indent=2), encoding="utf-8")
    print("wrote:", *made, sep="\n  ")


if __name__ == "__main__":
    main()
