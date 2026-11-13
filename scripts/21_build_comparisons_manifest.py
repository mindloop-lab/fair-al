"""Build the single authoritative comparisons manifest for the revision.

The manifest is written BEFORE the formal runs and fixes, per analysis:
which dataset / classifier / metric / arm pair / family / adjustment method, and why.

Rules encoded here (agreed after the 2026-10-29 external review):
  * A family is never split by significance: every contrast that bears on one claim sits
    in the same family and is adjusted together.
  * Historical FAIR-AL is the primary arm; the intra-batch variant is a separate,
    independently named implementation.
  * Analyses performed on already-inspected historical results are labelled
    RETROSPECTIVE_REVISION_ANALYSIS, not "pre-registered".
  * New runs whose design was fixed before the full sweep (but whose seed-0 smoke had been
    seen) are labelled NEW_RUN_DESIGN_FIXED_SEED0_SMOKE_SEEN.
"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

# Resolved from the environment or the file location: no machine-specific absolute path is
# needed to run this script (see README).
CODE_ROOT = Path(os.environ.get("FAIRAL_CODE_ROOT", Path(__file__).resolve().parents[1]))
PROJECT_ROOT = Path(os.environ.get("FAIRAL_PROJECT_ROOT", CODE_ROOT.parents[1]))
OUT = CODE_ROOT / "results_rev1/stat/comparisons_manifest.csv"
FROZEN = PROJECT_ROOT / "results_as_submitted"
FORMAL = "results_rev1/formal"

DS_CLF = [("ABIDE_A", "lr"), ("ABIDE_A", "svm"), ("ABIDE_A", "mlp"),
          ("ABIDE_B", "lr"), ("ABIDE_B", "svm"), ("ABIDE_B", "mlp")]
# datasets/classifiers covered by the new runs (MLP excluded: the ablation question is
# about acquisition mechanics, and MLP adds no information while tripling cost)
DS_CLF_NEW = [("ABIDE_A", "lr"), ("ABIDE_A", "svm"), ("ABIDE_B", "lr"), ("ABIDE_B", "svm")]

FAIR_AL_MAIN = ["baal", "baal_weighted", "baal_hard"]
COMPOSITION = ["quota", "stratified_uncertainty"]
METRICS_MAIN = ["q_bias", "balanced_accuracy"]
METRICS_R1A = ["q_bias", "balanced_accuracy", "auc"]


def v9_quota(ds: str, clf: str) -> str:
    suffix = "_sex46proxy" if ds == "ABIDE_B" else ""
    return f"{ds}_{clf}_quota_baselines{suffix}_v9.csv"


def v9_pi(ds: str, clf: str, mode: str) -> str:
    suffix = "_sex46proxy" if ds == "ABIDE_B" else ""
    return f"{ds}_{clf}_pi_sensitivity_{mode}{suffix}_v9.csv"


def core(ds: str, clf: str) -> str:
    suffix = "_sex46" if ds == "ABIDE_B" else ""
    return f"{ds}_{clf}_core{suffix}_v8.csv"


def rows() -> list[dict]:
    out: list[dict] = []

    # ---------------------------------------------------------------- R1-A (historical v9)
    r1a_contrasts = [
        ("quota", "uncertainty"),
        ("stratified_uncertainty", "uncertainty"),
        ("baal", "uncertainty"),
        ("baal_weighted", "uncertainty"),
        ("baal_hard", "uncertainty"),
        ("baal", "quota"),
        ("baal_weighted", "quota"),
        ("baal", "stratified_uncertainty"),
        ("baal_weighted", "stratified_uncertainty"),
    ]
    for ds, clf in DS_CLF:
        for metric in METRICS_R1A:
            for a, b in r1a_contrasts:
                # the constrained arm only exists in the frozen core file; the composition
                # controls only exist in the v9 enhanced file. Both share seeds and splits,
                # so a paired comparison across the two files is valid and is declared here.
                src = core(ds, clf) if "baal_hard" in (a, b) else v9_quota(ds, clf)
                out.append(dict(
                    analysis_id=f"R1A|{ds}|{clf}|{metric}|{a}-{b}",
                    dataset=ds, classifier=clf, metric=metric,
                    strategy_a=a, strategy_b=b,
                    source_a=src, source_b=src,
                    family_id=f"FAM_R1A|{ds}|{clf}|{metric}",
                    adjustment="holm",
                    analysis_status="RETROSPECTIVE_REVISION_ANALYSIS",
                    rationale="Strong composition controls vs FAIR-AL; all contrasts bearing on "
                              "R1-A in one family, no splitting by significance.",
                ))

    # ---------------------------------------------------------------- R1-B (pi targets, descriptive)
    for ds, clf in DS_CLF_NEW:
        for metric in METRICS_MAIN:
            for mode in ["pool", "balanced", "explicit"]:
                out.append(dict(
                    analysis_id=f"R1B|{ds}|{clf}|{metric}|{mode}",
                    dataset=ds, classifier=clf, metric=metric,
                    strategy_a="baal_weighted", strategy_b="uncertainty",
                    source_a=v9_pi(ds, clf, mode), source_b=v9_pi(ds, clf, mode),
                    family_id=f"FAM_R1B|{ds}|{clf}|{metric}",
                    adjustment="none",
                    analysis_status="DESCRIPTIVE",
                    rationale="Target-distribution sensitivity. Q-Bias references differ across "
                              "targets, so these rows are descriptive and NOT tested as one family.",
                ))

    # ---------------------------------------------------------------- S2-3 attribute removal
    for ds, clf in DS_CLF_NEW:
        for metric in METRICS_R1A:
            for strat in ["uncertainty", "quota", "stratified_uncertainty", "baal", "baal_weighted", "baal_hard"]:
                out.append(dict(
                    analysis_id=f"S23|{ds}|{clf}|{metric}|{strat}",
                    dataset=ds, classifier=clf, metric=metric,
                    strategy_a=strat, strategy_b=strat,
                    # delta is reported as (attribute removed) - (attribute kept).
                    # The kept arm comes from the frozen submission: the core file for the
                    # FAIR-AL / uncertainty arms, the v9 enhanced file for quota / stratified.
                    source_a=f"{FORMAL}/drop_sensitive/{ds}_{clf}_dropsens.csv",
                    source_b=(v9_quota(ds, clf)
                              if strat in ("quota", "stratified_uncertainty") else core(ds, clf)),
                    family_id=f"FAM_S23|{ds}|{clf}|{metric}",
                    adjustment="holm",
                    analysis_status="NEW_RUN_DESIGN_FIXED_SEED0_SMOKE_SEEN",
                    rationale="Sensitive-attribute columns removed from model input, sampler keeps "
                              "the attribute, splits/initial set unchanged. Kept arm = frozen "
                              "submission (historical code path).",
                ))

    # ---------------------------------------------------------------- S2-4 batch-prior vs intra
    for ds, clf in DS_CLF_NEW:
        for metric in METRICS_MAIN:
            for strat in ["baal", "baal_weighted", "baal_hard", "uncertainty", "quota", "stratified_uncertainty"]:
                out.append(dict(
                    analysis_id=f"S24|{ds}|{clf}|{metric}|{strat}",
                    dataset=ds, classifier=clf, metric=metric,
                    strategy_a=strat, strategy_b=strat,
                    # delta is reported as (intra-batch) - (batch-prior)
                    source_a=f"{FORMAL}/intra_batch/{ds}_{clf}_intra_batch.csv",
                    source_b=f"{FORMAL}/intra_batch/{ds}_{clf}_batch_prior.csv",
                    family_id=f"FAM_S24|{ds}|{clf}|{metric}",
                    adjustment="holm",
                    analysis_status="NEW_RUN_DESIGN_FIXED_SEED0_SMOKE_SEEN",
                    rationale="Batch-prior (historical) vs intra-batch (new variant) acquisition "
                              "under the same candidate rule and budget.",
                ))

    # ------------------------------------------- R1-A follow-up on the intra-batch runs
    # Declared AFTER seeing the S2-4 results (the finding that the sequential variant closes
    # the drift gap to quota). Recorded as post hoc, family-wide, never split by significance.
    for ds, clf in DS_CLF_NEW:
        for metric in METRICS_MAIN:
            for a in ["baal_weighted", "baal"]:
                for b in ["quota", "stratified_uncertainty"]:
                    out.append(dict(
                        analysis_id=f"R1Ai|{ds}|{clf}|{metric}|{a}-{b}",
                        dataset=ds, classifier=clf, metric=metric,
                        strategy_a=a, strategy_b=b,
                        source_a=f"{FORMAL}/intra_batch/{ds}_{clf}_intra_batch.csv",
                        source_b=f"{FORMAL}/intra_batch/{ds}_{clf}_intra_batch.csv",
                        family_id=f"FAM_R1Ai|{ds}|{clf}|{metric}",
                        adjustment="holm",
                        analysis_status="POST_HOC_AFTER_SEEING_S24_RESULTS",
                        rationale="When does joint optimisation beat direct composition control? "
                                  "Intra-batch FAIR-AL vs quota / stratified at the same budget. "
                                  "Added after the S2-4 runs were seen; reported as post hoc.",
                    ))

    # ---------------------------------------------------------------- R2-1 lambda grid
    for ds, clf in DS_CLF_NEW:
        for metric in METRICS_MAIN:
            for lam in ["0.2", "0.35", "0.65", "0.8"]:
                out.append(dict(
                    analysis_id=f"R21LAM|{ds}|{clf}|{metric}|{lam}",
                    dataset=ds, classifier=clf, metric=metric,
                    strategy_a="baal_weighted", strategy_b="baal_weighted",
                    source_a=f"{FORMAL}/lambda_tau/{ds}_{clf}_lambda_{lam}.csv",
                    source_b=f"{FORMAL}/lambda_tau/{ds}_{clf}_lambda_0.5.csv",
                    family_id=f"FAM_R21LAM|{ds}|{clf}|{metric}",
                    adjustment="holm",
                    analysis_status="NEW_RUN_DESIGN_FIXED_SEED0_SMOKE_SEEN",
                    rationale="Each lambda vs the pre-set default 0.5; sensitivity only, the "
                              "default is never re-selected from these results.",
                ))

    # ---------------------------------------------------------------- R2-1 tau grid
    for ds, clf in DS_CLF_NEW:
        for metric in METRICS_MAIN:
            for tau in ["0.01", "0.02", "0.05", "0.08"]:
                out.append(dict(
                    analysis_id=f"R21TAU|{ds}|{clf}|{metric}|{tau}",
                    dataset=ds, classifier=clf, metric=metric,
                    strategy_a="baal_hard", strategy_b="baal_hard",
                    source_a=f"{FORMAL}/lambda_tau/{ds}_{clf}_tau_{tau}.csv",
                    source_b=f"{FORMAL}/lambda_tau/{ds}_{clf}_tau_0.03.csv",
                    family_id=f"FAM_R21TAU|{ds}|{clf}|{metric}",
                    adjustment="holm",
                    analysis_status="NEW_RUN_DESIGN_FIXED_SEED0_SMOKE_SEEN",
                    rationale="Each tau vs the pre-set default 0.03, including tolerance levels "
                              "that trigger the recorded fallback.",
                ))

    # ---------------------------------------------------------------- R2-2 external method
    for ds, clf in DS_CLF_NEW:
        for metric in METRICS_R1A:
            for b in ["uncertainty", "quota", "stratified_uncertainty", "baal_weighted"]:
                out.append(dict(
                    analysis_id=f"R22|{ds}|{clf}|{metric}|fal_cur-{b}",
                    dataset=ds, classifier=clf, metric=metric,
                    strategy_a="fal_cur", strategy_b=b,
                    source_a=f"{FORMAL}/fal_cur_v2/{ds}_{clf}_falcur.csv",
                    source_b=f"{FORMAL}/fal_cur_v2/{ds}_{clf}_falcur.csv",
                    family_id=f"FAM_R22|{ds}|{clf}|{metric}",
                    adjustment="holm",
                    analysis_status="NEW_RUN_IMPLEMENTATION_CORRECTED_20261103",
                    rationale="External fair-AL comparison against OUR implementation of the "
                              "published FAL-CUR algorithm (FairKM marginal clustering, Eq. 4 "
                              "cluster ranking, Eq. 7 score, Algorithm 1 round-robin; choices "
                              "D1-D6 in 02_FAL_CUR_FIDELITY_AUDIT.md), never the authors' "
                              "script. The 2026-10-30 run used a superseded fixed-penalty "
                              "clustering and is not used for any reported number; the "
                              "implementation was corrected before this run, not chosen by "
                              "outcome.",
                ))

    # ---------------------------------------------------------------- R2-3 delayed control
    for ds, clf in DS_CLF_NEW:
        for mode in ["batch_prior", "intra_batch"]:
            for metric in METRICS_MAIN:
                for frac in ["0.5", "0.75"]:
                    for strat in ["baal_weighted", "baal_hard"]:
                        out.append(dict(
                            analysis_id=f"R23|{ds}|{clf}|{mode}|{metric}|{strat}|{frac}",
                            dataset=ds, classifier=clf, metric=metric,
                            strategy_a=strat, strategy_b=strat,
                            source_a=f"{FORMAL}/delayed_control/{ds}_{clf}_{mode}_frac_{frac}.csv",
                            source_b=f"{FORMAL}/delayed_control/{ds}_{clf}_{mode}_frac_0.0.csv",
                            family_id=f"FAM_R23|{ds}|{clf}|{mode}|{metric}",
                            adjustment="holm",
                            analysis_status="NEW_RUN_DESIGN_FIXED_SEED0_SMOKE_SEEN",
                            rationale="Delayed enablement vs control from the start, both batch "
                                      "implementations, Weighted and Constrained.",
                        ))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(OUT),
                    help="output manifest CSV (default: <code root>/results_rev1/stat/comparisons_manifest.csv)")
    args = ap.parse_args()
    out = Path(args.out)
    rows_ = rows()
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_[0].keys()))
        w.writeheader()
        w.writerows(rows_)
    fams = sorted({r["family_id"] for r in rows_})
    print(f"wrote {out}: {len(rows_)} comparisons, {len(fams)} families")
    for f in fams:
        print(f"  {f}  m={sum(1 for r in rows_ if r['family_id'] == f)}")


if __name__ == "__main__":
    main()
