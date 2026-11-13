#!/usr/bin/env python3
"""36: build every revision comparison with the strict pairing core.

Fails loudly on: missing declared endpoint, missing/extra/duplicate seeds, ambiguous
source, unknown adjustment method.  Writes the per-seed pair table that the previous
round lacked, plus the family ledger.

Usage
    python scripts/36_build_final_tables.py --out <dir> [--plan <csv>]

Inputs
    results_rev1/stat/comparisons_manifest.csv   declared comparisons
    results_as_submitted/**, results_rev1/formal/**   result CSVs (read-only)

Outputs (in <dir>)
    comparison_endpoint_pairs.csv
    05_CONTRASTS_FINAL.csv
    06_FAMILY_LEDGER.csv
    guard_probes.csv
    verification.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CODE_ROOT = Path(os.environ.get("FAIRAL_CODE_ROOT", Path(__file__).resolve().parents[1]))
PROJECT_ROOT = Path(os.environ.get("FAIRAL_PROJECT_ROOT", CODE_ROOT.parents[1]))
sys.path.insert(0, str(CODE_ROOT / "src"))

from baalexp import pairing  # noqa: E402

BUDGET = 120.0
SEEDS = "0,1,2,3,4,5,6,7,8,9"


def build_plan(manifest: pd.DataFrame) -> pd.DataFrame:
    plan = manifest.copy()
    for column, default in (("budget", BUDGET), ("expected_seeds", SEEDS),
                            ("split_proof_grade", "reconstructed_protocol")):
        if column not in plan:
            plan[column] = default
    return plan


def probe_guards(out_dir: Path, tmp: Path) -> pd.DataFrame:
    """Reproduce the three reviewer probes plus extra rejection cases."""
    rng = np.random.default_rng(0)
    rows = []

    def frame(seeds, q_end, strategy="fal_cur", metric_value=0.5):
        recs = []
        for s in seeds:
            for q, r in ((0, 0), (10, 1), (q_end, 2)):
                recs.append(dict(dataset="ABIDE_A", classifier="svm", strategy=strategy,
                                 seed=s, round=r, queried_size=q, auc=metric_value + 0.01 * s,
                                 q_bias=0.01, balanced_accuracy=0.6))
        return pd.DataFrame(recs)

    cases = {
        "complete_control": (frame(range(10), 120), frame(range(10), 120, "uncertainty"), 120,
                             SEEDS, "none"),
        "missing_120_endpoint": (frame(range(10), 110), frame(range(10), 110, "uncertainty"), 120,
                                 SEEDS, "none"),
        "one_side_missing_seed9": (frame(range(10), 120), frame(range(9), 120, "uncertainty"), 120,
                                   SEEDS, "none"),
        "duplicate_seed_endpoint": (pd.concat([frame(range(10), 120),
                                               frame([3], 120)]),
                                    frame(range(10), 120, "uncertainty"), 120, SEEDS, "none"),
        "unknown_adjustment": (frame(range(10), 120), frame(range(10), 120, "uncertainty"), 120,
                               SEEDS, "holm_bonferroni"),
    }
    for name, (fa, fb, budget, seeds, method) in cases.items():
        pa, pb = tmp / f"{name}_a.csv", tmp / f"{name}_b.csv"
        fa.to_csv(pa, index=False)
        fb.to_csv(pb, index=False)
        spec = dict(analysis_id=name, dataset="ABIDE_A", classifier="svm", metric="auc",
                    strategy_a="fal_cur", strategy_b="uncertainty",
                    source_a=str(pa), source_b=str(pb), family_id="PROBE",
                    adjustment=method, analysis_status="probe", budget=budget,
                    expected_seeds=seeds)
        try:
            row, _ = pairing.run_comparison(spec, CODE_ROOT, PROJECT_ROOT)
            adj = pairing.adjust([row["p_raw"]], method)
            rows.append({"probe": name, "accepted": True, "n_pairs": row["n_pairs"],
                         "result_status": row["status"], "error": ""})
        except pairing.PairingError as exc:
            rows.append({"probe": name, "accepted": False, "n_pairs": 0,
                         "result_status": type(exc).__name__, "error": str(exc)[:300]})
    # ambiguous basename probe: two files with one basename
    if (PROJECT_ROOT / "code/code/results_rev1/formal/fal_cur").is_dir():
        try:
            pairing.resolve_source("ABIDE_A_svm_falcur.csv", CODE_ROOT, PROJECT_ROOT)
            rows.append({"probe": "ambiguous_basename", "accepted": True, "n_pairs": 0,
                         "result_status": "resolved", "error": ""})
        except pairing.PairingError as exc:
            rows.append({"probe": "ambiguous_basename", "accepted": False, "n_pairs": 0,
                         "result_status": type(exc).__name__, "error": str(exc)[:300]})
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "guard_probes.csv", index=False)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--plan", default=None)
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / "_probe_inputs"
    tmp.mkdir(exist_ok=True)

    plan_path = Path(args.plan) if args.plan else CODE_ROOT / "results_rev1/stat/comparisons_manifest.csv"
    if not plan_path.exists():
        raise SystemExit(
            f"comparison plan not found: {plan_path}\n"
            "Pass --plan <csv> with the declared comparison plan, or point FAIRAL_CODE_ROOT at "
            "the tree that holds results_rev1/stat/comparisons_manifest.csv and "
            "FAIRAL_PROJECT_ROOT at the tree that holds results_as_submitted/ and data/.  "
            "The package alone cannot rebuild the paper tables.")
    plan = build_plan(pd.read_csv(plan_path))
    if plan.empty or plan.analysis_id.isna().any() or not plan.analysis_id.is_unique:
        raise SystemExit("Comparison plan must contain unique nonempty analysis IDs")
    print(f"declared comparisons: {len(plan)}")

    probes = probe_guards(out_dir, tmp)
    print(probes[["probe", "accepted", "n_pairs", "result_status"]].to_string(index=False))

    cache: dict = {}
    rows, pairs, failures = [], [], []
    for spec in plan.to_dict("records"):
        try:
            row, pr = pairing.run_comparison(spec, CODE_ROOT, PROJECT_ROOT, cache)
            rows.append(row)
            pairs.append(pr)
        except pairing.PairingError as exc:
            failures.append({"analysis_id": spec["analysis_id"], "error": str(exc)[:400]})
    if failures:
        failure_path = out_dir / "comparison_failures.json"
        failure_path.write_text(json.dumps(failures, indent=2) + "\n", encoding="utf-8")
        raise SystemExit(f"{len(failures)} comparison(s) failed; see {failure_path}")
    res = pd.DataFrame(rows)
    print(f"comparisons ok: {int((res.status == 'ok').sum())}/{len(res)}")

    # family adjustment over the evaluated rows only
    res["p_adjusted"] = np.nan
    res["family_planned"] = 0
    res["family_testable"] = 0
    for fam, g in res[res.status == "ok"].groupby("family_id"):
        method = str(g["adjustment"].iloc[0]).lower()
        mask = g["p_raw"].notna()
        idx = g.index[mask]
        res.loc[g.index, "family_planned"] = len(g)
        res.loc[g.index, "family_testable"] = len(idx)
        if len(idx):
            res.loc[idx, "p_adjusted"] = pairing.adjust(
                g.loc[idx, "p_raw"].to_numpy(float), method)

    res.to_csv(out_dir / "05_CONTRASTS_FINAL.csv", index=False)
    if pairs:
        pd.concat(pairs, ignore_index=True).to_csv(
            out_dir / "comparison_endpoint_pairs.csv", index=False)
    ledger = pairing.family_ledger(res[res.status == "ok"])
    ledger.to_csv(out_dir / "06_FAMILY_LEDGER.csv", index=False)

    # end-to-end acceptance
    ok = res[res.status == "ok"]
    ids_declared = set(plan.analysis_id.astype(str))
    ids_ok = set(ok.analysis_id.astype(str))
    ver = {
        "declared": len(plan), "evaluated_ok": int(len(ok)),
        "failed": int((res.status != "ok").sum()),
        "ids_match_manifest": ids_declared == ids_ok,
        "ids_missing": sorted(ids_declared - ids_ok)[:10],
        "all_n_pairs_10": bool((ok.n_pairs == 10).all()),
        "n_pairs_histogram": {int(k): int(v) for k, v in ok.n_pairs.value_counts().items()},
        "families": int(ledger.shape[0]),
        "families_declared_holm": int((ledger.declared_adjustment.str.lower() == "holm").sum()),
        "families_declared_bh": int((ledger.declared_adjustment.str.lower() == "bh").sum()),
        "families_declared_none": int((ledger.declared_adjustment.str.lower() == "none").sum()),
        "testable_rows": int(ok.p_raw.notna().sum()),
        "zero_variance_rows": int((ok.note != "ok").sum()),
        "probes": probes.to_dict("records"),
        "failures": failures[:20],
        "columns_checked": ["n_pairs", "mean_a", "mean_b", "delta", "ci95_low", "ci95_high",
                            "p_raw", "p_adjusted", "status", "source_a_sha256",
                            "source_b_sha256", "seeds_paired"],
    }
    with open(out_dir / "verification.json", "w", encoding="utf-8") as fh:
        json.dump(ver, fh, indent=2, ensure_ascii=False)
    print(json.dumps({k: v for k, v in ver.items() if k != "probes"}, indent=2)[:1200])

    # the three reviewer probes must now be rejected except the control
    expect = {"complete_control": True, "missing_120_endpoint": False,
              "one_side_missing_seed9": False, "duplicate_seed_endpoint": False,
              "unknown_adjustment": False}
    bad = [r for r in probes.to_dict("records")
           if r["probe"] in expect and bool(r["accepted"]) != expect[r["probe"]]]
    if bad:
        raise SystemExit(f"GUARD PROBES FAIL: {bad}")
    print("GUARD PROBES PASS")


if __name__ == "__main__":
    main()
