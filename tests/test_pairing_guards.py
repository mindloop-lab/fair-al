"""Guards that must reject incomplete inputs (synthetic tables, no data needed)."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG / "src"))
from baalexp import pairing  # noqa: E402


def _frame(seeds, q_end, strategy="fal_cur"):
    rows = []
    for s in seeds:
        for q, r in ((0, 0), (10, 1), (q_end, 2)):
            rows.append(dict(dataset="ABIDE_A", classifier="svm", strategy=strategy,
                             seed=s, round=r, queried_size=q, auc=0.5 + 0.01 * s,
                             q_bias=0.01, balanced_accuracy=0.6))
    return pd.DataFrame(rows)


def _run(tmp, fa, fb, budget=120, seeds="0,1,2,3,4,5,6,7,8,9", method="none"):
    pa, pb = tmp / "a.csv", tmp / "b.csv"
    fa.to_csv(pa, index=False)
    fb.to_csv(pb, index=False)
    spec = dict(analysis_id="probe", dataset="ABIDE_A", classifier="svm", metric="auc",
                strategy_a="fal_cur", strategy_b="uncertainty", source_a=str(pa),
                source_b=str(pb), family_id="PROBE", adjustment=method,
                analysis_status="probe", budget=budget, expected_seeds=seeds)
    return pairing.run_comparison(spec, PKG, PKG.parent)


def test_complete_control_is_accepted():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        row, pairs = _run(tmp, _frame(range(10), 120), _frame(range(10), 120, "uncertainty"))
        assert row["n_pairs"] == 10 and row["status"] == "ok"
        assert len(pairs) == 10


def test_missing_budget_endpoint_is_rejected():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        try:
            _run(tmp, _frame(range(10), 110), _frame(range(10), 110, "uncertainty"))
        except pairing.MissingEndpoint:
            return
        raise AssertionError("a 110-query endpoint was accepted as if it were 120")


def test_missing_seed_is_rejected():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        try:
            _run(tmp, _frame(range(10), 120), _frame(range(9), 120, "uncertainty"))
        except pairing.SeedCoverageError:
            return
        raise AssertionError("a nine-seed side was silently intersected")


def test_duplicate_seed_endpoint_is_rejected():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        dup = pd.concat([_frame(range(10), 120), _frame([3], 120)])
        try:
            _run(tmp, dup, _frame(range(10), 120, "uncertainty"))
        except pairing.DuplicateEndpoint:
            return
        raise AssertionError("a duplicated seed was averaged")


def test_unknown_adjustment_is_rejected():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        try:
            _run(tmp, _frame(range(10), 120), _frame(range(10), 120, "uncertainty"),
                 method="holm_bonferroni")
        except pairing.UnknownAdjustment:
            return
        raise AssertionError("an undeclared adjustment method was accepted")
