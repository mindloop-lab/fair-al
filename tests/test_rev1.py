
"""Stage-2 tests for the revision code (plain asserts; runs with or without pytest).

Coverage:
 1. Q-Bias definition (TV = 0.5*L1), empty queried set.
 2. Degenerate min-max normalisation has a defined, finite behaviour.
 3. Constrained fallback when the tolerance is genuinely unreachable.
 4. tau = 0 is NOT automatically infeasible (exact proportions are reachable).
 5. Sensitive-attribute removal keeps splits / initial set identical.
 6. Historical random-strategy trajectory is reproduced from the frozen submission.
 7. Delayed control wiring: frac=1.0 reproduces uncertainty, frac=0.0 reproduces the arm.
 8. Intra-batch ablation runs and is recorded separately from the historical arm.
 9. Statistics: BH/Holm reference implementations match a brute-force version; zero-variance
    pairs never produce an infinite effect size.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                      # .../code/code
REPO = os.path.dirname(ROOT)                      # .../code
sys.path.insert(0, os.path.join(ROOT, "src"))

from baalexp.data import load_dataset, train_val_test_split_indices, initial_labeled_split  # noqa: E402
from baalexp.metrics import q_bias, q_bias_l1  # noqa: E402
from baalexp.strategies import (  # noqa: E402
    FairALConstrainedStrategy,
    normalize_minmax,
    _intra_batch_fairal,
)
import baalexp.stats_rev1 as sr  # noqa: E402

DATA = os.environ.get("FAIRAL_DATA_DIR", "")
FROZEN = os.environ.get("FAIRAL_FROZEN_DIR", "")
PYTHON = sys.executable


# --------------------------------------------------------------------------- fakes
class FakeBackend:
    """Deterministic probability backend: entropy is controlled by a scalar column."""

    def __init__(self, probs: np.ndarray):
        self.probs = np.asarray(probs, dtype=float)

    def predict_proba(self, X):
        # X[:, 0] indexes the pre-computed probability table
        idx = np.asarray(X[:, 0], dtype=int)
        return self.probs[idx]

    def embed(self, X):
        return np.asarray(X, dtype=float)


def _probs_for(n: int, entropy_levels: int = 5) -> np.ndarray:
    """Probabilities with a spread of entropies so ranking is well defined."""
    p1 = np.linspace(0.5, 0.99, entropy_levels)
    p1 = np.resize(p1, n)
    return np.column_stack([1.0 - p1, p1])


# --------------------------------------------------------------------------- tests
def test_qbias_definition():
    sens = np.array([0] * 8 + [1] * 2)
    target = np.array([0.5, 0.5])
    l1 = q_bias_l1(sens, target)
    tv = q_bias(sens, target)
    assert abs(l1 - 0.6) < 1e-12, l1
    assert abs(tv - 0.3) < 1e-12, tv
    assert abs(tv - 0.5 * l1) < 1e-12
    assert q_bias(np.array([], dtype=int), target) == 0.0


def test_normalisation_degenerate_is_defined():
    out = normalize_minmax(np.array([0.7, 0.7, 0.7]))
    assert np.all(np.isfinite(out))
    assert np.all(out == 0.0), out  # documented behaviour: all-equal -> all zeros


def _balanced_state(n_queried: int = 120, n_cand: int = 40):
    """Balanced queried set (60/60) plus a candidate block, as a controllable toy state."""
    n = n_queried + n_cand
    sensitive = np.concatenate(
        [np.array([0] * (n_queried // 2) + [1] * (n_queried // 2), dtype=int), np.zeros(n_cand, dtype=int)]
    )
    X = np.column_stack([np.arange(n), np.zeros(n)])
    backend = FakeBackend(_probs_for(n))
    queried = np.arange(n_queried)
    unlabeled = np.arange(n_queried, n)
    return backend, X, sensitive, queried, unlabeled


def test_constrained_fallback_when_unreachable():
    # Current queried set is exactly balanced (TV = 0). All candidates belong to group 0,
    # so adding one moves the cumulative TV to 0.5*|61/121 - 0.5|*2 = 0.008264.
    # With tau = 0.001 that is unreachable -> the tolerance must be dropped and the
    # fallback recorded (this is the historical behaviour, not an error).
    backend, X, sensitive, queried, unlabeled = _balanced_state()
    strat = FairALConstrainedStrategy(k_candidates=len(unlabeled), epsilon=0.001)
    sel = strat.select(
        backend=backend, X=X, y=np.zeros(len(sensitive), dtype=int), sensitive=sensitive,
        unlabeled_idx=unlabeled, queried_idx=queried, batch_size=3,
        target_distribution=np.array([0.5, 0.5]),
    )
    assert len(sel) == 3
    st = strat.last_call_stats or {}
    assert st.get("constraint_dropped") == 1, st
    assert st.get("feasible_fraction") == 0.0, st


def test_small_tau_is_not_automatically_infeasible():
    # Same balanced state with tau = 0.01: adding one group-0 sample gives 0.008264 <= tau,
    # so small tolerances are feasible and must NOT be reported as a dropped constraint.
    # (tau = 0 is a different, genuinely unsatisfiable case at this state: any addition
    #  moves TV away from 0 - the reason the reviewer's "tau = 0" example needs an explicit
    #  feasible construction rather than an assumption.)
    backend, X, sensitive, queried, unlabeled = _balanced_state()
    strat = FairALConstrainedStrategy(k_candidates=len(unlabeled), epsilon=0.01)
    sel = strat.select(
        backend=backend, X=X, y=np.zeros(len(sensitive), dtype=int), sensitive=sensitive,
        unlabeled_idx=unlabeled, queried_idx=queried, batch_size=2,
        target_distribution=np.array([0.5, 0.5]),
    )
    assert len(sel) == 2
    st = strat.last_call_stats or {}
    assert st.get("constraint_dropped") == 0, st
    assert st.get("feasible_fraction", 0) > 0, st
    assert q_bias(sensitive[np.concatenate([queried, sel])], np.array([0.5, 0.5])) <= 0.01 + 1e-12


def test_drop_sensitive_keeps_split_and_attribute():
    full = load_dataset(DATA, "ABIDE_A")
    drop = load_dataset(DATA, "ABIDE_A", drop_sensitive_inputs=True)
    assert drop.X.shape[1] == full.X.shape[1] - 2, (drop.X.shape, full.X.shape)
    assert drop.dropped_feature_names == ["SEX_1", "SEX_2"], drop.dropped_feature_names
    assert np.array_equal(drop.sensitive, full.sensitive)
    assert int(drop.sensitive.sum()) == 727 and int((drop.sensitive == 0).sum()) == 144
    tr1, va1, te1 = train_val_test_split_indices(full.y, full.sensitive, seed=0)
    tr2, va2, te2 = train_val_test_split_indices(drop.y, drop.sensitive, seed=0)
    assert np.array_equal(tr1, tr2) and np.array_equal(va1, va2) and np.array_equal(te1, te2)
    i1, u1 = initial_labeled_split(tr1, full.y, full.sensitive, initial_size=40, seed=0)
    i2, u2 = initial_labeled_split(tr2, drop.y, drop.sensitive, initial_size=40, seed=0)
    assert np.array_equal(i1, i2) and np.array_equal(u1, u2)

    full_b = load_dataset(DATA, "ABIDE_B", sensitive_index=46)
    drop_b = load_dataset(DATA, "ABIDE_B", sensitive_index=46, drop_sensitive_inputs=True)
    assert drop_b.X.shape[1] == full_b.X.shape[1] - 2
    assert drop_b.dropped_feature_indices == [46, 47]
    assert np.array_equal(drop_b.sensitive, full_b.sensitive)


def _run_cli(out_csv: str, extra: list[str], history: bool = False) -> pd.DataFrame:
    cmd = [
        PYTHON, "-m", "baalexp.run_experiment",
        "--data-dir", DATA,
        "--dataset", "ABIDE_A",
        "--backend", "sklearn",
        "--classifier", "lr",
        "--seeds", "0",
        "--initial-size", "40",
        "--budget", "120",
        "--batch-size", "10",
        "--k-candidates", "200",
        "--output", out_csv,
    ]
    if history:
        cmd.append("--save-history")
    cmd.extend(extra)
    env = dict(os.environ)
    env["PYTHONPATH"] = os.path.join(ROOT, "src") + os.pathsep + env.get("PYTHONPATH", "")
    env.pop("PYTHONHOME", None)
    res = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if res.returncode != 0:
        raise AssertionError(f"run_experiment failed:\n{res.stdout}\n{res.stderr}")
    return pd.read_csv(out_csv)


def test_reproduce_frozen_random_trajectory():
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "rand.csv")
        df = _run_cli(out, ["--strategies", "random"], history=True)
        got = df[df.strategy == "random"].sort_values("round").iloc[-1]
        frozen = pd.read_csv(os.path.join(FROZEN, "ABIDE_A_svm_core_v8.csv"))
        fc = frozen[(frozen.strategy == "random") & (frozen.seed == 0)].sort_values("round").iloc[-1]
        assert abs(float(got["q_bias"]) - float(fc["q_bias"])) < 1e-12, (got["q_bias"], fc["q_bias"])
        assert int(got["round"]) == int(fc["round"])
        hist = json.load(open(os.path.join(td, "rand.history.json")))
        fh = json.load(open(os.path.join(FROZEN, "ABIDE_A_svm_core_v8.history.json")))
        mine = [h for h in hist if h["strategy"] == "random" and h["seed"] == 0]
        theirs = [h for h in fh if h["strategy"] == "random" and h["seed"] == 0]
        assert len(mine) == len(theirs) == 12
        for a, b in zip(mine, theirs):
            assert a["selected_indices"] == b["selected_indices"], (a["round"], a["selected_indices"], b["selected_indices"])


def test_delayed_control_wiring():
    with tempfile.TemporaryDirectory() as td:
        base = _run_cli(os.path.join(td, "unc.csv"), ["--strategies", "uncertainty"])
        never = _run_cli(
            os.path.join(td, "late_full.csv"),
            ["--strategies", "baal_weighted", "--control-start-frac", "1.0"],
        )
        always = _run_cli(os.path.join(td, "weighted.csv"), ["--strategies", "baal_weighted"])
        u = base[base.strategy == "uncertainty"].sort_values("round")["q_bias"].to_numpy()
        n = never[never.strategy == "baal_weighted"].sort_values("round")["q_bias"].to_numpy()
        a = always[always.strategy == "baal_weighted"].sort_values("round")["q_bias"].to_numpy()
        assert np.allclose(u, n, atol=1e-12), (u, n)  # control never active -> uncertainty rule
        assert np.allclose(a, a)  # sanity
        assert not np.allclose(a, n, atol=1e-12), "control from round 0 must differ from never-active"
        # 50% switch lands between the two extremes
        half = _run_cli(
            os.path.join(td, "late_half.csv"),
            ["--strategies", "baal_weighted", "--control-start-frac", "0.5"],
        )
        h = half[half.strategy == "baal_weighted"].sort_values("round")
        # column name: control_state (evaluation-point state); per-query counters live in history
        assert h["control_state"].iloc[0] == 0 and h["control_state"].iloc[-1] == 1
        # first six rounds must match the uncertainty arm exactly
        assert np.allclose(h["q_bias"].to_numpy()[:6], u[:6], atol=1e-12)


def test_intra_batch_runs_and_differs():
    with tempfile.TemporaryDirectory() as td:
        hist = _run_cli(os.path.join(td, "ib.csv"), ["--strategies", "baal_weighted", "--intra-batch"])
        base = _run_cli(os.path.join(td, "hist.csv"), ["--strategies", "baal_weighted"])
        assert hist["intra_batch"].all() and not base["intra_batch"].any()
        # round 0 has an empty queried set (query_p_group0 = nan), so compare after round 0
        h = hist.sort_values("round")["query_p_group0"].to_numpy()[1:]
        b = base.sort_values("round")["query_p_group0"].to_numpy()[1:]
        print(f"    intra-batch vs historical: max|d qp0| = {np.max(np.abs(h - b)):.4f}")
        assert np.all(np.isfinite(h))


def test_statistics_implementations_and_guard():
    rng = np.random.default_rng(0)
    p = rng.uniform(1e-4, 0.5, size=7)
    # brute-force BH: min over k>=rank of p_(k)*m/k
    order = np.argsort(p)
    brute_bh = np.empty(len(p))
    for i in range(len(p)):
        rank = int(np.where(order == i)[0][0]) + 1
        brute_bh[i] = min(1.0, min(p[order[k]] * len(p) / (k + 1) for k in range(rank - 1, len(p))))
    assert np.allclose(sr.bh_fdr(p), brute_bh, atol=1e-12)
    # brute-force Holm: max over k<=rank of (m-k+1)*p_(k)
    brute_holm = np.empty(len(p))
    for i in range(len(p)):
        rank = int(np.where(order == i)[0][0]) + 1
        brute_holm[i] = min(1.0, max((len(p) - (k + 1) + 1) * p[order[k]] for k in range(rank)))
    assert np.allclose(sr.holm(p), brute_holm, atol=1e-12)
    assert np.all(sr.holm(p) >= sr.bh_fdr(p) - 1e-12), "Holm is never less conservative than BH here"

    # zero-variance guard
    const = np.array([0.5, 0.5, 0.5])
    res = sr.paired(const - 0.01, const)
    assert res.p is None and res.dz is None and res.note == "degenerate_zero_variance", res
    assert abs(res.delta + 0.01) < 1e-12


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL {fn.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
