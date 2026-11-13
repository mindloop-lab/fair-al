"""Tests for the second-round FAL-CUR implementation (FairKM assignment + Algorithm 1).

Plain asserts; runs with or without pytest:
    PYTHONPATH=src python tests/test_fal_cur_v2.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from baalexp.fal_cur import (  # noqa: E402
    FalCurStrategy,
    _cluster_deviation,
    entropy_from_proba,
    make_fal_cur,
)


class StubBackend:
    """Minimal backend: identity embedding and a fixed two-class probability matrix."""

    def __init__(self, proba: np.ndarray):
        self.proba = np.asarray(proba, dtype=float)

    def embed(self, X):
        return np.asarray(X, dtype=float)

    def predict_proba(self, X):
        return self.proba[: len(X)]


def _proba(n: int, ent: np.ndarray | None = None) -> np.ndarray:
    if ent is None:
        p = np.linspace(0.5001, 0.9, n)
    else:
        lo, hi = float(np.min(ent)), float(np.max(ent))
        span = max(hi - lo, 1e-9)
        p = np.clip(0.999 - 0.45 * (ent - lo) / span, 0.5001, 0.999)
    return np.column_stack([1.0 - p, p])


def test_cluster_deviation_matches_eq4():
    n, pi0 = 100, 0.3
    dev = _cluster_deviation(np.array([10.0, 20.0, 0.0]),
                             np.array([2.0, 10.0, 0.0]), n, pi0)
    assert np.isclose(dev[0], (10 / 100) ** 2 * (0.2 - pi0) ** 2)
    assert np.isclose(dev[1], (20 / 100) ** 2 * (0.5 - pi0) ** 2)
    assert dev[2] == 0.0


def test_fairkm_marginal_assignment_balances_groups():
    rng = np.random.default_rng(0)
    z = np.concatenate([rng.normal(0.0, 0.2, 40), rng.normal(3.0, 0.2, 40)])
    sens = np.concatenate([np.zeros(40, dtype=int), np.ones(40, dtype=int)])
    Z = z[:, None]
    pi0 = float((sens == 0).mean())

    def imbalance(labels):
        # size-weighted mean absolute deviation of the group-0 share from the pool share
        return sum((labels == c).mean() * abs((sens[labels == c] == 0).mean() - pi0)
                   for c in np.unique(labels))

    lab0 = FalCurStrategy(lam_scale=0.0, seed=0)._fair_clusters(Z, sens, 2, pi0)
    lab1 = FalCurStrategy(lam_scale=1.0, seed=0)._fair_clusters(Z, sens, 2, pi0)
    assert np.isclose(imbalance(lab0), 0.5)   # plain KMeans separates the groups completely
    assert imbalance(lab1) < imbalance(lab0)


def test_clustering_is_deterministic():
    rng = np.random.default_rng(1)
    Z = rng.normal(size=(60, 4))
    sens = (np.arange(60) % 3 == 0).astype(int)
    pi0 = float((sens == 0).mean())
    a = FalCurStrategy(seed=3)._fair_clusters(Z, sens, 5, pi0)
    b = FalCurStrategy(seed=3)._fair_clusters(Z, sens, 5, pi0)
    assert np.array_equal(a, b)


def _patched(labels):
    strat = FalCurStrategy(seed=0)
    strat._fair_clusters = lambda Z, sens, k, pi0: np.asarray(labels)  # type: ignore[method-assign]
    return strat


def test_round_robin_over_fairness_ranked_clusters():
    Z = np.array([[0.0], [0.1], [0.2],
                  [5.0], [5.1], [5.2],
                  [9.0], [9.1], [9.2]], float)
    sens = np.array([1, 1, 1, 0, 0, 1, 0, 0, 0])
    ent = np.array([0.9, 0.5, 0.1, 0.8, 0.4, 0.2, 0.7, 0.3, 0.6])
    labels = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    strat = _patched(labels)
    backend = StubBackend(_proba(9, ent))
    picked = strat.select(backend, Z, np.zeros(9), sens, np.arange(9), np.array([], int), 4,
                          np.array([0.5, 0.5]))
    assert len(picked) == 4
    pi0 = float((sens == 0).mean())
    dev = _cluster_deviation(np.array([3.0, 3.0, 3.0]), np.array([0.0, 2.0, 3.0]), 9, pi0)
    order = np.argsort(dev, kind="stable")
    assert int(order[0]) == 1                 # cluster B (share 2/3) is the fairest
    assert int(order[1]) == 2                 # then C (share 1.0), then A (share 0)
    # first round: one sample from each cluster, in fairness order
    assert int(picked[0]) in (3, 4, 5)
    assert int(picked[1]) in (6, 7, 8)
    assert int(picked[2]) in (0, 1, 2)
    # second pass returns to the fairest cluster for its 2nd-ranked sample
    assert int(picked[3]) in (3, 4, 5)
    assert len(set(int(p) for p in picked)) == 4


def test_singleton_cluster_contributes_one_sample():
    Z = np.array([[0.0], [1.0], [2.0], [3.0], [4.0]])
    sens = np.array([0, 0, 1, 1, 1])
    labels = np.array([0, 0, 1, 1, 2])
    strat = _patched(labels)
    backend = StubBackend(_proba(5))
    picked = strat.select(backend, Z, np.zeros(5), sens, np.arange(5), np.array([], int), 3,
                          np.array([0.5, 0.5]))
    assert len(picked) == 3
    assert 4 in picked


def test_selection_ignores_labels():
    rng = np.random.default_rng(2)
    Z = rng.normal(size=(30, 3))
    sens = (np.arange(30) % 2).astype(int)
    labels = np.arange(30) // 10
    strat = _patched(labels)
    backend = StubBackend(_proba(30))
    a = strat.select(backend, Z, np.zeros(30), sens, np.arange(30), np.array([], int), 3,
                     np.array([0.5, 0.5]))
    b = strat.select(backend, Z, np.ones(30), sens, np.arange(30), np.array([], int), 3,
                     np.array([0.5, 0.5]))
    assert np.array_equal(a, b)


def test_variants_registered():
    for code in ("fal_cur", "fal_cur_rawscale", "fal_cur_entclust", "fal_cur_l05", "fal_cur_l2"):
        assert make_fal_cur(code, 0).name.startswith("fal_cur")
    assert make_fal_cur("fal_cur_l2", 0).lam_scale == 4.0


def test_entropy_convention_is_bits():
    ent = entropy_from_proba(np.array([[0.5, 0.5], [1.0, 0.0]]))
    assert np.isclose(ent[0], 1.0)
    assert np.isclose(ent[1], 0.0, atol=1e-6)


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
