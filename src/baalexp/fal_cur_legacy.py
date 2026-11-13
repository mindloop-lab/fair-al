"""SUPERSEDED 2026-11-03 — first-round FAL-CUR baseline, kept for provenance only.

The fidelity audit of 2026-11-03 (02_FAL_CUR_FIDELITY_AUDIT.md) found that this module's
clustering is NOT the FairKM mechanism the FAL-CUR paper relies on: the cluster-level share
deviation enters every candidate's assignment cost as a fixed per-cluster penalty, with no
marginal effect of a candidate joining or leaving a cluster. The scored implementation is
``fal_cur.py``. This file is retained only so the superseded CSVs in
``results_rev1/formal/fal_cur/`` remain reproducible; it is never used for manuscript numbers.

Original header (2026-10-30): FAL-CUR comparison baseline (NOT the authors' code).

Reference: FAL-CUR — Fair Active Learning using fair Clustering, Uncertainty and
Representativeness (arXiv:2209.12756v2). The method has two stages:
  (1) fair clustering of the unlabeled pool (inspired by FairKM, i.e. KMeans plus a
      fairness term on sensitive-attribute proportions inside clusters);
  (2) per-cluster selection of the sample maximising
          score(x) = beta * Rep(x) + (1 - beta) * H(x),   beta = 0.6 (paper value),
      where Rep is a distance-based representativeness score and H the predictive entropy.

Why this is a reimplementation and not a copy of the public script
------------------------------------------------------------------
The only public implementation found (github.com/rickymaulanafajri/FAL-CUR) does NOT
implement the paper's fairness-constrained clustering: its ``fairKMeans`` clusters
[features, entropy] with plain KMeans and then keeps only points whose normalised
distance to their centroid falls in [0.3, 0.6]. That is a similarity band, not a
demographic-fairness constraint, and the script performs a single one-shot selection with
k = 180 rather than per-round batches. It also carries no licence, so it cannot be
redistributed. This module therefore implements the published algorithm and records every
adaptation explicitly (see FAL_CUR_DEVIATIONS below).

Documented deviations / choices (reported in the manuscript and in
METHOD_COMPATIBILITY.md):
  D1  Clustering: penalised KMeans whose assignment cost is
          ||z - mu_c||^2 + fair_weight * |share_group0(c) - pi0(pool)|
      i.e. the fairness term of FairKM applied per cluster (the paper ranks clusters by a
      cluster-level fairness deviation; we use it inside the assignment). fair_weight is a
      hyperparameter fixed before any run (default 1.0).
  D2  k = batch size per round (paper: k = 180 one-shot), so the shared AL protocol
      (initial 40, budget 120, batch 10) can be used unchanged.
  D3  Rep(x) = mean Euclidean distance from x to the other members of its cluster;
      Rep and H are min-max normalised WITHIN each cluster before mixing, so that beta
      has a stable meaning (the paper mixes unnormalised quantities).
  D4  Clusters with a single member are dropped from the per-cluster pick and the batch is
      completed by highest entropy outside the selected set.
  D5  Only unlabeled features, predicted probabilities and the sensitive attribute are
      used. No labels, no test information.

Every deviation is also written into the run CSV via the ``strategy`` name
(``fal_cur``) and this docstring, so results cannot be mistaken for the authors' script.
"""
from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans

from .metrics import EPS, distribution


def entropy_from_proba(proba: np.ndarray) -> np.ndarray:
    """Predictive entropy in bits (local copy to avoid a circular import)."""
    proba = np.clip(np.asarray(proba, dtype=float), EPS, 1.0)
    return -np.sum(proba * np.log2(proba), axis=1)

FAL_CUR_DEVIATIONS = ["D1_penalised_kmeans", "D2_k_equals_batch", "D3_within_cluster_norm",
                      "D4_singleton_clusters_dropped", "D5_no_labels_used"]


class FalCurV1Strategy:
    """First-round (superseded) FAL-CUR baseline; see module docstring."""

    def __init__(self, beta: float = 0.6, fair_weight: float = 1.0, max_iter: int = 20,
                 seed: int = 0, name: str = "fal_cur_v1"):
        self.beta = float(beta)
        self.fair_weight = float(fair_weight)
        self.max_iter = int(max_iter)
        self.seed = int(seed)
        self.name = name

    # ------------------------------------------------------------------ clustering
    def _fair_clusters(self, Z: np.ndarray, sens: np.ndarray, k: int, pi0: float) -> np.ndarray:
        n = len(Z)
        k = max(1, min(k, n))
        init = KMeans(n_clusters=k, random_state=self.seed, n_init=1, max_iter=50).fit(Z)
        labels = init.labels_.copy()
        centroids = init.cluster_centers_.copy()
        for _ in range(self.max_iter):
            shares = np.array([float((sens[labels == c] == 0).mean()) if np.any(labels == c) else pi0
                               for c in range(k)])
            pen = self.fair_weight * np.abs(shares - pi0)
            d2 = ((Z[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
            new_labels = np.argmin(d2 + pen[None, :], axis=1)
            if np.array_equal(new_labels, labels):
                break
            labels = new_labels
            for c in range(k):
                if np.any(labels == c):
                    centroids[c] = Z[labels == c].mean(axis=0)
        return labels

    # ------------------------------------------------------------------ selection
    def select(self, backend, X, y, sensitive, unlabeled_idx, queried_idx, batch_size,
               target_distribution):
        n = len(unlabeled_idx)
        need = min(batch_size, n)
        if n == 0:
            return np.array([], dtype=int)

        Z = np.asarray(backend.embed(X[unlabeled_idx]), dtype=float)
        Z = np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)
        ent = entropy_from_proba(backend.predict_proba(X[unlabeled_idx]))
        sens = sensitive[unlabeled_idx]
        pi0 = float(distribution(sensitive[unlabeled_idx], 2)[0])

        # entropy enters the clustering space as one extra (standardised) dimension,
        # matching the reference implementation's [features, entropy] input.
        e = ent.astype(float)
        e = (e - e.mean()) / (e.std() + EPS)
        Zc = np.column_stack([Z, e])

        labels = self._fair_clusters(Zc, sens, need, pi0)

        fallback_order = unlabeled_idx[np.argsort(-ent)]
        selected: list[int] = []
        for c in np.unique(labels):
            members = np.where(labels == c)[0]
            if len(members) < 2:
                continue  # D4
            rep = np.array([np.linalg.norm(Z[members] - Z[members[j]], axis=1).mean()
                            for j in range(len(members))])
            rep_n = _minmax(rep)
            h_n = _minmax(ent[members])
            score = self.beta * rep_n + (1.0 - self.beta) * h_n
            best = members[int(np.lexsort((members, -score))[0])]
            selected.append(int(unlabeled_idx[best]))
            if len(selected) >= need:
                break
        # complete the batch by entropy among samples not already chosen
        chosen = set(selected)
        for idx in fallback_order:
            if len(selected) >= need:
                break
            if int(idx) not in chosen:
                selected.append(int(idx))
                chosen.add(int(idx))
        return np.asarray(selected[:need], dtype=int)


def _minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = float(np.min(x)), float(np.max(x))
    if hi - lo <= EPS:
        return np.zeros_like(x, dtype=float)
    return (x - lo) / (hi - lo)
