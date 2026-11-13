"""FAL-CUR comparison baseline — an adapted reimplementation for the shared FAIR-AL protocol.

Reference
---------
R. M. Fajri, A. Saxena, Y. Pei, M. Pechenizkiy, "FAL-CUR: Fair Active Learning using
Uncertainty and Representativeness on Fair Clustering", Expert Systems with Applications
242:122842, 2024 (doi:10.1016/j.eswa.2023.122842); preprinted as arXiv:2209.12756v2.
The fairness-aware clustering step is taken from FairKM (Abraham, Deepak, Sundaram,
"Fairness in clustering with multiple sensitive attributes", arXiv:1910.05113), which is
the reference FAL-CUR cites for its clustering stage.

What is implemented (verified against the paper text, see
02_FAL_CUR_FIDELITY_AUDIT.md in the revision package):

  Stage 1  Fair clustering, FairKM objective
           O = sum_C sum_{X in C} dist_N(X, C) + lambda * fairness_S(C, X),
           with the cluster-level deviation of published Eq. (3):
               dev(C) = [ (|C|/|X|)^2 * sum_{s in Values(S)} (Fr_C(s) - Fr_X(s))^2
                          / |Values(S)| ].
           Assignment is marginal, as in FairKM Eq. (8)/(9): for each sample the change
           delta(O) of moving it from its current cluster to every other cluster is
           evaluated with the receiving/leaving cluster shares recomputed, and the best
           move is applied in round-robin fashion until convergence.

  Stage 2  Cluster ranking and sample selection
           Clusters are ranked from most fair to most unfair by dev(C) (published Eq. (3)).
           Inside each cluster the score
               score(x) = beta * Rep(x) + (1 - beta) * H(x),        beta = 0.6
           is computed, where Rep(x) is the sum of Euclidean distances from x to the other
           members of its cluster (the Rep expression preceding published Eq. (6)) and H(x)
           is the predictive entropy in bits (published Eq. (5)). Samples are ranked by
           descending score; the batch is filled round-robin over the fairness-ranked
           clusters, taking the j-th ranked sample of each cluster (paper Algorithm 1).

Deviations from the published description (all reported with the results; D1-D6 in the
audit file):
  D1  lambda is not reported by FAL-CUR. FairKM's own guidance (Sec. 5.4) sets
      lambda = (|X|/k)^2 so that the fairness and KMeans terms carry comparable weight;
      we use that heuristic with a multiplier fixed a priori at 1.0 and report a
      sensitivity sweep (x0.25, x4). The assignment itself follows FairKM's marginal
      delta(O) rule.
  D2  k = batch size (10) with re-clustering every round, so the shared active-learning
      protocol (initial 40 / budget 120 / batch 10) is used unchanged. The paper treats
      k as an independent hyperparameter.
  D3  score scaling: the paper combines a sum of distances with predictive entropy
      and prints acquisition-score Eq. (6) with the sign "(beta - 1)", which
      contradicts its own statement that a small beta favours uncertainty. We use
      beta * Rep~ + (1 - beta) * H~ with Rep~ and H~ min-max normalised WITHIN each
      cluster, i.e. the sign convention of the paper's text and of the authors'
      released script. A variant without normalisation is available for sensitivity.
  D4  published Eq. (1) defines the distance on nonsensitive coordinates. This
      implementation uses standardized model features for clustering, including the
      group-encoding columns when present. An alternate run adds entropy as a clustering
      coordinate. Group membership separately enters the fairness penalty.
  D5  clusters with a single member are kept: they simply contribute one sample, as
      Algorithm 1 implies. If every cluster is exhausted before the batch is full, the
      remaining slots are filled by highest entropy and the fill count is returned.
  D6  only unlabeled features, predicted probabilities and the sensitive attribute are
      used. No labels, no test information.

The authors' released script (github.com/rickymaulanafajri/FAL-CUR, 192 lines, no
licence, last pushed 2023-12-01) does NOT implement this clustering: its ``fairKMeans``
runs plain KMeans and keeps points whose normalised distance to their centroid lies in
[0.3, 0.6]; it scores ``beta * distance_to_centroid + (1 - beta) * entropy`` and selects
one point per cluster once, with k = 180. It is therefore neither copied nor
redistributed here; this module is a reimplementation of the paper, not the authors' code.
"""
from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans

from typing import Sequence

from .metrics import EPS, distribution

VARIANT_MAIN = "main"
VARIANT_RAW_SCALE = "raw_scale"
VARIANT_CLUSTER_ENTROPY = "cluster_entropy"

FAL_CUR_DEVIATIONS = [
    "D1_fairkm_marginal_assignment_lambda_fixed_1.0",
    "D2_k_equals_batch",
    "D3_within_cluster_norm_and_set_minus_sign",
    "D4_cluster_on_features_only",
    "D5_singletons_kept_entropy_fill_recorded",
    "D6_no_labels_used",
]


def entropy_from_proba(proba: np.ndarray) -> np.ndarray:
    """Predictive entropy in bits (local copy to avoid a circular import)."""
    proba = np.clip(np.asarray(proba, dtype=float), EPS, 1.0)
    return -np.sum(proba * np.log2(proba), axis=1)


def _minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = float(np.min(x)), float(np.max(x))
    if hi - lo <= EPS:
        return np.zeros_like(x, dtype=float)
    return (x - lo) / (hi - lo)


def _cluster_deviation(n_c: np.ndarray, n0_c: np.ndarray, n_total: int, pi0: float) -> np.ndarray:
    """Cluster-level fairness deviation of Eq. (4), weighted by (|C|/|X|)^2.

    Binary sensitive attribute: |Values(S)| = 2 and the two shares are p and 1 - p, so
    the deviation is ((p - pi0)^2 + ((1-p) - (1-pi0))^2) / 2 = (p - pi0)^2.
    Empty clusters contribute 0 by definition (FairKM Eq. 3).
    """
    n_c = np.asarray(n_c, dtype=float)
    n0_c = np.asarray(n0_c, dtype=float)
    share = np.divide(n0_c, n_c, out=np.full_like(n_c, pi0), where=n_c > 0)
    dev = (share - pi0) ** 2
    weight = (n_c / float(n_total)) ** 2
    return np.where(n_c > 0, weight * dev, 0.0)


class FalCurStrategy:
    """Our implementation of the published FAL-CUR algorithm (see module docstring)."""

    name = "fal_cur"

    def __init__(self, beta: float = 0.6, lam_scale: float = 1.0, max_iter: int = 20,
                 seed: int = 0, name: str = "fal_cur", variant: str = VARIANT_MAIN,
                 cluster_drop_cols: Sequence[int] | None = None):
        self.beta = float(beta)
        self.lam_scale = float(lam_scale)
        self.max_iter = int(max_iter)
        self.seed = int(seed)
        self.name = name
        self.variant = variant
        # D4 sensitivity: group-encoding columns neutralised in the distance computations
        # used by FAL-CUR (the fair-cluster assignment and the within-cluster
        # representativeness term, both computed from Z).  A neutralised column is set to
        # its own mean, so it is constant and adds nothing to any Euclidean distance.
        # Predictor input, group penalty and protocol untouched.
        self.cluster_drop_cols = sorted({int(c) for c in (cluster_drop_cols or [])})
        if variant not in {VARIANT_MAIN, VARIANT_RAW_SCALE, VARIANT_CLUSTER_ENTROPY}:
            raise ValueError(f"unknown FAL-CUR variant: {variant}")

    # ------------------------------------------------------------------ clustering
    def _fair_clusters(self, Z: np.ndarray, sens: np.ndarray, k: int, pi0: float) -> np.ndarray:
        """FairKM-style marginal assignment (FairKM Eq. 8/9)."""
        n, d = Z.shape
        k = max(1, min(k, n))
        init = KMeans(n_clusters=k, random_state=self.seed, n_init=1, max_iter=50).fit(Z)
        labels = init.labels_.copy()

        # FairKM Sec. 5.4: lambda = (|X| / k)^2 balances the two terms of the objective.
        lam = self.lam_scale * (n / float(k)) ** 2

        cnt = np.bincount(labels, minlength=k).astype(float)
        cnt0 = np.bincount(labels, weights=(sens == 0).astype(float), minlength=k)
        S = np.zeros((k, d))
        S2 = np.zeros(k)
        np.add.at(S, labels, Z)
        np.add.at(S2, labels, (Z * Z).sum(axis=1))

        def sse_one(sum_vec, sum_sq, counts):
            with np.errstate(invalid="ignore", divide="ignore"):
                mean_term = np.where(counts > 0, (sum_vec * sum_vec).sum(axis=-1) / counts, 0.0)
            return np.where(counts > 0, sum_sq - mean_term, 0.0)

        for _ in range(self.max_iter):
            moved = False
            for i in range(n):
                c0 = labels[i]
                z = Z[i]
                zz = float(z @ z)
                take = 1.0 if sens[i] == 0 else 0.0

                base_dev = _cluster_deviation(cnt, cnt0, n, pi0)
                base_sse = sse_one(S, S2, cnt)

                out_cnt = cnt.copy()
                out_cnt[c0] -= 1.0
                out0 = cnt0.copy()
                out0[c0] -= take
                out_S = S.copy()
                out_S[c0] -= z
                out_S2 = S2.copy()
                out_S2[c0] -= zz

                # candidate cluster c receives sample i; every other cluster keeps its state
                add_cnt = out_cnt + 1.0
                add0 = out0 + take
                add_S = out_S + z[None, :]
                add_S2 = out_S2 + zz
                add_sse = sse_one(add_S, add_S2, add_cnt) - sse_one(out_S, out_S2, out_cnt)
                add_dev = _cluster_deviation(add_cnt, add0, n, pi0) - \
                    _cluster_deviation(out_cnt, out0, n, pi0)

                # the origin cluster c0 loses the sample
                rem_sse = (sse_one(out_S[c0], out_S2[c0], out_cnt[c0]) -
                           sse_one(S[c0], S2[c0], cnt[c0]))
                rem_dev = (_cluster_deviation(out_cnt[c0], out0[c0], n, pi0) -
                           base_dev[c0])

                total = (add_sse + add_dev * lam) + (rem_sse + rem_dev * lam)
                best = int(np.argmin(total))
                if best != c0:
                    labels[i] = best
                    moved = True
                    cnt[best] += 1.0
                    cnt[c0] -= 1.0
                    if sens[i] == 0:
                        cnt0[best] += 1.0
                        cnt0[c0] -= 1.0
                    S[best] += z
                    S[c0] -= z
                    S2[best] += zz
                    S2[c0] -= zz
            if not moved:
                break
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
        sens = np.asarray(sensitive[unlabeled_idx]).astype(int)
        pi0 = float(distribution(sensitive[unlabeled_idx], 2)[0])

        if self.cluster_drop_cols:
            keep = [c for c in self.cluster_drop_cols if 0 <= c < Z.shape[1]]
            if keep:
                Z = Z.copy()
                Z[:, keep] = Z[:, keep].mean(axis=0)
        if self.variant == VARIANT_CLUSTER_ENTROPY:
            e = ent.astype(float)
            e = (e - e.mean()) / (e.std() + EPS)
            Zc = np.column_stack([Z, e])
        else:
            Zc = Z

        labels = self._fair_clusters(Zc, sens, need, pi0)

        # ---- Published Eq. (3): rank clusters from most fair to most unfair
        n_c = np.bincount(labels, minlength=int(labels.max()) + 1).astype(float)
        n0_c = np.bincount(labels, weights=(sens == 0).astype(float),
                           minlength=int(labels.max()) + 1)
        dev = _cluster_deviation(n_c, n0_c, n, pi0)
        present = np.where(n_c > 0)[0]
        order = present[np.argsort(dev[present], kind="stable")]

        # ---- score samples within each cluster (adapted from published Eq. (6))
        ranked: list[np.ndarray] = []
        for c in order:
            members = np.where(labels == c)[0]
            D = np.linalg.norm(Z[members][:, None, :] - Z[members][None, :, :], axis=2)
            rep = D.sum(axis=1)                       # sum of distances to other members
            h = ent[members]
            if self.variant == VARIANT_RAW_SCALE:
                score = self.beta * rep + (1.0 - self.beta) * h
            else:
                score = self.beta * _minmax(rep) + (1.0 - self.beta) * _minmax(h)
            ranked.append(members[np.lexsort((members, -score))])

        # ---- Algorithm 1: round-robin j-th ranked sample over fairness-ranked clusters
        selected: list[int] = []
        j = 0
        max_len = max((len(r) for r in ranked), default=0)
        while len(selected) < need and j < max_len:
            for r in ranked:
                if len(selected) >= need:
                    break
                if j < len(r):
                    selected.append(int(unlabeled_idx[r[j]]))
            j += 1

        fill_count = 0
        if len(selected) < need:
            chosen = set(selected)
            for idx in unlabeled_idx[np.argsort(-ent)]:
                if len(selected) >= need:
                    break
                if int(idx) not in chosen:
                    selected.append(int(idx))
                    chosen.add(int(idx))
                    fill_count += 1

        self.last_fill_count = fill_count
        return np.asarray(selected[:need], dtype=int)


def make_fal_cur(code: str, seed: int) -> FalCurStrategy:
    """Factory for the FAL-CUR strategy codes registered in ``strategies.py``."""
    if code == "fal_cur":
        return FalCurStrategy(seed=seed)
    if code == "fal_cur_rawscale":
        return FalCurStrategy(seed=seed, name="fal_cur_rawscale", variant=VARIANT_RAW_SCALE)
    if code == "fal_cur_entclust":
        return FalCurStrategy(seed=seed, name="fal_cur_entclust",
                              variant=VARIANT_CLUSTER_ENTROPY)
    if code == "fal_cur_l05":
        return FalCurStrategy(seed=seed, lam_scale=0.25, name="fal_cur_l05")
    if code == "fal_cur_l2":
        return FalCurStrategy(seed=seed, lam_scale=4.0, name="fal_cur_l2")
    raise ValueError(f"unknown FAL-CUR strategy code: {code}")
