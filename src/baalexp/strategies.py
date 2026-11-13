
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np
from sklearn.metrics import pairwise_distances

from .metrics import EPS, distribution, q_bias
from .naming import canonical_strategy_name
from .fal_cur import FalCurStrategy, make_fal_cur
from .fal_cur_legacy import FalCurV1Strategy


def entropy_from_proba(proba: np.ndarray) -> np.ndarray:
    """Predictive entropy in bits."""
    proba = np.clip(np.asarray(proba, dtype=float), EPS, 1.0)
    return -np.sum(proba * np.log2(proba), axis=1)


class Strategy(Protocol):
    """Protocol for active-learning acquisition strategies."""

    name: str

    def select(
        self,
        backend,
        X: np.ndarray,
        y: np.ndarray,
        sensitive: np.ndarray,
        unlabeled_idx: np.ndarray,
        queried_idx: np.ndarray,
        batch_size: int,
        target_distribution: np.ndarray,
    ) -> np.ndarray: ...


def _fill_to_batch(selected: np.ndarray, fallback_order: np.ndarray, batch_size: int) -> np.ndarray:
    """Return unique indices from selected, then fill from fallback_order."""
    selected = np.asarray(selected, dtype=int).ravel()
    fallback_order = np.asarray(fallback_order, dtype=int).ravel()
    out: list[int] = []
    seen: set[int] = set()
    for idx in np.concatenate([selected, fallback_order]):
        ii = int(idx)
        if ii in seen:
            continue
        out.append(ii)
        seen.add(ii)
        if len(out) >= batch_size:
            break
    return np.asarray(out, dtype=int)


@dataclass
class RandomStrategy:
    seed: int = 0
    name: str = "random"

    def select(self, backend, X, y, sensitive, unlabeled_idx, queried_idx, batch_size, target_distribution):
        rng = np.random.default_rng(self.seed + len(queried_idx))
        return rng.choice(unlabeled_idx, size=min(batch_size, len(unlabeled_idx)), replace=False)


@dataclass
class UncertaintyStrategy:
    name: str = "uncertainty"

    def select(self, backend, X, y, sensitive, unlabeled_idx, queried_idx, batch_size, target_distribution):
        prob = backend.predict_proba(X[unlabeled_idx])
        ent = entropy_from_proba(prob)
        return unlabeled_idx[np.argsort(-ent)[: min(batch_size, len(unlabeled_idx))]]


@dataclass
class ReweightingStrategy:
    name: str = "reweighting"

    def select(self, backend, X, y, sensitive, unlabeled_idx, queried_idx, batch_size, target_distribution):
        prob = backend.predict_proba(X[unlabeled_idx])
        ent = entropy_from_proba(prob)
        pool_sens = sensitive[unlabeled_idx]
        pool_dist = distribution(pool_sens, len(target_distribution))
        weights = np.asarray([1.0 / (pool_dist[int(a)] + EPS) for a in pool_sens], dtype=float)
        score = ent * weights
        return unlabeled_idx[np.argsort(-score)[: min(batch_size, len(unlabeled_idx))]]


@dataclass
class MinorityHeuristicStrategy:
    """Alternating minority-focused uncertainty heuristic.

    This is intentionally a simple baseline to demonstrate possible
    over-correction in acquisition.
    """

    seed: int = 0
    name: str = "heuristic"

    def select(self, backend, X, y, sensitive, unlabeled_idx, queried_idx, batch_size, target_distribution):
        t = len(queried_idx) // max(batch_size, 1)
        prob_all = backend.predict_proba(X[unlabeled_idx])
        ent_all = entropy_from_proba(prob_all)
        fallback = unlabeled_idx[np.argsort(-ent_all)]
        if t % 2 == 0:
            minority = int(np.argmin(target_distribution))
            candidates = unlabeled_idx[sensitive[unlabeled_idx] == minority]
            if len(candidates) > 0:
                prob = backend.predict_proba(X[candidates])
                ent = entropy_from_proba(prob)
                selected = candidates[np.argsort(-ent)[: min(batch_size, len(candidates))]]
                return _fill_to_batch(selected, fallback, min(batch_size, len(unlabeled_idx)))
        return fallback[: min(batch_size, len(unlabeled_idx))]




@dataclass
class QuotaSamplingStrategy:
    """Fixed-quota demographic baseline.

    This baseline greedily selects candidates that most reduce the cumulative
    TV distance between the queried protected-group distribution and the target
    distribution.  It deliberately ignores model informativeness, so it provides
    a strong but simple comparison for distinguishing proportion control from
    FAIR-AL's joint informativeness--drift objective.
    """

    seed: int = 0
    name: str = "quota"

    def select(self, backend, X, y, sensitive, unlabeled_idx, queried_idx, batch_size, target_distribution):
        rng = np.random.default_rng(self.seed + len(queried_idx))
        need = min(batch_size, len(unlabeled_idx))
        remaining = list(map(int, rng.permutation(unlabeled_idx)))
        selected: list[int] = []
        current = np.asarray(queried_idx, dtype=int).copy()
        for _ in range(need):
            if not remaining:
                break
            scores = np.asarray(
                [q_bias(sensitive[np.concatenate([current, [idx]])], target_distribution) for idx in remaining],
                dtype=float,
            )
            best_pos = int(np.argmin(scores))
            chosen = remaining.pop(best_pos)
            selected.append(chosen)
            current = np.concatenate([current, [chosen]])
        return np.asarray(selected, dtype=int)


@dataclass
class StratifiedUncertaintyStrategy:
    """Quota-aware uncertainty baseline.

    The group of each next query is chosen by the same cumulative-distribution
    correction rule as QuotaSamplingStrategy; within the chosen group, the most
    uncertain candidate is selected.  This baseline is useful for separating
    fixed stratification from FAIR-AL's candidate-level multi-objective scoring.
    """

    seed: int = 0
    name: str = "stratified_uncertainty"

    def select(self, backend, X, y, sensitive, unlabeled_idx, queried_idx, batch_size, target_distribution):
        rng = np.random.default_rng(self.seed + len(queried_idx))
        need = min(batch_size, len(unlabeled_idx))
        prob = backend.predict_proba(X[unlabeled_idx])
        ent = entropy_from_proba(prob)
        ent_by_idx = {int(idx): float(score) for idx, score in zip(unlabeled_idx, ent)}
        remaining = list(map(int, rng.permutation(unlabeled_idx)))
        selected: list[int] = []
        current = np.asarray(queried_idx, dtype=int).copy()
        for _ in range(need):
            if not remaining:
                break
            scores = np.asarray(
                [q_bias(sensitive[np.concatenate([current, [idx]])], target_distribution) for idx in remaining],
                dtype=float,
            )
            min_score = float(np.min(scores))
            best_positions = np.where(np.isclose(scores, min_score))[0]
            # Among equally quota-improving candidates, use model uncertainty.
            best_pos = max((int(pos) for pos in best_positions), key=lambda pos: ent_by_idx[remaining[pos]])
            chosen = remaining.pop(best_pos)
            selected.append(chosen)
            current = np.concatenate([current, [chosen]])
        return np.asarray(selected, dtype=int)


@dataclass
class CoreSetStrategy:
    name: str = "coreset"

    def select(self, backend, X, y, sensitive, unlabeled_idx, queried_idx, batch_size, target_distribution):
        emb_unl = backend.embed(X[unlabeled_idx])
        if len(queried_idx) == 0:
            center = emb_unl.mean(axis=0, keepdims=True)
            min_d = pairwise_distances(emb_unl, center).ravel()
        else:
            emb_lab = backend.embed(X[queried_idx])
            min_d = pairwise_distances(emb_unl, emb_lab).min(axis=1)
        selected_local: list[int] = []
        available = np.arange(len(unlabeled_idx))
        for _ in range(min(batch_size, len(unlabeled_idx))):
            j = int(np.argmax(min_d[available]))
            local = int(available[j])
            selected_local.append(local)
            available = np.delete(available, j)
            if len(available) == 0:
                break
            new_d = pairwise_distances(emb_unl[available], emb_unl[[local]]).ravel()
            min_d[available] = np.minimum(min_d[available], new_d)
        return unlabeled_idx[np.asarray(selected_local, dtype=int)]


@dataclass
class BADGEStrategy:
    name: str = "badge"
    seed: int = 0

    def select(self, backend, X, y, sensitive, unlabeled_idx, queried_idx, batch_size, target_distribution):
        prob = backend.predict_proba(X[unlabeled_idx])
        emb = backend.embed(X[unlabeled_idx])
        pred = np.argmax(prob, axis=1)
        grad_scale = prob[:, 1] - (pred == 1).astype(float)
        grad_emb = np.nan_to_num(emb * grad_scale[:, None], nan=0.0, posinf=0.0, neginf=0.0)
        return kmeanspp_select(unlabeled_idx, grad_emb, min(batch_size, len(unlabeled_idx)), seed=self.seed + len(queried_idx))


@dataclass
class FairALParetoStrategy:
    k_candidates: int = 200
    seed: int = 0
    intra_batch: bool = False
    name: str = "baal"  # historical CSV code for FAIR-AL-Pareto

    def select(self, backend, X, y, sensitive, unlabeled_idx, queried_idx, batch_size, target_distribution):
        if self.intra_batch:
            return _intra_batch_fairal(
                backend, X, sensitive, unlabeled_idx, queried_idx, batch_size,
                target_distribution, self.k_candidates, mode="knee", seed=self.seed,
            )
        return _fairal_select(
            backend,
            X,
            sensitive,
            unlabeled_idx,
            queried_idx,
            batch_size,
            target_distribution,
            self.k_candidates,
            mode="knee",
            seed=self.seed,
        )


@dataclass
class FairALRandomParetoStrategy:
    k_candidates: int = 200
    seed: int = 0
    intra_batch: bool = False
    name: str = "baal_random_pareto"

    def select(self, backend, X, y, sensitive, unlabeled_idx, queried_idx, batch_size, target_distribution):
        if self.intra_batch:
            return _intra_batch_fairal(
                backend, X, sensitive, unlabeled_idx, queried_idx, batch_size,
                target_distribution, self.k_candidates, mode="random_pareto", seed=self.seed,
            )
        return _fairal_select(
            backend,
            X,
            sensitive,
            unlabeled_idx,
            queried_idx,
            batch_size,
            target_distribution,
            self.k_candidates,
            mode="random_pareto",
            seed=self.seed,
        )


@dataclass
class FairALWeightedStrategy:
    k_candidates: int = 200
    lambda_weight: float = 0.5
    intra_batch: bool = False
    name: str = "baal_weighted"

    def select(self, backend, X, y, sensitive, unlabeled_idx, queried_idx, batch_size, target_distribution):
        if self.intra_batch:
            return _intra_batch_fairal(
                backend, X, sensitive, unlabeled_idx, queried_idx, batch_size,
                target_distribution, self.k_candidates, mode="weighted", seed=0,
                lambda_weight=self.lambda_weight,
            )
        prob = backend.predict_proba(X[unlabeled_idx])
        ent = entropy_from_proba(prob)
        cand_n = min(self.k_candidates, len(unlabeled_idx))
        cand_local = np.argsort(-ent)[:cand_n]
        cand_idx = unlabeled_idx[cand_local]
        f1 = -ent[cand_local]
        f2 = candidate_qbias_values(sensitive, queried_idx, cand_idx, target_distribution)
        score = self.lambda_weight * normalize_minmax(f1) + (1.0 - self.lambda_weight) * normalize_minmax(f2)
        selected = cand_idx[np.argsort(score)[: min(batch_size, len(cand_idx))]]
        fallback = unlabeled_idx[np.argsort(-ent)]
        return _fill_to_batch(selected, fallback, min(batch_size, len(unlabeled_idx)))


@dataclass
class FairALConstrainedStrategy:
    k_candidates: int = 200
    epsilon: float = 0.03  # TV-distance threshold for feasible candidates
    intra_batch: bool = False
    name: str = "baal_hard"
    # Filled on every select() call so the driver can record constraint fallbacks.
    last_call_stats: dict | None = None

    def select(self, backend, X, y, sensitive, unlabeled_idx, queried_idx, batch_size, target_distribution):
        if self.intra_batch:
            return _intra_batch_fairal(
                backend, X, sensitive, unlabeled_idx, queried_idx, batch_size,
                target_distribution, self.k_candidates, mode="constrained", seed=0,
                epsilon=self.epsilon,
            )
        prob = backend.predict_proba(X[unlabeled_idx])
        ent = entropy_from_proba(prob)
        cand_n = min(self.k_candidates, len(unlabeled_idx))
        cand_local = np.argsort(-ent)[:cand_n]
        cand_idx = unlabeled_idx[cand_local]
        f2 = candidate_qbias_values(sensitive, queried_idx, cand_idx, target_distribution)
        feasible = np.where(f2 <= self.epsilon)[0]
        dropped = 0
        feasible_fraction = float(len(feasible)) / max(len(cand_idx), 1)
        if len(feasible) == 0:
            # Historical behaviour: the tolerance is dropped for this batch and the
            # most informative candidates are taken instead. Recorded for reporting.
            feasible = np.arange(len(cand_idx))
            dropped = 1
        self.last_call_stats = {
            "constraint_dropped": dropped,
            "feasible_fraction": feasible_fraction,
        }
        feasible_order = feasible[np.argsort(-ent[cand_local][feasible])]
        selected = cand_idx[feasible_order[: min(batch_size, len(feasible_order))]]
        fallback = unlabeled_idx[np.argsort(-ent)]
        return _fill_to_batch(selected, fallback, min(batch_size, len(unlabeled_idx)))


# Backward-compatible class aliases.
BAALStrategy = FairALParetoStrategy
BAALRandomParetoStrategy = FairALRandomParetoStrategy
BAALWeightedStrategy = FairALWeightedStrategy
BAALHardConstraintStrategy = FairALConstrainedStrategy
HeuristicStrategy = MinorityHeuristicStrategy


def make_strategy(
    name: str,
    seed: int,
    k_candidates: int,
    lambda_weight: float,
    epsilon: float,
    intra_batch: bool = False,
    cluster_drop_cols: Sequence[int] | None = None,
) -> Strategy:
    code = canonical_strategy_name(name)
    if code == "random":
        return RandomStrategy(seed=seed)
    if code == "uncertainty":
        return UncertaintyStrategy()
    if code == "reweighting":
        return ReweightingStrategy()
    if code == "quota":
        return QuotaSamplingStrategy(seed=seed)
    if code == "stratified_uncertainty":
        return StratifiedUncertaintyStrategy(seed=seed)
    if code == "heuristic":
        return MinorityHeuristicStrategy(seed=seed)
    if code == "coreset":
        return CoreSetStrategy()
    if code == "badge":
        return BADGEStrategy(seed=seed)
    if code == "baal":
        return FairALParetoStrategy(k_candidates=k_candidates, seed=seed, intra_batch=intra_batch)
    if code == "baal_random_pareto":
        return FairALRandomParetoStrategy(k_candidates=k_candidates, seed=seed, intra_batch=intra_batch)
    if code == "baal_weighted":
        return FairALWeightedStrategy(k_candidates=k_candidates, lambda_weight=lambda_weight, intra_batch=intra_batch)
    if code == "baal_hard":
        return FairALConstrainedStrategy(k_candidates=k_candidates, epsilon=epsilon, intra_batch=intra_batch)
    if code == "fal_cur":
        return FalCurStrategy(seed=seed, cluster_drop_cols=cluster_drop_cols)
    if code == "fal_cur_v1":
        return FalCurV1Strategy(seed=seed)
    if code in {"fal_cur_rawscale", "fal_cur_entclust", "fal_cur_l05", "fal_cur_l2"}:
        return make_fal_cur(code, seed)
    raise ValueError(f"Unknown strategy: {name}")


def candidate_qbias_values(sensitive: np.ndarray, queried_idx: np.ndarray, cand_idx: np.ndarray, target_distribution: np.ndarray) -> np.ndarray:
    """Candidate-level TV Q-Bias values after adding one candidate."""
    current = sensitive[queried_idx] if len(queried_idx) else np.array([], dtype=int)
    return np.asarray([q_bias(np.concatenate([current, [sensitive[idx]]]), target_distribution) for idx in cand_idx], dtype=float)


def non_dominated_mask(f1: np.ndarray, f2: np.ndarray) -> np.ndarray:
    n = len(f1)
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        dominated = ((f1 <= f1[i]) & (f2 <= f2[i]) & ((f1 < f1[i]) | (f2 < f2[i]))).any()
        if dominated:
            mask[i] = False
    return mask


def normalize_minmax(x: np.ndarray) -> np.ndarray:
    x = np.nan_to_num(np.asarray(x, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    return (x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x) + EPS)


def _intra_batch_fairal(
    backend,
    X,
    sensitive,
    unlabeled_idx,
    queried_idx,
    batch_size,
    target_distribution,
    k_candidates,
    mode,
    seed,
    lambda_weight: float = 0.5,
    epsilon: float = 0.03,
) -> np.ndarray:
    """Intra-batch (sequential) FAIR-AL ablation.

    The historical implementation scores every candidate of a batch against the same
    pre-batch cumulative queried distribution. This variant instead updates the
    cumulative distribution after each pick inside the batch, matching the sequential
    rule used by the quota / stratified-uncertainty baselines. Modes: "knee",
    "random_pareto", "weighted", "constrained". Historical behaviour is untouched:
    this function is only reached when ``intra_batch=True``.
    """
    rng = np.random.default_rng(seed + len(queried_idx))
    need = min(batch_size, len(unlabeled_idx))
    prob = backend.predict_proba(X[unlabeled_idx])
    ent = entropy_from_proba(prob)
    fallback = unlabeled_idx[np.argsort(-ent)]
    cand_n = min(k_candidates, len(unlabeled_idx))
    cand_local = np.argsort(-ent)[:cand_n]
    cand_idx = unlabeled_idx[cand_local]
    cand_ent = ent[cand_local]

    current = sensitive[queried_idx] if len(queried_idx) else np.array([], dtype=int)
    available = list(range(len(cand_idx)))
    selected: list[int] = []
    constraint_dropped = 0

    for _ in range(min(need, len(cand_idx))):
        remaining = np.asarray(available, dtype=int)
        if len(remaining) == 0:
            break
        f2 = np.asarray(
            [q_bias(np.concatenate([current, [sensitive[cand_idx[p]]]]), target_distribution) for p in remaining],
            dtype=float,
        )
        f1 = -cand_ent[remaining]

        if mode == "constrained":
            feasible_local = np.where(f2 <= epsilon)[0]
            if len(feasible_local) == 0:
                constraint_dropped += 1
                feasible_local = np.arange(len(remaining))
            order = feasible_local[np.lexsort((remaining[feasible_local], f1[feasible_local]))]
            pick_local = int(remaining[order[0]])
        elif mode == "weighted":
            score = lambda_weight * normalize_minmax(f1) + (1.0 - lambda_weight) * normalize_minmax(f2)
            pick_local = int(remaining[int(np.argmin(score))])
        elif mode == "random_pareto":
            nd = non_dominated_mask(f1, f2)
            pool = remaining[nd] if nd.any() else remaining
            pick_local = int(pool[0]) if len(pool) == 1 else int(rng.choice(pool))
        elif mode == "knee":
            nd = non_dominated_mask(f1, f2)
            pool = remaining[nd] if nd.any() else remaining
            f1n = normalize_minmax(f1[np.isin(remaining, pool)])
            f2n = normalize_minmax(f2[np.isin(remaining, pool)])
            dist = np.sqrt(f1n ** 2 + f2n ** 2)
            order = np.lexsort((pool, f2[np.isin(remaining, pool)], f1[np.isin(remaining, pool)], dist))
            pick_local = int(pool[order[0]])
        else:  # pragma: no cover
            raise ValueError(f"Unknown intra-batch mode: {mode}")

        chosen = int(cand_idx[pick_local])
        selected.append(chosen)
        current = np.concatenate([current, [sensitive[chosen]]])
        available.remove(pick_local)

    return _fill_to_batch(np.asarray(selected, dtype=int), fallback, need)


def _fairal_select(backend, X, sensitive, unlabeled_idx, queried_idx, batch_size, target_distribution, k_candidates, mode, seed) -> np.ndarray:
    rng = np.random.default_rng(seed + len(queried_idx))
    need = min(batch_size, len(unlabeled_idx))
    prob = backend.predict_proba(X[unlabeled_idx])
    ent = entropy_from_proba(prob)
    fallback = unlabeled_idx[np.argsort(-ent)]
    cand_n = min(k_candidates, len(unlabeled_idx))
    cand_local = np.argsort(-ent)[:cand_n]
    cand_idx = unlabeled_idx[cand_local]
    f1 = -ent[cand_local]
    f2 = candidate_qbias_values(sensitive, queried_idx, cand_idx, target_distribution)
    nd = non_dominated_mask(f1, f2)
    pareto_idx = cand_idx[nd]
    pareto_f1 = f1[nd]
    pareto_f2 = f2[nd]
    if len(pareto_idx) == 0:
        return fallback[:need]

    if mode == "random_pareto":
        n_take = min(need, len(pareto_idx))
        selected = rng.choice(pareto_idx, size=n_take, replace=False)
        # Fill with randomized candidates rather than uncertainty order so the ablation
        # does not collapse to FAIR-AL-Pareto when the Pareto set is small.
        random_candidate_fill = rng.permutation(cand_idx)
        random_pool_fill = rng.permutation(unlabeled_idx)
        return _fill_to_batch(selected, np.concatenate([random_candidate_fill, random_pool_fill]), need)

    f1n = normalize_minmax(pareto_f1)
    f2n = normalize_minmax(pareto_f2)
    dist = np.sqrt(f1n**2 + f2n**2)
    # Deterministic tie-breaking: distance, then f1, then f2, then sample index.
    order = np.lexsort((pareto_idx, pareto_f2, pareto_f1, dist))
    selected = pareto_idx[order[: min(need, len(order))]]
    return _fill_to_batch(selected, fallback, need)


def kmeanspp_select(indices: np.ndarray, X: np.ndarray, k: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    indices = np.asarray(indices)
    X = np.nan_to_num(np.asarray(X, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    n = len(indices)
    if n == 0 or k <= 0:
        return np.array([], dtype=indices.dtype)
    if k >= n:
        return indices
    centers: list[int] = [int(rng.integers(0, n))]
    dist_sq = pairwise_distances(X, X[[centers[0]]], metric="euclidean").ravel() ** 2
    dist_sq = np.nan_to_num(dist_sq, nan=0.0, posinf=0.0, neginf=0.0)
    for _ in range(1, k):
        total = float(dist_sq.sum())
        if (not np.isfinite(total)) or total <= EPS:
            remaining = np.setdiff1d(np.arange(n), np.asarray(centers), assume_unique=False)
            if len(remaining) == 0:
                break
            nxt = int(rng.choice(remaining))
        else:
            probs = np.nan_to_num(dist_sq / total, nan=0.0, posinf=0.0, neginf=0.0)
            psum = float(probs.sum())
            if (not np.isfinite(psum)) or psum <= EPS:
                remaining = np.setdiff1d(np.arange(n), np.asarray(centers), assume_unique=False)
                nxt = int(rng.choice(remaining))
            else:
                probs = probs / psum
                probs[-1] += 1.0 - float(probs.sum())
                probs = np.clip(probs, 0.0, 1.0)
                probs = probs / probs.sum()
                nxt = int(rng.choice(np.arange(n), p=probs))
        centers.append(nxt)
        new_dist_sq = pairwise_distances(X, X[[nxt]], metric="euclidean").ravel() ** 2
        new_dist_sq = np.nan_to_num(new_dist_sq, nan=0.0, posinf=0.0, neginf=0.0)
        dist_sq = np.minimum(dist_sq, new_dist_sq)
    return indices[np.asarray(centers[:k], dtype=int)]

