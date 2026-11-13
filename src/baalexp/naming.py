
from __future__ import annotations

"""Public naming helpers.

The code keeps the historical internal strategy identifiers (``baal``,
``baal_weighted`` ...) so old CSV files and scripts remain readable.  The
manuscript-facing names are FAIR-AL names.
"""

INTERNAL_TO_PUBLIC_STRATEGY = {
    "random": "Random",
    "uncertainty": "Uncertainty",
    "reweighting": "Reweighting",
    "quota": "Quota sampling",
    "stratified_uncertainty": "Stratified uncertainty",
    "heuristic": "Minority heuristic",
    "coreset": "CoreSet",
    "badge": "BADGE",
    "baal": "FAIR-AL-Pareto",
    "fairal": "FAIR-AL-Pareto",
    "baal_weighted": "FAIR-AL-Weighted",
    "fairal_weighted": "FAIR-AL-Weighted",
    "baal_hard": "FAIR-AL-Constrained",
    "fairal_constrained": "FAIR-AL-Constrained",
    "baal_random_pareto": "FAIR-AL-RandomPareto",
    "baal_random_pareto": "FAIR-AL-RandomPareto",
    # Our implementation of the published FAL-CUR algorithm; the "(reimpl.)" suffix marks
    # that it is not the authors' released script (see fal_cur.py and 02_FAL_CUR_FIDELITY_AUDIT.md).
    "fal_cur": "FAL-CUR (reimpl.)",
    # Superseded 2026-11-03 (fixed-penalty clustering, not FairKM). Kept so the earlier
    # revision-round results stay reproducible; never used in the manuscript.
    "fal_cur_v1": "FAL-CUR v1 (superseded)",
    "fal_cur_rawscale": "FAL-CUR (reimpl., raw scale)",
    "fal_cur_entclust": "FAL-CUR (reimpl., entropy in clustering)",
    "fal_cur_l05": "FAL-CUR (reimpl., lambda=0.5)",
    "fal_cur_l2": "FAL-CUR (reimpl., lambda=2.0)",
}

PUBLIC_ORDER = [
    "Random",
    "Uncertainty",
    "CoreSet",
    "BADGE",
    "FAIR-AL-Pareto",
    "FAIR-AL-Weighted",
    "FAIR-AL-Constrained",
    "FAIR-AL-RandomPareto",
    "Quota sampling",
    "Stratified uncertainty",
    "Minority heuristic",
    "Reweighting",
]

STRATEGY_ALIASES = {
    "quota_sampling": "quota",
    "stratified": "stratified_uncertainty",
    "stratified_sampling": "stratified_uncertainty",
    "stratified_uncertainty_sampling": "stratified_uncertainty",
    "fairal": "baal",
    "fairal_pareto": "baal",
    "fairal_weighted": "baal_weighted",
    "fairal_constrained": "baal_hard",
    "fairal_hard": "baal_hard",
    "fairal_random_pareto": "baal_random_pareto",
}


def canonical_strategy_name(name: str) -> str:
    key = name.strip().lower().replace("-", "_")
    return STRATEGY_ALIASES.get(key, key)


def public_strategy_name(name: str) -> str:
    return INTERNAL_TO_PUBLIC_STRATEGY.get(name, INTERNAL_TO_PUBLIC_STRATEGY.get(canonical_strategy_name(name), name))

