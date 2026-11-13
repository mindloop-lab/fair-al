
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, roc_auc_score

EPS = 1e-12


@dataclass
class MetricBundle:
    accuracy: float
    balanced_accuracy: float
    auc: float
    sensitivity: float
    specificity: float
    precision: float
    f1: float
    fpr: float
    fnr: float


def safe_auc(y_true: np.ndarray, prob_pos: np.ndarray) -> float:
    try:
        if len(np.unique(y_true)) < 2:
            return float("nan")
        return float(roc_auc_score(y_true, prob_pos))
    except Exception:
        return float("nan")


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, prob_pos: np.ndarray | None = None) -> MetricBundle:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    if prob_pos is None:
        prob_pos = y_pred.astype(float)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn + EPS)
    spec = tn / (tn + fp + EPS)
    prec = tp / (tp + fp + EPS)
    f1 = 2 * prec * sens / (prec + sens + EPS)
    fpr = fp / (fp + tn + EPS)
    fnr = fn / (fn + tp + EPS)
    return MetricBundle(
        accuracy=float(accuracy_score(y_true, y_pred)),
        balanced_accuracy=float(balanced_accuracy_score(y_true, y_pred)),
        auc=safe_auc(y_true, prob_pos),
        sensitivity=float(sens),
        specificity=float(spec),
        precision=float(prec),
        f1=float(f1),
        fpr=float(fpr),
        fnr=float(fnr),
    )


def subgroup_metrics(y_true: np.ndarray, y_pred: np.ndarray, prob_pos: np.ndarray, sensitive: np.ndarray) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for group_value, name in [(0, "group0"), (1, "group1")]:
        mask = sensitive == group_value
        if mask.sum() == 0:
            continue
        m = binary_metrics(y_true[mask], y_pred[mask], prob_pos[mask])
        out.update({f"{name}_{k}": v for k, v in m.__dict__.items()})
        tn, fp, fn, tp = confusion_matrix(y_true[mask], y_pred[mask], labels=[0, 1]).ravel()
        out.update(
            {
                f"{name}_tn": int(tn),
                f"{name}_fp": int(fp),
                f"{name}_fn": int(fn),
                f"{name}_tp": int(tp),
                f"{name}_n": int(mask.sum()),
            }
        )
    if "group0_sensitivity" in out and "group1_sensitivity" in out:
        out["eod_abs"] = abs(out["group0_sensitivity"] - out["group1_sensitivity"])
        out["sensitivity_gap_g0_minus_g1"] = out["group0_sensitivity"] - out["group1_sensitivity"]
        out["fpr_gap_abs"] = abs(out["group0_fpr"] - out["group1_fpr"])
        out["fnr_gap_abs"] = abs(out["group0_fnr"] - out["group1_fnr"])
    return out


def distribution(values: np.ndarray, n_groups: int = 2) -> np.ndarray:
    values = np.asarray(values).astype(int)
    if len(values) == 0:
        return np.ones(n_groups) / n_groups
    counts = np.array([(values == g).sum() for g in range(n_groups)], dtype=float)
    if counts.sum() == 0:
        return np.ones(n_groups) / n_groups
    return counts / counts.sum()


def q_bias_l1(query_sensitive: np.ndarray, target_distribution: np.ndarray) -> float:
    """L1 distance between queried and target protected-group distributions.

    This helper is kept for diagnostics and for comparing against legacy v7
    results, where the manuscript-facing ``q_bias`` column accidentally used
    the unscaled L1 distance.  For binary protected groups this value is in
    ``[0, 2]``.
    """
    query_sensitive = np.asarray(query_sensitive).astype(int)
    target_distribution = np.asarray(target_distribution, dtype=float)
    if len(query_sensitive) == 0:
        return 0.0
    if target_distribution.ndim != 1 or len(target_distribution) == 0:
        raise ValueError("target_distribution must be a non-empty 1D array")
    target_distribution = target_distribution / (target_distribution.sum() + EPS)
    values = np.arange(len(target_distribution))
    qdist = np.array([(query_sensitive == v).mean() for v in values], dtype=float)
    return float(np.abs(qdist - target_distribution).sum())


def q_bias(query_sensitive: np.ndarray, target_distribution: np.ndarray) -> float:
    """Total-variation query bias.

    Q-Bias is defined as the total variation distance between the protected-group
    distribution of the cumulative queried set and the target protected-group
    distribution:

        TV(q, pi) = 0.5 * ||q - pi||_1.

    The TV definition is used by the updated manuscript and all v8 result files.
    For a categorical protected attribute it is always in ``[0, 1]``.
    """
    return 0.5 * q_bias_l1(query_sensitive, target_distribution)


def evaluate_predictions(
    y_true: np.ndarray,
    prob: np.ndarray,
    sensitive: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, Any]:
    prob_pos = prob[:, 1] if prob.ndim == 2 else prob
    y_pred = (prob_pos >= threshold).astype(int)
    result = binary_metrics(y_true, y_pred, prob_pos).__dict__.copy()
    result.update(subgroup_metrics(y_true, y_pred, prob_pos, sensitive))
    return result


def clinical_impact_per_1000(row_a: dict[str, Any], row_b: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for group in ["group0", "group1"]:
        n_key = f"{group}_n"
        if n_key not in row_a or n_key not in row_b:
            continue
        n = max(float(row_a[n_key]), 1.0)
        pos = float(row_a.get(f"{group}_tp", 0) + row_a.get(f"{group}_fn", 0))
        neg = float(row_a.get(f"{group}_tn", 0) + row_a.get(f"{group}_fp", 0))
        pos_rate = pos / n
        neg_rate = neg / n
        out[f"{group}_delta_fn_per_1000"] = 1000.0 * pos_rate * (row_a[f"{group}_fnr"] - row_b[f"{group}_fnr"])
        out[f"{group}_delta_fp_per_1000"] = 1000.0 * neg_rate * (row_a[f"{group}_fpr"] - row_b[f"{group}_fpr"])
    return out

