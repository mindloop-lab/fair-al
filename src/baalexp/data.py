
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass
class ABIDEDataset:
    name: str
    X: np.ndarray
    y: np.ndarray
    sensitive: np.ndarray
    modal_dims: list[int]
    feature_names: list[str] | None = None
    sensitive_name: str = "sex"
    notes: str = ""
    # Indices (within the returned X) of the protected-attribute columns that are
    # also model inputs; recorded for disclosure in the manuscript.
    sensitive_feature_cols: list[int] = field(default_factory=list)
    # Columns removed from X when drop_sensitive_inputs=True (audit ablation).
    dropped_feature_indices: list[int] = field(default_factory=list)
    dropped_feature_names: list[str] = field(default_factory=list)


def _binarize_sensitive(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a).copy()
    vals = np.unique(a[~pd.isna(a)])
    if len(vals) == 2:
        mapping = {vals[0]: 0, vals[1]: 1}
        return np.vectorize(mapping.get)(a).astype(int)
    med = np.nanmedian(a)
    return (a > med).astype(int)


def _normalise_label(y_raw: np.ndarray) -> np.ndarray:
    y = np.asarray(y_raw).astype(int)
    if set(np.unique(y)) == {1, 2}:
        y = y - 1
    elif y.min() != 0:
        y = y - y.min()
    return y.astype(int)


def _drop_columns(X: np.ndarray, idx: list[int]) -> np.ndarray:
    """Return X with the given column indices removed (order preserved)."""
    if not idx:
        return X
    keep = [j for j in range(X.shape[1]) if j not in set(idx)]
    return np.ascontiguousarray(X[:, keep])


def _guess_sensitive_from_dataframe(
    df: pd.DataFrame,
    X: np.ndarray,
    index: Optional[int],
) -> tuple[np.ndarray, str, str]:
    if index is not None:
        return _binarize_sensitive(X[:, index]), f"feature_{index}", "sensitive attribute supplied by --sensitive-index"

    lower = {str(c).lower(): c for c in df.columns}
    # Prefer explicit sex/gender columns. Avoid broad substring matches such as
    # "female_score" unless the original table really contains only that column.
    for key in ["sex", "gender", "SEX", "Gender"]:
        for low, orig in lower.items():
            if key.lower() == low or low.endswith("_" + key.lower()) or key.lower() in low:
                return _binarize_sensitive(df[orig].to_numpy()), str(orig), "sensitive column inferred from column name"

    default_idx = 1 if X.shape[1] > 1 else 0
    return (
        _binarize_sensitive(X[:, default_idx]),
        f"feature_{default_idx}",
        "WARNING: sensitive column was guessed as feature index 1; verify with the data dictionary",
    )


def load_abide_a(
    data_dir: str | Path,
    sensitive_index: Optional[int] = None,
    drop_sensitive_inputs: bool = False,
) -> ABIDEDataset:
    root = Path(data_dir) / "ABIDE_A"
    csv_path = root / "processed_standard_data.csv"
    modal_path = root / "modal_feat_dict.npy"
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    if not modal_path.exists():
        raise FileNotFoundError(modal_path)

    df = pd.read_csv(csv_path)
    values = df.to_numpy()
    X = values[:, :-1].astype(np.float32)
    y = _normalise_label(values[:, -1])
    modal_feat_dict = np.load(modal_path, allow_pickle=True).item()
    modal_dims = [len(modal_feat_dict[k]) for k in modal_feat_dict.keys()]
    sensitive, sensitive_name, notes = _guess_sensitive_from_dataframe(df.iloc[:, :-1], X, sensitive_index)
    feature_names = [str(c) for c in df.columns[:-1]]
    # Protected-attribute columns that are also model inputs. ABIDE-A encodes sex as a
    # complementary one-hot pair (SEX_1, SEX_2); both columns are recorded and, when the
    # ablation is enabled, both are removed while the sampler keeps the attribute.
    sens_cols = [j for j, name in enumerate(feature_names) if "sex" in name.lower() or "gender" in name.lower()]
    if not sens_cols:
        sens_cols = [j for j, name in enumerate(feature_names) if name == sensitive_name]
    dropped: list[int] = []
    dropped_names: list[str] = []
    if drop_sensitive_inputs and sens_cols:
        dropped = list(sens_cols)
        dropped_names = [feature_names[j] for j in dropped]
        X = _drop_columns(X, dropped)
        feature_names = [n for j, n in enumerate(feature_names) if j not in set(dropped)]
        modal_dims = []
        notes = notes + " | drop_sensitive_inputs=True: removed " + ",".join(dropped_names)
    return ABIDEDataset(
        name="ABIDE_A",
        X=X,
        y=y,
        sensitive=sensitive,
        modal_dims=modal_dims,
        feature_names=feature_names,
        sensitive_name=sensitive_name,
        notes=notes,
        sensitive_feature_cols=sens_cols,
        dropped_feature_indices=dropped,
        dropped_feature_names=dropped_names,
    )


def load_abide_b(
    data_dir: str | Path,
    sensitive_index: Optional[int] = None,
    drop_sensitive_inputs: bool = False,
) -> ABIDEDataset:
    root = Path(data_dir) / "ABIDE_B"
    npz_path = root / "RFE_512_processed_standard_data.npz"
    if not npz_path.exists():
        raise FileNotFoundError(npz_path)
    data = np.load(npz_path, allow_pickle=True)
    X = data["data"].astype(np.float32)
    y = _normalise_label(data["label"])

    modal_dims = [48, 512, 512, 512]
    if "sex" in data:
        sensitive = _binarize_sensitive(data["sex"])
        sensitive_name = "sex"
        notes = "sensitive attribute loaded from npz key 'sex'"
    elif "gender" in data:
        sensitive = _binarize_sensitive(data["gender"])
        sensitive_name = "gender"
        notes = "sensitive attribute loaded from npz key 'gender'"
    elif sensitive_index is not None:
        sensitive = _binarize_sensitive(X[:, sensitive_index])
        sensitive_name = f"feature_{sensitive_index}"
        notes = (
            "WARNING: sensitive attribute was inferred from PHENO feature index "
            f"{sensitive_index}; verify this with the ABIDE_B data dictionary. "
            "For the current FAIR-AL experiments, feature_46 is the intended sex proxy."
        )
    else:
        raise ValueError(
            "ABIDE_B has no explicit 'sex' or 'gender' key. Pass --sensitive-index 46 "
            "for the current sex-proxy experiments, or provide an official sex/gender key in the NPZ file. "
            "Refusing to silently fall back to feature_1 because that caused invalid earlier results."
        )

    # Protected-attribute columns that are also model inputs. For the PHENO-feature-46
    # proxy, feature 47 is its exact complement (one-hot pair); both are removed when the
    # ablation is enabled, while the sampler keeps using the attribute.
    sens_cols: list[int] = []
    if sensitive_index is not None:
        sens_cols = [int(sensitive_index)]
        j = int(sensitive_index)
        if j + 1 < X.shape[1] and np.allclose(X[:, j] + X[:, j + 1], 1.0):
            sens_cols.append(j + 1)
    dropped: list[int] = []
    dropped_names: list[str] = []
    if drop_sensitive_inputs and sens_cols:
        dropped = list(sens_cols)
        dropped_names = [f"col_{j}" for j in dropped]
        X = _drop_columns(X, dropped)
        modal_dims = []
        notes = notes + " | drop_sensitive_inputs=True: removed cols " + ",".join(dropped_names)
    return ABIDEDataset(
        name="ABIDE_B",
        X=X,
        y=y,
        sensitive=sensitive,
        modal_dims=modal_dims,
        sensitive_name=sensitive_name,
        notes=notes,
        sensitive_feature_cols=sens_cols,
        dropped_feature_indices=dropped,
        dropped_feature_names=dropped_names,
    )


def load_dataset(
    data_dir: str | Path,
    dataset: str,
    sensitive_index: Optional[int] = None,
    drop_sensitive_inputs: bool = False,
) -> ABIDEDataset:
    dataset = dataset.upper()
    if dataset == "ABIDE_A":
        return load_abide_a(data_dir, sensitive_index, drop_sensitive_inputs=drop_sensitive_inputs)
    if dataset == "ABIDE_B":
        return load_abide_b(data_dir, sensitive_index, drop_sensitive_inputs=drop_sensitive_inputs)
    raise ValueError(f"Unknown dataset: {dataset}")


def make_stratification_labels(y: np.ndarray, sensitive: np.ndarray) -> np.ndarray:
    return np.asarray([f"y{yy}_a{aa}" for yy, aa in zip(y, sensitive)])


def train_val_test_split_indices(
    y: np.ndarray,
    sensitive: np.ndarray,
    seed: int,
    test_size: float = 0.2,
    val_size: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx = np.arange(len(y))
    strat = make_stratification_labels(y, sensitive)
    try:
        train_val_idx, test_idx = train_test_split(idx, test_size=test_size, random_state=seed, stratify=strat)
    except ValueError:
        train_val_idx, test_idx = train_test_split(idx, test_size=test_size, random_state=seed, stratify=y)

    strat_train_val = strat[train_val_idx]
    rel_val_size = val_size / (1.0 - test_size)
    try:
        train_idx, val_idx = train_test_split(
            train_val_idx, test_size=rel_val_size, random_state=seed, stratify=strat_train_val
        )
    except ValueError:
        train_idx, val_idx = train_test_split(
            train_val_idx, test_size=rel_val_size, random_state=seed, stratify=y[train_val_idx]
        )
    return np.asarray(train_idx), np.asarray(val_idx), np.asarray(test_idx)


def initial_labeled_split(
    pool_idx: np.ndarray,
    y: np.ndarray,
    sensitive: np.ndarray,
    initial_size: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    pool_idx = np.asarray(pool_idx)
    strat = make_stratification_labels(y[pool_idx], sensitive[pool_idx])
    initial_size = min(initial_size, len(pool_idx) - 1)
    try:
        init_idx, unlabeled_idx = train_test_split(pool_idx, train_size=initial_size, random_state=seed, stratify=strat)
    except ValueError:
        init_idx = rng.choice(pool_idx, size=initial_size, replace=False)
        unlabeled_idx = np.setdiff1d(pool_idx, init_idx, assume_unique=False)
    return np.asarray(init_idx), np.asarray(unlabeled_idx)


def binary_column_report(X: np.ndarray, max_cols: int = 80) -> list[dict[str, object]]:
    rows = []
    for j in range(min(max_cols, X.shape[1])):
        vals = np.unique(X[:, j])
        if len(vals) <= 4:
            counts = {str(v): int((X[:, j] == v).sum()) for v in vals}
            rows.append({"index": j, "unique_values": [float(v) for v in vals], "counts": counts})
    return rows

