
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


class ALBackend(Protocol):
    def fit(self, X: np.ndarray, y: np.ndarray) -> "ALBackend": ...
    def predict_proba(self, X: np.ndarray) -> np.ndarray: ...
    def embed(self, X: np.ndarray) -> np.ndarray: ...


@dataclass
class SklearnBackend:
    classifier: str = "svm"
    seed: int = 0
    max_iter: int = 1000

    def __post_init__(self) -> None:
        if self.classifier == "svm":
            base = SVC(C=1.0, kernel="rbf", probability=True, class_weight="balanced", random_state=self.seed)
        elif self.classifier == "lr":
            base = LogisticRegression(max_iter=self.max_iter, class_weight="balanced", random_state=self.seed)
        elif self.classifier == "mlp":
            base = MLPClassifier(
                hidden_layer_sizes=(128, 64),
                activation="relu",
                alpha=1e-4,
                learning_rate_init=1e-3,
                max_iter=self.max_iter,
                early_stopping=True,
                random_state=self.seed,
            )
        else:
            raise ValueError(f"Unknown classifier: {self.classifier}")
        self.model = make_pipeline(StandardScaler(), base)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SklearnBackend":
        self.model.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        prob = self.model.predict_proba(X)
        if prob.shape[1] == 1:
            p1 = prob[:, 0]
            return np.column_stack([1 - p1, p1])
        return prob

    def embed(self, X: np.ndarray) -> np.ndarray:
        scaler = self.model.named_steps.get("standardscaler")
        if scaler is not None:
            return scaler.transform(X)
        return X


def make_backend(
    backend: str,
    classifier: str,
    seed: int,
    modal_dims: list[int] | None = None,
    epochs: int = 100,
    device: str = "auto",
    lr: float = 1e-3,
) -> ALBackend:
    if backend == "sklearn":
        return SklearnBackend(classifier=classifier, seed=seed)
    if backend == "deepasd":
        from .deepasd_style import DeepASDStyleBackend

        if modal_dims is None:
            raise ValueError("modal_dims are required for the DeepASD-style backend")
        return DeepASDStyleBackend(modal_dims=modal_dims, seed=seed, epochs=epochs, device=device, lr=lr)
    raise ValueError(f"Unknown backend: {backend}")

