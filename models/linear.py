from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class LinearPointModel:
    """Decoded point model: f(x) = intercept + X @ weights."""

    intercept: float
    weights: np.ndarray
    feature_names: Sequence[str] = field(default_factory=tuple)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_arr = np.asarray(X, dtype=float)
        weights = np.asarray(self.weights, dtype=float).reshape(-1)
        if X_arr.ndim != 2:
            raise ValueError("X must be 2D")
        if X_arr.shape[1] != weights.shape[0]:
            raise ValueError(f"X has {X_arr.shape[1]} columns but model expects {weights.shape[0]}")
        return np.asarray(float(self.intercept) + X_arr @ weights, dtype=float).reshape(-1)

    def parameter_gradient(self, X: np.ndarray, y: np.ndarray, *, l2: float = 0.0) -> np.ndarray:
        X_arr = np.asarray(X, dtype=float)
        target = np.asarray(y, dtype=float).reshape(-1)
        pred = self.predict(X_arr)
        err = pred - target
        n = max(1, int(err.shape[0]))
        grad_intercept = 2.0 * float(np.mean(err))
        grad_weights = (2.0 / float(n)) * (X_arr.T @ err) + (2.0 * float(l2) * np.asarray(self.weights, dtype=float))
        return np.concatenate([[grad_intercept], np.asarray(grad_weights, dtype=float).reshape(-1)])


@dataclass(frozen=True)
class OrthogonalFeatureMap:
    """Data-side feature map used by decoded point models.

    It maps raw numeric X to an orthogonal basis Q. The map is prepared once
    before optimization and then used by decoded models during evaluation.
    """

    mean: np.ndarray
    components: np.ndarray
    raw_feature_names: Sequence[str]
    expanded_feature_names: Sequence[str]
    selected_energy: float
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def fit(
        cls,
        X: np.ndarray,
        *,
        feature_names: Sequence[str] | None = None,
        include_raw: bool = True,
        include_square: bool = True,
        include_interactions: bool = True,
        max_components: int | None = None,
        energy_threshold: float | None = 0.999,
    ) -> "OrthogonalFeatureMap":
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim != 2:
            raise ValueError("X must be 2D")
        names = tuple(feature_names or tuple(f"x{i}" for i in range(X_arr.shape[1])))
        expanded, expanded_names = _expand_features(
            X_arr,
            names,
            include_raw=bool(include_raw),
            include_square=bool(include_square),
            include_interactions=bool(include_interactions),
        )
        mean = np.mean(expanded, axis=0)
        centered = expanded - mean
        if centered.shape[0] == 0 or centered.shape[1] == 0:
            raise ValueError("expanded feature matrix is empty")
        _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
        energy = singular_values ** 2
        total_energy = float(np.sum(energy))
        if total_energy <= 0.0:
            rank = 1
            selected_energy = 0.0
        else:
            cumulative = np.cumsum(energy) / total_energy
            if energy_threshold is None:
                rank = len(singular_values)
            else:
                rank = int(np.searchsorted(cumulative, float(energy_threshold)) + 1)
            selected_energy = float(cumulative[min(rank - 1, len(cumulative) - 1)])
        if max_components is not None:
            rank = min(rank, int(max_components))
        rank = max(1, min(rank, vt.shape[0]))
        components = np.asarray(vt[:rank, :], dtype=float)
        selected_energy = 0.0 if total_energy <= 0.0 else float(np.sum(energy[:rank]) / total_energy)
        return cls(
            mean=np.asarray(mean, dtype=float),
            components=components,
            raw_feature_names=names,
            expanded_feature_names=tuple(expanded_names),
            selected_energy=selected_energy,
            metadata={
                "include_raw": bool(include_raw),
                "include_square": bool(include_square),
                "include_interactions": bool(include_interactions),
                "expanded_dim": int(expanded.shape[1]),
                "rank": int(rank),
                "energy_threshold": energy_threshold,
                "selected_energy": selected_energy,
            },
        )

    @property
    def output_dim(self) -> int:
        return int(self.components.shape[0])

    def transform(self, X: np.ndarray) -> np.ndarray:
        X_arr = np.asarray(X, dtype=float)
        expanded, _ = _expand_features(
            X_arr,
            tuple(self.raw_feature_names),
            include_raw=bool(self.metadata.get("include_raw", True)),
            include_square=bool(self.metadata.get("include_square", True)),
            include_interactions=bool(self.metadata.get("include_interactions", True)),
        )
        centered = expanded - self.mean
        return centered @ self.components.T

    def feature_names(self) -> tuple[str, ...]:
        return tuple(f"q{i}" for i in range(self.output_dim))


@dataclass(frozen=True)
class OrthogonalLinearPointModel:
    """Decoded point model: f(x) = intercept + Q(x) @ weights."""

    feature_map: OrthogonalFeatureMap
    intercept: float
    weights: np.ndarray
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def predict(self, X: np.ndarray) -> np.ndarray:
        Q = self.feature_map.transform(np.asarray(X, dtype=float))
        return np.asarray(float(self.intercept) + Q @ np.asarray(self.weights, dtype=float), dtype=float).reshape(-1)

    def parameter_gradient(self, X: np.ndarray, y: np.ndarray, *, l2: float = 0.0) -> np.ndarray:
        Q = self.feature_map.transform(np.asarray(X, dtype=float))
        target = np.asarray(y, dtype=float).reshape(-1)
        pred = self.predict(X)
        err = pred - target
        n = max(1, int(err.shape[0]))
        grad_intercept = 2.0 * float(np.mean(err))
        grad_weights = (2.0 / float(n)) * (Q.T @ err) + (2.0 * float(l2) * np.asarray(self.weights, dtype=float))
        return np.concatenate([[grad_intercept], np.asarray(grad_weights, dtype=float).reshape(-1)])


def _expand_features(
    X: np.ndarray,
    feature_names: Sequence[str],
    *,
    include_raw: bool,
    include_square: bool,
    include_interactions: bool,
) -> tuple[np.ndarray, tuple[str, ...]]:
    X_arr = np.asarray(X, dtype=float)
    cols: list[np.ndarray] = []
    names: list[str] = []
    base_names = tuple(str(x) for x in feature_names)
    if include_raw:
        for j, name in enumerate(base_names):
            cols.append(X_arr[:, j])
            names.append(name)
    if include_square:
        for j, name in enumerate(base_names):
            cols.append(X_arr[:, j] ** 2)
            names.append(f"{name}^2")
    if include_interactions:
        for i in range(X_arr.shape[1]):
            for j in range(i + 1, X_arr.shape[1]):
                cols.append(X_arr[:, i] * X_arr[:, j])
                names.append(f"{base_names[i]}*{base_names[j]}")
    if not cols:
        raise ValueError("at least one feature expansion block must be enabled")
    return np.column_stack(cols), tuple(names)
