from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class IntervalPredictionModel:
    """Two-bound interval predictor built from lower and upper models."""

    lower_model: Any
    upper_model: Any
    enforce_order: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def predict_interval(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        lower = np.asarray(self.lower_model.predict(X), dtype=float).reshape(-1)
        upper = np.asarray(self.upper_model.predict(X), dtype=float).reshape(-1)
        if self.enforce_order:
            lo = np.minimum(lower, upper)
            hi = np.maximum(lower, upper)
            return lo, hi
        return lower, upper

    def predict(self, X: np.ndarray) -> np.ndarray:
        lower, upper = self.predict_interval(X)
        return np.column_stack([lower, upper])


@dataclass(frozen=True)
class CenterRadiusIntervalModel:
    """Interval predictor built from center and radius models."""

    center_model: Any
    radius_model: Any
    radius_transform: str = "softplus"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def predict_interval(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        center = np.asarray(self.center_model.predict(X), dtype=float).reshape(-1)
        raw_radius = np.asarray(self.radius_model.predict(X), dtype=float).reshape(-1)
        radius = _positive_radius(raw_radius, self.radius_transform)
        return center - radius, center + radius

    def predict(self, X: np.ndarray) -> np.ndarray:
        lower, upper = self.predict_interval(X)
        return np.column_stack([lower, upper])


def _positive_radius(values: np.ndarray, transform: str) -> np.ndarray:
    key = str(transform or "softplus").strip().lower()
    arr = np.asarray(values, dtype=float)
    if key == "softplus":
        return np.log1p(np.exp(-np.abs(arr))) + np.maximum(arr, 0.0)
    if key == "abs":
        return np.abs(arr)
    if key == "exp":
        return np.exp(np.clip(arr, -50.0, 50.0))
    if key in {"identity", "none"}:
        return np.maximum(arr, 0.0)
    raise ValueError(f"unsupported radius transform: {transform}")
