from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np


class Router:
    name = "router"

    def route(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def describe(self) -> dict[str, Any]:
        return {"name": self.name}


@dataclass(frozen=True)
class ThresholdRouter(Router):
    """Route rows by threshold bins on one feature column."""

    feature_index: int = 0
    thresholds: Sequence[float] = field(default_factory=tuple)
    name = "threshold_router"

    def route(self, X: np.ndarray) -> np.ndarray:
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim != 2:
            raise ValueError("X must be 2D")
        values = X_arr[:, int(self.feature_index)]
        return np.searchsorted(np.asarray(tuple(self.thresholds), dtype=float), values, side="right").astype(int)

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "feature_index": int(self.feature_index),
            "thresholds": [float(v) for v in self.thresholds],
        }


@dataclass(frozen=True)
class PiecewiseModel:
    router: Router
    branch_models: Sequence[Any]
    default_branch: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_arr = np.asarray(X, dtype=float)
        routes = np.asarray(self.router.route(X_arr), dtype=int).reshape(-1)
        out = np.zeros(X_arr.shape[0], dtype=float)
        branches = tuple(self.branch_models)
        if not branches:
            raise ValueError("PiecewiseModel requires at least one branch model")
        default = int(np.clip(int(self.default_branch), 0, len(branches) - 1))
        for idx in range(X_arr.shape[0]):
            branch_idx = int(routes[idx])
            if branch_idx < 0 or branch_idx >= len(branches):
                branch_idx = default
            out[idx] = float(np.asarray(branches[branch_idx].predict(X_arr[idx : idx + 1]), dtype=float).reshape(-1)[0])
        return out

    def describe(self) -> dict[str, Any]:
        return {
            "router": self.router.describe(),
            "num_branches": len(tuple(self.branch_models)),
            "default_branch": int(self.default_branch),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ProbabilityCalibrationModel:
    """Simple probability calibration wrapper."""

    base_model: Any
    temperature: float = 1.0
    clip_eps: float = 1e-12
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if hasattr(self.base_model, "predict_proba"):
            proba = np.asarray(self.base_model.predict_proba(X), dtype=float)
        else:
            pred = np.asarray(self.base_model.predict(X), dtype=float).reshape(-1)
            proba = np.column_stack([1.0 - pred, pred])
        proba = np.clip(proba, float(self.clip_eps), 1.0)
        logits = np.log(proba)
        scaled = logits / max(float(self.temperature), float(self.clip_eps))
        scaled = scaled - np.max(scaled, axis=1, keepdims=True)
        exp = np.exp(scaled)
        return exp / np.sum(exp, axis=1, keepdims=True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)
