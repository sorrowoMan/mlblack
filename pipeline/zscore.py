from __future__ import annotations

from typing import Any, Dict

import numpy as np

from .base import BasePipeline


def _as_2d(arr: np.ndarray) -> np.ndarray:
    x = np.asarray(arr, dtype=float)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    return x


class ZScorePipeline(BasePipeline):
    name = "zscore"

    def __init__(self, eps: float = 1e-8) -> None:
        self.eps = float(eps)
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "ZScorePipeline":
        _ = y
        x = _as_2d(X)
        self.mean_ = np.mean(x, axis=0)
        self.std_ = np.std(x, axis=0) + self.eps
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("ZScorePipeline must be fitted before transform")
        x = _as_2d(X)
        return (x - self.mean_) / self.std_

    def state_dict(self) -> Dict[str, Any]:
        return {
            "eps": float(self.eps),
            "mean": None if self.mean_ is None else self.mean_.tolist(),
            "std": None if self.std_ is None else self.std_.tolist(),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> "ZScorePipeline":
        self.eps = float(state.get("eps", 1e-8))
        mean = state.get("mean")
        std = state.get("std")
        self.mean_ = None if mean is None else np.asarray(mean, dtype=float)
        self.std_ = None if std is None else np.asarray(std, dtype=float)
        return self
