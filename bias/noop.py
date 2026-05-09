from __future__ import annotations

import numpy as np

from .base import BaseTrainingBias, FitContext


class NoOpBias(BaseTrainingBias):
    name = "noop"

    def apply(self, X: np.ndarray, Y: np.ndarray, context: FitContext) -> tuple[np.ndarray, np.ndarray]:
        _ = context
        return np.asarray(X, dtype=float), np.asarray(Y, dtype=float)
