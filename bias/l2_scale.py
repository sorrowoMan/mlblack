from __future__ import annotations

import numpy as np

from .base import BaseTrainingBias, FitContext


class L2ScaleBias(BaseTrainingBias):
    name = "l2_scale"

    def __init__(self, scale: float = 1.0) -> None:
        self.scale = float(scale)

    def apply(self, X: np.ndarray, Y: np.ndarray, context: FitContext) -> tuple[np.ndarray, np.ndarray]:
        context.l2_multiplier *= self.scale
        context.metadata["l2_scale"] = float(self.scale)
        return np.asarray(X, dtype=float), np.asarray(Y, dtype=float)
