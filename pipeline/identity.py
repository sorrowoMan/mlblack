from __future__ import annotations

import numpy as np

from .base import BasePipeline


def _as_2d(arr: np.ndarray) -> np.ndarray:
    x = np.asarray(arr, dtype=float)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    return x


class IdentityPipeline(BasePipeline):
    name = "identity"

    def transform(self, X: np.ndarray) -> np.ndarray:
        return _as_2d(X)
