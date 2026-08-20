from __future__ import annotations

import numpy as np

from mlblack.pipeline.data_views import ImageDataView


def build_data() -> ImageDataView:
    X = np.zeros((6, 1, 4, 4), dtype=float)
    X[:3, :, :2, :2] = 1.0
    X[3:, :, 2:, 2:] = 1.0
    y = np.asarray([0, 0, 0, 1, 1, 1], dtype=float)
    return ImageDataView(X_train=X, y_train=y)


__all__ = ["build_data"]
