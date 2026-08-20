from __future__ import annotations

import numpy as np

from mlblack.pipeline.data_views import NumericDataView


def build_data() -> NumericDataView:
    X_train = np.asarray(
        [
            [1, 2, 3, 0],
            [2, 2, 1, 0],
            [7, 1, 1, 0],
            [8, 2, 1, 0],
            [1, 3, 5, 0],
            [9, 3, 1, 0],
        ],
        dtype=float,
    )
    X_valid = np.asarray([[1, 1, 2, 0], [8, 1, 1, 0]], dtype=float)
    return NumericDataView(
        X_train=X_train,
        y_train=(X_train[:, 0] >= 5).astype(float),
        X_valid=X_valid,
        y_valid=(X_valid[:, 0] >= 5).astype(float),
    )


__all__ = ["build_data"]
