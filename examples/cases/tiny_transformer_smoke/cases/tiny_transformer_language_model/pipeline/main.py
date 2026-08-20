from __future__ import annotations

import numpy as np

from mlblack.pipeline.data_views import NumericDataView


def build_data() -> NumericDataView:
    X_train = np.asarray(
        [
            [1, 2, 3, 4, 5],
            [1, 2, 2, 3, 4],
            [5, 4, 3, 2, 1],
            [6, 6, 4, 4, 2],
        ],
        dtype=float,
    )
    X_valid = np.asarray([[1, 2, 3, 4, 5], [6, 4, 3, 2, 1]], dtype=float)
    return NumericDataView(
        X_train=X_train,
        y_train=np.zeros(X_train.shape[0], dtype=float),
        X_valid=X_valid,
        y_valid=np.zeros(X_valid.shape[0], dtype=float),
    )


__all__ = ["build_data"]
