from __future__ import annotations

import numpy as np

from mlblack.pipeline.data_views import train_valid_split


def build_data():
    X = np.linspace(-1.0, 1.0, 40).reshape(-1, 1)
    y = 0.25 + 1.75 * X[:, 0]
    return train_valid_split(
        X,
        y,
        valid_ratio=0.25,
        feature_names=("x0",),
    )


__all__ = ["build_data"]
