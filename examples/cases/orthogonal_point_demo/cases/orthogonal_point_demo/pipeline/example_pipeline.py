# -*- coding: utf-8 -*-
"""Example data pipeline for the orthogonal point demo."""

from __future__ import annotations

import numpy as np

from mlblack.pipeline.data_views import NumericDataView


def build_data_view(
    X,
    y,
    *,
    feature_names=(),
    target_name="target",
):
    return NumericDataView(
        X_train=X,
        y_train=y,
        feature_names=list(feature_names or tuple(f"x{i}" for i in range(X.shape[1]))),
        target_name=target_name,
    )


def build_orthogonal_point_demo_data_view(
    *,
    seed: int = 7,
    n_samples: int = 240,
    valid_ratio: float = 0.2,
    feature_names=("x1", "x2"),
):
    from mlblack.pipeline.data_views import train_valid_split

    rng = np.random.default_rng(int(seed))
    X = rng.normal(size=(int(n_samples), 2))
    y = 2.0 + 3.0 * X[:, 0] + 4.0 * X[:, 1] + 1.5 * X[:, 0] * X[:, 1] + rng.normal(scale=0.05, size=int(n_samples))
    return train_valid_split(X, y, valid_ratio=float(valid_ratio), seed=int(seed), feature_names=tuple(feature_names))
