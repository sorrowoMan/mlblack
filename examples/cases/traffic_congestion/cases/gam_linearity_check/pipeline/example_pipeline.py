# -*- coding: utf-8 -*-
"""Example data pipeline for supervised regression."""

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
