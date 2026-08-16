"""Canonical DataView pipeline for the tiny Transformer smoke Case."""

from __future__ import annotations

import numpy as np

from mlblack.pipeline.data_views import NumericDataView, PreferencePairDataView


def build_classification_data() -> NumericDataView:
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
    y_train = (X_train[:, 0] >= 5).astype(float)
    X_valid = np.asarray([[1, 1, 2, 0], [8, 1, 1, 0]], dtype=float)
    y_valid = (X_valid[:, 0] >= 5).astype(float)
    return NumericDataView(X_train=X_train, y_train=y_train, X_valid=X_valid, y_valid=y_valid)


def build_lm_data() -> NumericDataView:
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


def build_preference_data() -> PreferencePairDataView:
    return PreferencePairDataView(
        chosen_train=np.asarray(
            [[1, 2, 3, 4], [1, 3, 4, 5], [2, 3, 5, 6]],
            dtype=float,
        ),
        rejected_train=np.asarray(
            [[1, 2, 2, 2], [1, 3, 3, 3], [2, 3, 3, 3]],
            dtype=float,
        ),
    )


def build_pipeline() -> dict[str, object]:
    """Build all DataViews consumed by the three smoke subflows."""

    return {
        "classification": build_classification_data(),
        "language_model": build_lm_data(),
        "preference": build_preference_data(),
    }


__all__ = ["build_classification_data", "build_lm_data", "build_pipeline", "build_preference_data"]
