"""Canonical data and representation pipeline for traffic-CI regression."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from mlblack.core.representation import ModelRepresentation
from mlblack.core.types import UnknownState
from mlblack.pipeline.base import DataPipeline
from mlblack.pipeline.components import ZScoreNormalizeComponent
from mlblack.pipeline.data_views import NumericDataView, train_valid_split


def add_intercept(X):
    values = np.asarray(X, dtype=float)
    return np.column_stack([np.ones(values.shape[0], dtype=float), values])


class CIDirectRepresentation(ModelRepresentation):
    """Linear coefficient-vector representation, including an intercept."""

    context_requires = ()
    context_provides = ("model.coefficients",)
    context_mutates = ()
    context_cache = ()
    context_notes = "Linear coefficient vector for CI regression (bias + feature weights)."

    def __init__(self, n_features, *, name="ci_linear"):
        self.name = str(name)
        self.n_features = int(n_features) + 1

    def init(self, context):
        rng = np.random.default_rng()
        return UnknownState(values=rng.normal(0.0, 0.01, size=(self.n_features,)))

    def encode(self, coeffs, context=None):
        return UnknownState(values=np.asarray(coeffs, dtype=float).ravel())

    def decode(self, state, context=None):
        return np.asarray(state.as_array(), dtype=float).ravel()


def build_pipeline(csv_path: str | Path) -> NumericDataView:
    df = pd.read_csv(Path(csv_path))
    feature_names = tuple(
        column
        for column in df.columns
        if column not in {"date", "ci", "Unnamed: 0"}
        and not column.startswith("test_fold_")
    )
    raw_data = train_valid_split(
        df.loc[:, feature_names].to_numpy(dtype=float),
        df["ci"].to_numpy(dtype=float),
        feature_names=feature_names,
        target_name="ci",
        valid_ratio=0.2,
        seed=42,
    )
    return DataPipeline([ZScoreNormalizeComponent()]).fit_transform(raw_data)


def build_representation(data: NumericDataView) -> CIDirectRepresentation:
    return CIDirectRepresentation(data.n_features)


def run_pipeline_slot(*args, **kwargs) -> NumericDataView:
    return build_pipeline(*args, **kwargs)


__all__ = [
    "CIDirectRepresentation",
    "add_intercept",
    "build_pipeline",
    "build_representation",
    "run_pipeline_slot",
]
