"""Canonical feature pipeline for the contribution-consistency case."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler

from mlblack.pipeline.data_views import NumericDataView


def build_pipeline(csv_path: str | Path) -> NumericDataView:
    df = pd.read_csv(Path(csv_path))
    feature_names = tuple(
        column
        for column in df.columns
        if column not in {"date", "ci", "Unnamed: 0"}
        and not column.startswith("test_fold_")
    )
    X = df.loc[:, feature_names].to_numpy(dtype=float)
    y = df["ci"].to_numpy(dtype=float)
    return NumericDataView(
        X_train=StandardScaler().fit_transform(X),
        y_train=y,
        feature_names=feature_names,
        target_name="ci",
    )


def run_pipeline_slot(*args, **kwargs) -> NumericDataView:
    return build_pipeline(*args, **kwargs)


__all__ = ["build_pipeline", "run_pipeline_slot"]
