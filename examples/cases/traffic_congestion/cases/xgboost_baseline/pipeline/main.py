"""Canonical data pipeline for the XGBoost baseline case."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from mlblack.pipeline.data_views import NumericDataView, train_valid_split


def build_pipeline(csv_path: str | Path) -> NumericDataView:
    df = pd.read_csv(Path(csv_path))
    feature_names = tuple(
        column
        for column in df.columns
        if column not in {"date", "ci", "Unnamed: 0"}
        and not column.startswith("test_fold_")
    )
    return train_valid_split(
        df.loc[:, feature_names].to_numpy(dtype=float),
        df["ci"].to_numpy(dtype=float),
        feature_names=feature_names,
        target_name="ci",
        valid_ratio=0.2,
        seed=42,
    )


def run_pipeline_slot(*args, **kwargs) -> NumericDataView:
    return build_pipeline(*args, **kwargs)


__all__ = ["build_pipeline", "run_pipeline_slot"]
