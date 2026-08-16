"""Canonical supervised-data pipeline for the GAM linearity case."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler

from .example_pipeline import build_data_view


def build_pipeline(csv_path: str | Path):
    df = pd.read_csv(Path(csv_path))
    feature_names = tuple(
        column
        for column in df.columns
        if column not in {"date", "ci", "Unnamed: 0"}
        and not column.startswith("test_fold_")
    )
    X = df.loc[:, feature_names].to_numpy(dtype=float)
    y = df["ci"].to_numpy(dtype=float)
    X_scaled = StandardScaler().fit_transform(X)
    return build_data_view(X_scaled, y, feature_names=feature_names, target_name="ci")


def run_pipeline_slot(*args, **kwargs):
    return build_pipeline(*args, **kwargs)


__all__ = ["build_pipeline", "run_pipeline_slot"]
