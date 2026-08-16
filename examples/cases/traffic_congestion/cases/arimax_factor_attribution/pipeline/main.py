"""Canonical data pipeline for the ARIMAX factor-attribution case."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class ArimaxFactorData:
    X_scaled: np.ndarray
    y: np.ndarray
    factor_groups: Mapping[str, tuple[str, ...]]
    feature_names: tuple[str, ...]
    scaler: StandardScaler


def build_pipeline(
    csv_path: str | Path,
    *,
    factor_groups: Mapping[str, Sequence[str]],
) -> ArimaxFactorData:
    """Load, validate, and standardize the exogenous factor matrix."""

    df = pd.read_csv(Path(csv_path))
    excluded = {
        "date",
        "ci",
        "Unnamed: 0",
        "dow",
        "month",
        *(f"test_fold_{index}" for index in range(1, 11)),
    }
    available_features = [
        column
        for column in df.columns
        if column not in excluded and not column.startswith("test_fold_")
    ]
    available = {
        name: tuple(feature for feature in features if feature in available_features)
        for name, features in factor_groups.items()
    }
    available = {name: features for name, features in available.items() if features}
    feature_names = tuple(feature for features in available.values() for feature in features)
    X = df.loc[:, feature_names].to_numpy(dtype=float)
    y = df["ci"].to_numpy(dtype=float)
    valid = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
    X = X[valid]
    y = y[valid]
    scaler = StandardScaler()
    return ArimaxFactorData(
        X_scaled=scaler.fit_transform(X),
        y=y,
        factor_groups=available,
        feature_names=feature_names,
        scaler=scaler,
    )


def run_pipeline_slot(*args, **kwargs) -> ArimaxFactorData:
    return build_pipeline(*args, **kwargs)


__all__ = ["ArimaxFactorData", "build_pipeline", "run_pipeline_slot"]
