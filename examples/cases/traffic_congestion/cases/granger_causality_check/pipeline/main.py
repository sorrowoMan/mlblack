"""Canonical time-series data pipeline for the Granger causality case."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class GrangerTrafficData:
    frame: pd.DataFrame
    factor_columns: tuple[str, ...]


def build_pipeline(csv_path: str | Path) -> GrangerTrafficData:
    df = pd.read_csv(Path(csv_path))
    candidates = {
        "weather_dummy",
        "wind",
        "aqi",
        "life_impact",
        "is_bad_weather",
        "is_aqi_high",
        "is_holiday_near",
        "is_holiday_mid",
        "is_nonwork_weekend",
        "is_holiday_day_or_window",
    }
    factors = tuple(
        column
        for column in df.columns
        if column in candidates and not column.startswith("test_fold_")
    )
    return GrangerTrafficData(frame=df, factor_columns=factors)


def run_pipeline_slot(*args, **kwargs) -> GrangerTrafficData:
    return build_pipeline(*args, **kwargs)


__all__ = ["GrangerTrafficData", "build_pipeline", "run_pipeline_slot"]
