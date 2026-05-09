from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from my_project.features.example_feature_builder import FeatureBundle


@dataclass(frozen=True)
class ModelResult:
    prediction: np.ndarray
    rmse: float


def train_and_predict(bundle: FeatureBundle, baseline: str) -> ModelResult:
    y = np.asarray(bundle.y, dtype=float).reshape(-1)
    if str(baseline).lower() == "mean":
        pred = np.full_like(y, float(np.mean(y)))
    else:
        pred = np.zeros_like(y)
    rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
    return ModelResult(prediction=pred, rmse=rmse)
