from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from mlblack.pipeline.data_views import NumericDataView
from mlblack.pipeline.base import DataPipelineComponent


class IdentityComponent(DataPipelineComponent):
    name = "identity"


@dataclass(frozen=True)
class ZScoreNormalizeComponent(DataPipelineComponent):
    name = "zscore_normalize"
    eps: float = 1e-12
    with_mean: bool = True
    with_std: bool = True

    def fit(self, data: NumericDataView, context: Mapping[str, Any] | None = None) -> Mapping[str, Any]:
        _ = context
        X = np.asarray(data.X_train, dtype=float)
        mean = np.mean(X, axis=0) if self.with_mean else np.zeros(X.shape[1], dtype=float)
        std = np.std(X, axis=0) if self.with_std else np.ones(X.shape[1], dtype=float)
        std = np.where(np.abs(std) <= float(self.eps), 1.0, std)
        return {"mean": mean.tolist(), "std": std.tolist()}

    def transform(
        self,
        data: NumericDataView,
        state: Mapping[str, Any] | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> NumericDataView:
        _ = context
        fit_state = dict(state or {})
        mean = np.asarray(fit_state.get("mean", np.zeros(data.n_features)), dtype=float).reshape(1, -1)
        std = np.asarray(fit_state.get("std", np.ones(data.n_features)), dtype=float).reshape(1, -1)

        def transform_X(X: np.ndarray | None) -> np.ndarray | None:
            if X is None:
                return None
            return (np.asarray(X, dtype=float) - mean) / std

        return NumericDataView(
            X_train=transform_X(data.X_train),
            y_train=data.y_train,
            X_valid=transform_X(data.X_valid),
            y_valid=data.y_valid,
            feature_names=data.feature_names,
            target_name=data.target_name,
            metadata={**dict(data.metadata), "pipeline.zscore": True},
        )

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "eps": float(self.eps),
            "with_mean": bool(self.with_mean),
            "with_std": bool(self.with_std),
        }


@dataclass(frozen=True)
class SelectColumnsComponent(DataPipelineComponent):
    name = "select_columns"
    columns: Sequence[int] = ()

    def transform(
        self,
        data: NumericDataView,
        state: Mapping[str, Any] | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> NumericDataView:
        _ = state
        _ = context
        cols = tuple(int(idx) for idx in self.columns)
        if not cols:
            return data
        feature_names = data.effective_feature_names
        selected_names = tuple(feature_names[idx] for idx in cols)
        return NumericDataView(
            X_train=np.asarray(data.X_train, dtype=float)[:, cols],
            y_train=data.y_train,
            X_valid=None if data.X_valid is None else np.asarray(data.X_valid, dtype=float)[:, cols],
            y_valid=data.y_valid,
            feature_names=selected_names,
            target_name=data.target_name,
            metadata={**dict(data.metadata), "pipeline.selected_columns": cols},
        )

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "columns": [int(idx) for idx in self.columns]}


