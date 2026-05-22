from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from mlblack.pipeline.base import DataPipelineComponent
from mlblack.pipeline.data import NumericDataView


@dataclass(frozen=True)
class ModelConditionedTargetConfig:
    """Configuration for building a next-stage target from a reference model."""

    mode: str = "residual"
    reference_name: str = "reference"
    append_prediction_feature: bool = False
    prediction_feature_name: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def describe(self) -> dict[str, Any]:
        return {
            "mode": str(self.mode),
            "reference_name": str(self.reference_name),
            "append_prediction_feature": bool(self.append_prediction_feature),
            "prediction_feature_name": str(self.prediction_feature_name or f"{self.reference_name}_prediction"),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ModelConditionedTargetComponent(DataPipelineComponent):
    """Transforms data by calling an already-trained reference model.

    The common residual stage is mode="residual":

    target_next = y - reference_model.predict(X)

    This keeps residual learning as a target transformation, not a special
    trainer or a special workflow.
    """

    reference_model: Any | None = None
    config: ModelConditionedTargetConfig = field(default_factory=ModelConditionedTargetConfig)
    reference_context_key: str = "reference_model"

    name = "model_conditioned_target"

    def fit(self, data: NumericDataView, context: Mapping[str, Any] | None = None) -> Mapping[str, Any]:
        model = self._resolve_reference_model(context)
        train_pred = _predict_1d(model, data.X_train)
        return {
            "mode": str(self.config.mode),
            "reference_name": str(self.config.reference_name),
            "train_prediction_mean": float(np.mean(train_pred)),
            "train_prediction_std": float(np.std(train_pred)),
        }

    def transform(
        self,
        data: NumericDataView,
        state: Mapping[str, Any] | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> NumericDataView:
        _ = state
        model = self._resolve_reference_model(context)
        train_pred = _predict_1d(model, data.X_train)
        y_train = _transform_target(data.y_train, train_pred, mode=self.config.mode)
        X_train, feature_names = self._maybe_append_prediction_feature(
            data.X_train,
            train_pred,
            feature_names=data.effective_feature_names,
        )

        X_valid = data.X_valid
        y_valid = data.y_valid
        if data.X_valid is not None and data.y_valid is not None:
            valid_pred = _predict_1d(model, data.X_valid)
            y_valid = _transform_target(data.y_valid, valid_pred, mode=self.config.mode)
            X_valid, _ = self._maybe_append_prediction_feature(
                data.X_valid,
                valid_pred,
                feature_names=data.effective_feature_names,
            )

        return NumericDataView(
            X_train=X_train,
            y_train=y_train,
            X_valid=X_valid,
            y_valid=y_valid,
            feature_names=feature_names,
            target_name=f"{data.target_name}.{self.config.mode}",
            metadata={
                **dict(data.metadata),
                "pipeline.model_conditioned_target": self.config.describe(),
                "pipeline.model_conditioned_target.reference_model_type": type(model).__name__,
            },
        )

    def build(
        self,
        data: NumericDataView,
        reference_model: Any | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> NumericDataView:
        ctx = dict(context or {})
        if reference_model is not None:
            ctx[str(self.reference_context_key)] = reference_model
        transformed, _state = self.fit_transform(data, ctx)
        return transformed

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "config": self.config.describe(),
            "reference_model_bound": self.reference_model is not None,
            "reference_context_key": str(self.reference_context_key),
        }

    def _resolve_reference_model(self, context: Mapping[str, Any] | None) -> Any:
        if self.reference_model is not None:
            return self.reference_model
        ctx = dict(context or {})
        key = str(self.reference_context_key)
        if key not in ctx:
            raise KeyError(f"missing reference model in context key {key!r}")
        model = ctx[key]
        if not hasattr(model, "predict") or not callable(getattr(model, "predict")):
            raise TypeError(f"context key {key!r} does not expose predict(X)")
        return model

    def _maybe_append_prediction_feature(
        self,
        X: np.ndarray,
        prediction: np.ndarray,
        *,
        feature_names: tuple[str, ...],
    ) -> tuple[np.ndarray, tuple[str, ...]]:
        X_arr = np.asarray(X, dtype=float)
        if not bool(self.config.append_prediction_feature):
            return X_arr, tuple(feature_names)
        name = str(self.config.prediction_feature_name or f"{self.config.reference_name}_prediction")
        return np.column_stack([X_arr, np.asarray(prediction, dtype=float).reshape(-1)]), tuple(feature_names) + (name,)


def build_model_conditioned_target(
    data: NumericDataView,
    reference_model: Any,
    *,
    mode: str = "residual",
    reference_name: str = "reference",
    append_prediction_feature: bool = False,
    prediction_feature_name: str = "",
    metadata: Mapping[str, Any] | None = None,
) -> NumericDataView:
    component = ModelConditionedTargetComponent(
        reference_model=reference_model,
        config=ModelConditionedTargetConfig(
            mode=str(mode),
            reference_name=str(reference_name),
            append_prediction_feature=bool(append_prediction_feature),
            prediction_feature_name=str(prediction_feature_name),
            metadata=dict(metadata or {}),
        ),
    )
    return component.build(data)


def _predict_1d(model: Any, X: np.ndarray) -> np.ndarray:
    if not hasattr(model, "predict") or not callable(getattr(model, "predict")):
        raise TypeError("reference model must expose predict(X)")
    pred = np.asarray(model.predict(np.asarray(X, dtype=float)), dtype=float).reshape(-1)
    if pred.shape[0] != np.asarray(X).shape[0]:
        raise ValueError("reference model prediction length differs from X rows")
    return pred


def _transform_target(y: np.ndarray, prediction: np.ndarray, *, mode: str) -> np.ndarray:
    target = np.asarray(y, dtype=float).reshape(-1)
    pred = np.asarray(prediction, dtype=float).reshape(-1)
    if target.shape[0] != pred.shape[0]:
        raise ValueError("target and prediction lengths differ")
    key = str(mode or "residual").strip().lower()
    if key in {"residual", "target_minus_prediction", "y_minus_pred"}:
        return np.asarray(target - pred, dtype=float).reshape(-1)
    if key in {"prediction_minus_target", "pred_minus_y"}:
        return np.asarray(pred - target, dtype=float).reshape(-1)
    if key in {"target", "identity", "none"}:
        return target
    if key in {"prediction", "reference_prediction"}:
        return pred
    raise ValueError(f"unsupported model-conditioned target mode: {mode}")


ModelConditionedTargetBuilder = ModelConditionedTargetComponent


__all__ = [
    "ModelConditionedTargetBuilder",
    "ModelConditionedTargetComponent",
    "ModelConditionedTargetConfig",
    "build_model_conditioned_target",
]
