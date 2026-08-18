from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from blackbase.contracts import ComponentContract
from mlblack.core.problem import LearningProblem
from mlblack.core.types import Feedback, UnknownState
from mlblack.pipeline.data_views import TimeSeriesDataView


class TimeSeriesForecastingProblem(LearningProblem):
    """Evaluator for univariate time-series forecasting models."""

    name = "time_series_forecasting"
    context_requires = ("candidate.forecast_model", "data.time_series_view")
    context_optional = ("resource.context", "time_series.validation_size", "time_series.objective_metrics")
    context_provides = ("feedback.objectives", "feedback.loss", "feedback.metrics", "feedback.residuals", "feedback.signals")
    context_mutates = ()
    context_cache = ()
    requires_metrics = ("rmse", "mae")
    metrics_fallback = "strict"
    context_notes = "Evaluates forecast(history, horizon) against a holdout tail or rolling-origin slice."
    contract = ComponentContract(
        name=name,
        requires=("candidate.forecast_model", "data.time_series_view"),
        optional=("resource.context", "time_series.validation_size", "time_series.objective_metrics"),
        provides=("feedback.objectives", "feedback.loss", "feedback.metrics", "feedback.residuals", "feedback.signals"),
        supports_gradient=False,
        supports_batch=False,
        supports_resume=False,
        metadata={"family": "time_series", "task": "forecasting"},
    )

    def __init__(
        self,
        data: TimeSeriesDataView,
        *,
        validation_size: int | float = 0.2,
        objective_metrics: Sequence[str] = ("valid.rmse", "valid.mae"),
        seasonal_period: int = 1,
        complexity_weight: float = 0.0,
    ) -> None:
        self.data = data
        self.validation_size = validation_size
        self.objective_metrics = tuple(str(metric) for metric in objective_metrics)
        self.seasonal_period = int(seasonal_period)
        self.complexity_weight = float(complexity_weight)

    def evaluate(self, model: Any, state: UnknownState, context: Mapping[str, Any]) -> Feedback:
        valid_size = _resolve_validation_size(context.get("time_series.validation_size", self.validation_size), self.data.n_obs)
        history = self.data.history_before_tail(valid_size)
        target = self.data.tail_target(valid_size)
        exogenous_future = None
        if self.data.exogenous is not None:
            exogenous_future = np.asarray(self.data.exogenous, dtype=float)[-valid_size:, :]
        pred = np.asarray(
            model.forecast(
                history,
                int(valid_size),
                exogenous_future=exogenous_future,
                context=dict(context or {}),
            ),
            dtype=float,
        ).reshape(-1)
        if pred.shape[0] != target.shape[0]:
            raise ValueError("forecast length differs from validation target length")

        metrics = _forecast_metrics(
            target,
            pred,
            train_history=history,
            prefix="valid",
            seasonal_period=self.seasonal_period,
        )
        residual = pred - target
        complexity = _model_complexity(model, state)
        metrics.update(
            {
                "complexity.model": float(complexity),
                "time_series.validation_size": int(valid_size),
                "time_series.history_size": int(history.shape[0]),
                "time_series.seasonal_period": int(self.seasonal_period),
            }
        )
        objectives = [float(_metric_value(metrics, metric)) for metric in self.objective_metrics]
        if self.complexity_weight:
            objectives.append(float(self.complexity_weight) * float(complexity))
        return Feedback(
            objectives=np.asarray(objectives, dtype=float),
            constraints=np.zeros(0, dtype=float),
            loss=float(objectives[0]) if objectives else float(metrics["valid.rmse"]),
            gradients=None,
            residuals=residual,
            metrics=metrics,
            signals={
                "task": "time_series_forecasting",
                "primary_objective": self.objective_metrics[0] if self.objective_metrics else "valid.rmse",
                "has_gradient": False,
                "forecast_horizon": int(valid_size),
                "model_type": type(model).__name__,
            },
        )

    def describe(self) -> Mapping[str, Any]:
        return {
            "name": self.name,
            "family": "time_series",
            "task": "forecasting",
            "n_obs": int(self.data.n_obs),
            "validation_size": self.validation_size,
            "objective_metrics": tuple(self.objective_metrics),
            "seasonal_period": int(self.seasonal_period),
            "complexity_weight": float(self.complexity_weight),
            "has_exogenous": bool(self.data.has_exogenous),
        }


class RollingOriginForecastingProblem(TimeSeriesForecastingProblem):
    """Rolling-origin evaluator for horizon-specific forecast quality."""

    name = "rolling_origin_forecasting"
    context_requires = ("candidate.forecast_model", "data.time_series_view")
    context_optional = ("resource.context", "time_series.min_train_size", "time_series.horizon")
    context_provides = ("feedback.objectives", "feedback.loss", "feedback.metrics", "feedback.residuals", "feedback.signals")
    context_mutates = ()
    context_cache = ()
    context_notes = "Runs repeated rolling-origin forecasts and aggregates horizon-specific errors."
    contract = ComponentContract(
        name=name,
        requires=("candidate.forecast_model", "data.time_series_view"),
        optional=("resource.context", "time_series.min_train_size", "time_series.horizon"),
        provides=("feedback.objectives", "feedback.loss", "feedback.metrics", "feedback.residuals", "feedback.signals"),
        supports_gradient=False,
        supports_batch=False,
        supports_resume=False,
        metadata={"family": "time_series", "task": "rolling_origin_forecasting"},
    )

    def __init__(
        self,
        data: TimeSeriesDataView,
        *,
        min_train_size: int | float = 0.6,
        horizon: int = 1,
        max_origins: int | None = None,
        objective_metrics: Sequence[str] = ("rolling.rmse", "rolling.mae"),
        seasonal_period: int = 1,
        complexity_weight: float = 0.0,
    ) -> None:
        super().__init__(
            data,
            validation_size=0,
            objective_metrics=objective_metrics,
            seasonal_period=seasonal_period,
            complexity_weight=complexity_weight,
        )
        self.min_train_size = min_train_size
        self.horizon = int(horizon)
        self.max_origins = None if max_origins is None else int(max_origins)

    def evaluate(self, model: Any, state: UnknownState, context: Mapping[str, Any]) -> Feedback:
        horizon = int(context.get("time_series.horizon", self.horizon))
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        min_train = _resolve_min_train_size(context.get("time_series.min_train_size", self.min_train_size), self.data.n_obs)
        last_origin = self.data.n_obs - horizon - 1
        if last_origin < min_train - 1:
            raise ValueError("series is too short for rolling-origin evaluation")
        origins = list(range(min_train - 1, last_origin + 1))
        if self.max_origins is not None and self.max_origins > 0 and len(origins) > self.max_origins:
            idx = np.linspace(0, len(origins) - 1, num=self.max_origins, dtype=int)
            origins = [origins[int(i)] for i in idx]

        preds: list[float] = []
        targets: list[float] = []
        for origin in origins:
            history = np.asarray(self.data.y, dtype=float)[: origin + 1]
            exogenous_future = None
            if self.data.exogenous is not None:
                exogenous_future = np.asarray(self.data.exogenous, dtype=float)[origin + 1 : origin + 1 + horizon, :]
            forecast = np.asarray(
                model.forecast(
                    history,
                    int(horizon),
                    exogenous_future=exogenous_future,
                    context=dict(context or {}),
                ),
                dtype=float,
            ).reshape(-1)
            preds.append(float(forecast[-1]))
            targets.append(float(np.asarray(self.data.y, dtype=float)[origin + horizon]))

        pred_arr = np.asarray(preds, dtype=float)
        target_arr = np.asarray(targets, dtype=float)
        train_history = np.asarray(self.data.y, dtype=float)[:min_train]
        metrics = _forecast_metrics(
            target_arr,
            pred_arr,
            train_history=train_history,
            prefix="rolling",
            seasonal_period=self.seasonal_period,
        )
        residual = pred_arr - target_arr
        complexity = _model_complexity(model, state)
        metrics.update(
            {
                "rolling.origins": int(len(origins)),
                "rolling.horizon": int(horizon),
                "rolling.min_train_size": int(min_train),
                "complexity.model": float(complexity),
            }
        )
        objectives = [float(_metric_value(metrics, metric)) for metric in self.objective_metrics]
        if self.complexity_weight:
            objectives.append(float(self.complexity_weight) * float(complexity))
        return Feedback(
            objectives=np.asarray(objectives, dtype=float),
            constraints=np.zeros(0, dtype=float),
            loss=float(objectives[0]) if objectives else float(metrics["rolling.rmse"]),
            gradients=None,
            residuals=residual,
            metrics=metrics,
            signals={
                "task": "rolling_origin_forecasting",
                "primary_objective": self.objective_metrics[0] if self.objective_metrics else "rolling.rmse",
                "has_gradient": False,
                "forecast_horizon": int(horizon),
                "model_type": type(model).__name__,
            },
        )

    def describe(self) -> Mapping[str, Any]:
        base = dict(super().describe())
        base.update(
            {
                "name": self.name,
                "task": "rolling_origin_forecasting",
                "min_train_size": self.min_train_size,
                "horizon": int(self.horizon),
                "max_origins": self.max_origins,
            }
        )
        return base


def _forecast_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    train_history: np.ndarray,
    prefix: str,
    seasonal_period: int,
) -> dict[str, float]:
    y = np.asarray(y_true, dtype=float).reshape(-1)
    pred = np.asarray(y_pred, dtype=float).reshape(-1)
    if y.shape[0] != pred.shape[0]:
        raise ValueError("prediction length differs from target length")
    err = pred - y
    abs_err = np.abs(err)
    mse = float(np.mean(err ** 2))
    mae = float(np.mean(abs_err))
    denom = np.maximum(np.abs(y), 1e-12)
    mape = float(np.mean(abs_err / denom))
    smape = float(np.mean((2.0 * abs_err) / np.maximum(np.abs(y) + np.abs(pred), 1e-12)))
    scale = _mase_scale(train_history, seasonal_period=seasonal_period)
    mase = float(mae / scale) if scale > 0.0 else (0.0 if mae <= 1e-12 else float("inf"))
    bias = float(np.mean(err))
    return {
        f"{prefix}.mse": mse,
        f"{prefix}.rmse": float(np.sqrt(mse)),
        f"{prefix}.mae": mae,
        f"{prefix}.mape": mape,
        f"{prefix}.smape": smape,
        f"{prefix}.mase": mase,
        f"{prefix}.bias": bias,
    }


def _mase_scale(train_history: np.ndarray, *, seasonal_period: int) -> float:
    y = np.asarray(train_history, dtype=float).reshape(-1)
    lag = max(1, int(seasonal_period))
    if y.shape[0] <= lag:
        return float(np.mean(np.abs(np.diff(y)))) if y.shape[0] > 1 else 0.0
    return float(np.mean(np.abs(y[lag:] - y[:-lag])))


def _resolve_validation_size(value: int | float, n_obs: int) -> int:
    if isinstance(value, float) and 0.0 < float(value) < 1.0:
        count = max(1, int(round(float(value) * float(n_obs))))
    else:
        count = int(value)
    if count <= 0:
        raise ValueError("validation_size must be positive")
    if count >= int(n_obs):
        raise ValueError("validation_size must be smaller than series length")
    return count


def _resolve_min_train_size(value: int | float, n_obs: int) -> int:
    if isinstance(value, float) and 0.0 < float(value) < 1.0:
        count = max(2, int(round(float(value) * float(n_obs))))
    else:
        count = int(value)
    if count < 2:
        raise ValueError("min_train_size must be at least 2")
    if count >= int(n_obs):
        raise ValueError("min_train_size must be smaller than series length")
    return count


def _metric_value(metrics: Mapping[str, Any], metric: str) -> float:
    key = str(metric)
    if key not in metrics:
        raise KeyError(f"unknown forecast metric: {metric}")
    return float(metrics[key])


def _model_complexity(model: Any, state: UnknownState) -> float:
    if hasattr(model, "weights"):
        try:
            return float(np.count_nonzero(np.abs(np.asarray(model.weights, dtype=float)) > 1e-10))
        except Exception:
            return 1.0
    values = state.as_array()
    return float(max(1, values.shape[0]))


__all__ = [
    "RollingOriginForecastingProblem",
    "TimeSeriesForecastingProblem",
]
