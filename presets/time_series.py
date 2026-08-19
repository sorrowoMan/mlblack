from __future__ import annotations

from typing import Any, Sequence

from mlblack.models import ARIMASARIMAXSpec, LinearAutoregressiveFitSpec
from mlblack.pipeline.data_views import TimeSeriesDataView
from mlblack.pipeline.time_series import TimeSeriesWindowConfig
from mlblack.problems import RollingOriginForecastingProblem, TimeSeriesForecastingProblem
from mlblack.representations import (
    BaselineForecastRepresentation,
    BaselineForecastSearchConfig,
    ForecastModelSpecRepresentation,
)


def build_baseline_forecast_search_trainer(
    data: TimeSeriesDataView,
    *,
    strategies: Sequence[str] = ("naive", "seasonal_naive", "moving_average"),
    window_bounds: tuple[int, int] = (2, 12),
    seasonal_period_bounds: tuple[int, int] = (1, 24),
    population_size: int = 8,
    mutation_scale: float = 0.25,
    validation_size: int | float = 0.2,
    objective_metrics: Sequence[str] | None = None,
    seasonal_period: int = 1,
    rolling_origin: bool = False,
    rolling_horizon: int = 1,
    rolling_min_train_size: int | float = 0.6,
    run_name: str = "baseline_forecast_search",
    **problem_kwargs: Any,
) -> Any:
    config = BaselineForecastSearchConfig(
        strategies=tuple(str(s) for s in strategies),
        window_bounds=tuple(int(v) for v in window_bounds),
        seasonal_period_bounds=tuple(int(v) for v in seasonal_period_bounds),
    )
    representation = BaselineForecastRepresentation(config)
    if rolling_origin:
        metrics = tuple(objective_metrics) if objective_metrics else ("rolling.rmse", "rolling.mae")
        problem = RollingOriginForecastingProblem(
            data,
            min_train_size=rolling_min_train_size,
            horizon=int(rolling_horizon),
            objective_metrics=metrics,
            seasonal_period=seasonal_period,
            **problem_kwargs,
        )
    else:
        metrics = tuple(objective_metrics) if objective_metrics else ("valid.rmse", "valid.mae")
        problem = TimeSeriesForecastingProblem(
            data,
            validation_size=validation_size,
            objective_metrics=metrics,
            seasonal_period=seasonal_period,
            **problem_kwargs,
        )
    adapter = _build_optimization_adapter(
        "search.random_gaussian",
        population_size=int(population_size),
        mutation_scale=float(mutation_scale),
    )
    return _build_learning_solver(problem=problem, representation=representation, adapter=adapter, run_name=run_name)


def build_linear_autoregressive_forecast_trainer(
    data: TimeSeriesDataView,
    *,
    lags: Sequence[int] = (1, 2, 3),
    horizon: int = 1,
    ridge: float = 0.0,
    include_exogenous: bool = True,
    validation_size: int | float = 0.2,
    objective_metrics: Sequence[str] | None = None,
    seasonal_period: int = 1,
    rolling_origin: bool = False,
    rolling_horizon: int = 1,
    rolling_min_train_size: int | float = 0.6,
    run_name: str = "linear_autoregressive_forecast",
    **problem_kwargs: Any,
) -> Any:
    window_cfg = TimeSeriesWindowConfig(
        lags=tuple(int(lag) for lag in lags),
        horizon=int(horizon),
        include_exogenous=bool(include_exogenous),
    )
    spec = LinearAutoregressiveFitSpec(
        window_config=window_cfg,
        ridge=float(ridge),
        metadata={"source": "linear_autoregressive_preset"},
    )
    representation = ForecastModelSpecRepresentation(
        spec,
        metadata={"materialization_owner": "TimeSeriesForecastingProblem"},
    )
    if rolling_origin:
        metrics = tuple(objective_metrics) if objective_metrics else ("rolling.rmse", "rolling.mae")
        problem = RollingOriginForecastingProblem(
            data,
            min_train_size=rolling_min_train_size,
            horizon=int(rolling_horizon),
            objective_metrics=metrics,
            seasonal_period=seasonal_period,
            **problem_kwargs,
        )
    else:
        metrics = tuple(objective_metrics) if objective_metrics else ("valid.rmse", "valid.mae")
        problem = TimeSeriesForecastingProblem(
            data,
            validation_size=validation_size,
            objective_metrics=metrics,
            seasonal_period=seasonal_period,
            **problem_kwargs,
        )
    adapter = _build_optimization_adapter("evaluation.fixed")
    return _build_learning_solver(problem=problem, representation=representation, adapter=adapter, run_name=run_name)


def build_arima_sarimax_forecast_trainer(
    data: TimeSeriesDataView,
    *,
    order: tuple[int, int, int] = (1, 0, 0),
    seasonal_order: tuple[int, int, int, int] = (0, 0, 0, 0),
    trend: str = "c",
    backend: str = "numpy_fallback",
    validation_size: int | float = 0.2,
    objective_metrics: Sequence[str] | None = None,
    seasonal_period: int = 1,
    rolling_origin: bool = False,
    rolling_horizon: int = 1,
    rolling_min_train_size: int | float = 0.6,
    run_name: str = "arima_sarimax_forecast",
    **problem_kwargs: Any,
) -> Any:
    spec = ARIMASARIMAXSpec(
        order=tuple(int(v) for v in order),
        seasonal_order=tuple(int(v) for v in seasonal_order),
        trend=str(trend),
        backend=str(backend),
    )
    representation = ForecastModelSpecRepresentation(
        spec,
        metadata={"materialization_owner": "TimeSeriesForecastingProblem"},
    )
    if rolling_origin:
        metrics = tuple(objective_metrics) if objective_metrics else ("rolling.rmse", "rolling.mae")
        problem = RollingOriginForecastingProblem(
            data,
            min_train_size=rolling_min_train_size,
            horizon=int(rolling_horizon),
            objective_metrics=metrics,
            seasonal_period=seasonal_period,
            **problem_kwargs,
        )
    else:
        metrics = tuple(objective_metrics) if objective_metrics else ("valid.rmse", "valid.mae")
        problem = TimeSeriesForecastingProblem(
            data,
            validation_size=validation_size,
            objective_metrics=metrics,
            seasonal_period=seasonal_period,
            **problem_kwargs,
        )
    adapter = _build_optimization_adapter("evaluation.fixed")
    return _build_learning_solver(problem=problem, representation=representation, adapter=adapter, run_name=run_name)


__all__ = [
    "build_arima_sarimax_forecast_trainer",
    "build_baseline_forecast_search_trainer",
    "build_linear_autoregressive_forecast_trainer",
]


def _build_optimization_adapter(method: str, **kwargs):
    from mlblack.integrations.nsgablack_optimization import build_optimization_adapter

    return build_optimization_adapter(method, **kwargs)


def _build_learning_solver(**kwargs):
    from mlblack.integrations.nsgablack_control import build_learning_solver

    return build_learning_solver(**kwargs)
