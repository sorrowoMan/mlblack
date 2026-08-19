from __future__ import annotations

import numpy as np
import pytest

from mlblack.catalog.registry import get_catalog
from mlblack.catalog.relations import usage_profile
from mlblack.core.types import UnknownState
from mlblack.models import (
    ARIMASARIMAXForecastModel,
    ARIMASARIMAXProvider,
    ARIMASARIMAXSpec,
    LinearAutoregressiveFitSpec,
    LinearAutoregressiveForecastModel,
    NaiveForecastModel,
)
from mlblack.pipeline.data_views import TimeSeriesDataView
from mlblack.pipeline.time_series import SeasonalDecompositionConfig, STLSeasonalDecompositionComponent, TimeSeriesWindowConfig, TimeSeriesWindowingComponent, seasonal_decompose
from mlblack.problems import RollingOriginForecastingProblem, TimeSeriesForecastingProblem
from mlblack.representations import BaselineForecastRepresentation, BaselineForecastSearchConfig
from mlblack.presets.time_series import (
    build_arima_sarimax_forecast_trainer,
    build_linear_autoregressive_forecast_trainer,
)


def test_time_series_windowing_builds_lagged_numeric_view() -> None:
    data = TimeSeriesDataView.from_values(np.arange(10, dtype=float))
    component = TimeSeriesWindowingComponent(TimeSeriesWindowConfig(lags=(1, 2), horizon=1, valid_size=2))

    numeric, state = component.fit_transform(data)

    assert numeric.X_train.shape == (6, 2)
    assert numeric.X_valid.shape == (2, 2)
    assert numeric.effective_feature_names == ("lag_1", "lag_2")
    assert numeric.y_train[:3].tolist() == [2.0, 3.0, 4.0]
    assert state["n_features"] == 2


def test_stl_like_seasonal_decomposition_returns_residual_view() -> None:
    seasonal = np.asarray([2.0, 0.0, -2.0, 0.0] * 8, dtype=float)
    trend = np.linspace(0.0, 3.1, seasonal.shape[0])
    data = TimeSeriesDataView.from_values(trend + seasonal, frequency="D")
    component = STLSeasonalDecompositionComponent(SeasonalDecompositionConfig(period=4, trend_window=5))

    transformed, state = component.fit_transform(data)
    result = seasonal_decompose(data, SeasonalDecompositionConfig(period=4, trend_window=5))

    assert isinstance(transformed, TimeSeriesDataView)
    assert transformed.n_obs == data.n_obs
    assert transformed.metadata["time_series.decomposition_component"] == "resid"
    assert result.seasonal.shape == result.original.shape
    assert state["period"] == 4
    assert np.std(result.resid) < np.std(result.original)


def test_baseline_forecast_representation_decodes_forecast_model() -> None:
    representation = BaselineForecastRepresentation(
        BaselineForecastSearchConfig(
            strategies=("naive", "seasonal_naive", "moving_average"),
            window_bounds=(2, 5),
            seasonal_period_bounds=(2, 12),
        )
    )
    state = UnknownState(values=np.asarray([2.0, 4.0, 7.0], dtype=float))

    model = representation.decode(state)
    forecast = model.forecast([1.0, 2.0, 3.0, 4.0], 2)

    assert isinstance(model, NaiveForecastModel)
    assert model.strategy == "moving_average"
    assert forecast.shape == (2,)
    assert np.all(np.isfinite(forecast))


def test_linear_autoregressive_forecast_problem_beats_naive_on_trend() -> None:
    series = np.arange(40, dtype=float)
    data = TimeSeriesDataView.from_values(series)
    config = TimeSeriesWindowConfig(lags=(1,), horizon=1, valid_size=8)
    linear = LinearAutoregressiveForecastModel.fit(data, config)
    naive = NaiveForecastModel(strategy="naive")
    problem = TimeSeriesForecastingProblem(data, validation_size=8, objective_metrics=("valid.rmse", "valid.mae"))
    state = UnknownState(values=np.zeros(0, dtype=float))

    linear_feedback = problem.evaluate(linear, state, {})
    naive_feedback = problem.evaluate(naive, state, {})

    assert linear_feedback.metrics["valid.rmse"] < 1e-8
    assert linear_feedback.metrics["valid.rmse"] < naive_feedback.metrics["valid.rmse"]
    assert linear_feedback.objectives.shape == (2,)


def test_arima_sarimax_provider_fits_difference_forecaster() -> None:
    data = TimeSeriesDataView.from_values(np.arange(40, dtype=float))
    provider = ARIMASARIMAXProvider(ARIMASARIMAXSpec(order=(1, 1, 0), trend="c"))
    model = provider.fit(data)

    forecast = model.forecast(np.arange(40, dtype=float), 3)

    assert forecast.shape == (3,)
    assert np.allclose(forecast, np.asarray([40.0, 41.0, 42.0]), atol=1e-4)
    assert model.describe()["model_type"] == "arima_sarimax_forecast"


def test_arima_sarimax_provider_can_use_statsmodels_route_if_installed() -> None:
    pytest.importorskip("statsmodels")
    data = TimeSeriesDataView.from_values(np.arange(30, dtype=float))
    provider = ARIMASARIMAXProvider(ARIMASARIMAXSpec(order=(1, 1, 0), trend="c", backend="statsmodels"))

    model = provider.fit(data)
    forecast = model.forecast(np.arange(30, dtype=float), 2)

    assert forecast.shape == (2,)
    assert np.all(np.isfinite(forecast))
    assert model.describe()["model_type"] == "statsmodels_sarimax_forecast"


def test_rolling_origin_forecasting_problem_reports_horizon_metrics() -> None:
    data = TimeSeriesDataView.from_values(np.arange(30, dtype=float))
    config = TimeSeriesWindowConfig(lags=(1,), horizon=1, valid_size=5)
    model = LinearAutoregressiveForecastModel.fit(data, config)
    problem = RollingOriginForecastingProblem(data, min_train_size=10, horizon=2, max_origins=6)
    feedback = problem.evaluate(model, UnknownState(values=np.zeros(0, dtype=float)), {})

    assert feedback.metrics["rolling.origins"] == 6
    assert feedback.metrics["rolling.horizon"] == 2
    assert feedback.metrics["rolling.rmse"] < 1e-8


def test_time_series_components_are_catalog_visible() -> None:
    catalog = get_catalog(refresh=True)
    keys = {entry.key for entry in catalog.list()}

    assert "data_view.time_series" in keys
    assert "pipeline.stl_seasonal_decomposition" in keys
    assert "pipeline.time_series_windowing" in keys
    assert "provider.arimasarimax" in keys
    assert "model.linear_autoregressive_forecast" in keys
    assert "problem.time_series_forecasting" in keys
    assert "representation.baseline_forecast" in keys

    usage = usage_profile(catalog.show("problem.time_series_forecasting"))
    assert "TimeSeriesDataView" in "\n".join(usage["minimal_wiring"])
    assert "forecast" in "\n".join(usage["use_when"]).lower()


def test_forecast_presets_defer_fitting_to_problem_lifecycle(monkeypatch) -> None:
    data = TimeSeriesDataView.from_values(np.arange(20, dtype=float))

    def fail_linear_fit(*args, **kwargs):
        del args, kwargs
        raise AssertionError("builder must not fit a linear forecast model")

    def fail_arima_fit(*args, **kwargs):
        del args, kwargs
        raise AssertionError("builder must not call the ARIMA provider")

    monkeypatch.setattr(LinearAutoregressiveForecastModel, "fit", fail_linear_fit)
    monkeypatch.setattr(ARIMASARIMAXProvider, "fit", fail_arima_fit)

    linear = build_linear_autoregressive_forecast_trainer(data)
    arima = build_arima_sarimax_forecast_trainer(data)

    assert isinstance(
        linear.model_representation.decode(linear.model_representation.init({}), {}),
        LinearAutoregressiveFitSpec,
    )
    assert isinstance(
        arima.model_representation.decode(arima.model_representation.init({}), {}),
        ARIMASARIMAXSpec,
    )


def test_forecast_fit_specs_materialize_inside_learning_solver() -> None:
    data = TimeSeriesDataView.from_values(np.arange(20, dtype=float))

    linear_result = build_linear_autoregressive_forecast_trainer(data).fit(max_steps=1)
    arima_result = build_arima_sarimax_forecast_trainer(data, order=(1, 1, 0)).fit(max_steps=1)

    assert isinstance(linear_result.best_model, LinearAutoregressiveForecastModel)
    assert isinstance(arima_result.best_model, ARIMASARIMAXForecastModel)
    assert linear_result.best_feedback is not None
    assert arima_result.best_feedback is not None
