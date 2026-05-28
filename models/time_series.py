from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from mlblack.pipeline.data_views import TimeSeriesDataView
from mlblack.pipeline.time_series import TimeSeriesWindowConfig, build_lagged_numeric_view


@dataclass(frozen=True)
class ARIMASARIMAXSpec:
    """Declarative ARIMA/SARIMAX order and fitting options."""

    order: tuple[int, int, int] = (1, 0, 0)
    seasonal_order: tuple[int, int, int, int] = (0, 0, 0, 0)
    trend: str = "c"
    include_exogenous: bool = True
    ridge: float = 1e-8
    backend: str = "numpy_fallback"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def normalized_order(self) -> tuple[int, int, int]:
        values = tuple(int(v) for v in self.order)
        if len(values) != 3:
            raise ValueError("ARIMASARIMAXSpec.order must be (p, d, q)")
        if any(v < 0 for v in values):
            raise ValueError("ARIMASARIMAXSpec.order values must be non-negative")
        return values

    def normalized_seasonal_order(self) -> tuple[int, int, int, int]:
        values = tuple(int(v) for v in self.seasonal_order)
        if len(values) != 4:
            raise ValueError("ARIMASARIMAXSpec.seasonal_order must be (P, D, Q, m)")
        if any(v < 0 for v in values):
            raise ValueError("ARIMASARIMAXSpec.seasonal_order values must be non-negative")
        return values

    def describe(self) -> dict[str, Any]:
        return {
            "order": self.normalized_order(),
            "seasonal_order": self.normalized_seasonal_order(),
            "trend": str(self.trend),
            "include_exogenous": bool(self.include_exogenous),
            "ridge": float(self.ridge),
            "backend": str(self.backend),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ARIMASARIMAXForecastModel:
    """Fitted ARIMA/SARIMAX-style forecaster with a numpy ARX fallback route."""

    spec: ARIMASARIMAXSpec
    intercept: float
    ar_params: np.ndarray
    seasonal_ar_params: np.ndarray
    exog_params: np.ndarray
    training_history: np.ndarray
    differenced_history: np.ndarray
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "ar_params", np.asarray(self.ar_params, dtype=float).reshape(-1))
        object.__setattr__(self, "seasonal_ar_params", np.asarray(self.seasonal_ar_params, dtype=float).reshape(-1))
        object.__setattr__(self, "exog_params", np.asarray(self.exog_params, dtype=float).reshape(-1))
        object.__setattr__(self, "training_history", np.asarray(self.training_history, dtype=float).reshape(-1))
        object.__setattr__(self, "differenced_history", np.asarray(self.differenced_history, dtype=float).reshape(-1))

    def forecast(
        self,
        history: Sequence[float] | np.ndarray,
        horizon: int,
        *,
        exogenous_future: np.ndarray | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> np.ndarray:
        _ = context
        hist = [float(value) for value in np.asarray(history, dtype=float).reshape(-1)]
        if len(hist) < 2:
            raise ValueError("ARIMA forecast requires at least two history observations")
        steps = int(horizon)
        if steps <= 0:
            raise ValueError("horizon must be positive")
        p, d, _q = self.spec.normalized_order()
        seasonal_p, seasonal_d, _seasonal_q, period = self.spec.normalized_seasonal_order()
        z_hist = list(_difference_for_supported_orders(np.asarray(hist, dtype=float), d=d, seasonal_d=seasonal_d, period=period))
        if not z_hist:
            z_hist = [0.0]
        exog = None if exogenous_future is None else np.asarray(exogenous_future, dtype=float)
        if exog is not None and exog.ndim == 1:
            exog = exog.reshape(-1, 1)
        if self.exog_params.size and (exog is None or exog.shape[0] < steps):
            raise ValueError("exogenous_future must have at least horizon rows for SARIMAX exogenous coefficients")

        out: list[float] = []
        for step in range(steps):
            z_pred = float(self.intercept)
            for lag_idx in range(p):
                lag = lag_idx + 1
                if len(z_hist) >= lag:
                    z_pred += float(self.ar_params[lag_idx]) * float(z_hist[-lag])
            for lag_idx in range(seasonal_p):
                lag = max(1, int(period)) * (lag_idx + 1)
                if len(z_hist) >= lag:
                    z_pred += float(self.seasonal_ar_params[lag_idx]) * float(z_hist[-lag])
            if self.exog_params.size and exog is not None:
                z_pred += float(np.asarray(exog[step, :], dtype=float).reshape(-1) @ self.exog_params)
            y_pred = _invert_next_difference(hist, z_pred, d=d, seasonal_d=seasonal_d, period=period)
            out.append(float(y_pred))
            hist.append(float(y_pred))
            z_hist.append(float(z_pred))
        return np.asarray(out, dtype=float)

    def describe(self) -> dict[str, Any]:
        return {
            "model_type": "arima_sarimax_forecast",
            "spec": self.spec.describe(),
            "intercept": float(self.intercept),
            "ar_params": self.ar_params.tolist(),
            "seasonal_ar_params": self.seasonal_ar_params.tolist(),
            "exog_params": self.exog_params.tolist(),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class StatsmodelsSARIMAXForecastModel:
    """Forecast wrapper around a fitted statsmodels SARIMAXResults object."""

    spec: ARIMASARIMAXSpec
    result: Any
    training_history: np.ndarray
    training_exogenous: np.ndarray | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "training_history", np.asarray(self.training_history, dtype=float).reshape(-1))
        if self.training_exogenous is not None:
            exog = np.asarray(self.training_exogenous, dtype=float)
            if exog.ndim == 1:
                exog = exog.reshape(-1, 1)
            object.__setattr__(self, "training_exogenous", exog)

    def forecast(
        self,
        history: Sequence[float] | np.ndarray,
        horizon: int,
        *,
        exogenous_future: np.ndarray | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> np.ndarray:
        _ = context
        hist = np.asarray(history, dtype=float).reshape(-1)
        steps = int(horizon)
        if steps <= 0:
            raise ValueError("horizon must be positive")
        exog_future = _normalize_exog_future(exogenous_future, steps=steps)
        if _same_history(hist, self.training_history):
            return np.asarray(self.result.forecast(steps=steps, exog=exog_future), dtype=float).reshape(-1)

        # Rolling-origin evaluation may pass a shorter history than the provider
        # was initially fit on. Refit behind the provider/model boundary so the
        # Problem still only sees forecast(history, horizon).
        exog_train = _matching_history_exog(
            hist,
            full_history=self.training_history,
            full_exog=self.training_exogenous,
        )
        refit = _fit_statsmodels_sarimax_result(hist, exog_train, self.spec)
        return np.asarray(refit.forecast(steps=steps, exog=exog_future), dtype=float).reshape(-1)

    def describe(self) -> dict[str, Any]:
        return {
            "model_type": "statsmodels_sarimax_forecast",
            "spec": self.spec.describe(),
            "statsmodels_class": type(self.result).__name__,
            "aic": _safe_float(getattr(self.result, "aic", None)),
            "bic": _safe_float(getattr(self.result, "bic", None)),
            "metadata": dict(self.metadata),
        }


class ARIMASARIMAXProvider:
    """Provider that fits ARIMA/SARIMAX specs through numpy or statsmodels.

    The default route is a dependency-free numpy AR/SARX approximation.
    `backend="statsmodels"` delegates fitting to statsmodels while preserving
    the same model/problem boundary.
    """

    name = "arima_sarimax_provider"

    def __init__(self, spec: ARIMASARIMAXSpec | Mapping[str, Any] | None = None) -> None:
        self.spec = spec if isinstance(spec, ARIMASARIMAXSpec) else ARIMASARIMAXSpec(**dict(spec or {}))

    def fit(
        self,
        data: TimeSeriesDataView,
        spec: ARIMASARIMAXSpec | Mapping[str, Any] | None = None,
        *,
        context: Mapping[str, Any] | None = None,
    ) -> ARIMASARIMAXForecastModel | StatsmodelsSARIMAXForecastModel:
        _ = context
        cfg = spec if isinstance(spec, ARIMASARIMAXSpec) else (self.spec if spec is None else ARIMASARIMAXSpec(**dict(spec)))
        if _backend_key(cfg.backend) in {"statsmodels", "statsmodels_sarimax", "sarimax"}:
            return _fit_arima_sarimax_statsmodels(data, cfg)
        if _backend_key(cfg.backend) not in {"numpy", "numpy_fallback", "fallback", "arx"}:
            raise ValueError(f"unknown ARIMA/SARIMAX backend: {cfg.backend}")
        return _fit_arima_sarimax_numpy(data, cfg)

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "spec": self.spec.describe(), "route": _backend_key(self.spec.backend)}


@dataclass(frozen=True)
class ForecastResult:
    """Forecast payload with explicit horizon and optional time index."""

    values: np.ndarray
    horizon: int
    time_index: Sequence[Any] = tuple()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=float).reshape(-1)
        if values.shape[0] != int(self.horizon):
            raise ValueError("forecast values length must match horizon")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "time_index", tuple(self.time_index))

    def as_dict(self) -> dict[str, Any]:
        return {
            "values": np.asarray(self.values, dtype=float).reshape(-1).tolist(),
            "horizon": int(self.horizon),
            "time_index": tuple(self.time_index),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class NaiveForecastModel:
    """Naive, seasonal-naive, or moving-average univariate forecaster."""

    strategy: str = "naive"
    window: int = 3
    seasonal_period: int = 1
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def forecast(
        self,
        history: Sequence[float] | np.ndarray,
        horizon: int,
        *,
        exogenous_future: np.ndarray | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> np.ndarray:
        _ = exogenous_future
        _ = context
        hist = [float(value) for value in np.asarray(history, dtype=float).reshape(-1)]
        steps = int(horizon)
        if steps <= 0:
            raise ValueError("horizon must be positive")
        if not hist:
            raise ValueError("forecast history is empty")
        out: list[float] = []
        key = str(self.strategy or "naive").strip().lower()
        for _step in range(steps):
            if key in {"seasonal_naive", "seasonal", "snaive"} and int(self.seasonal_period) > 0 and len(hist) >= int(self.seasonal_period):
                pred = float(hist[-int(self.seasonal_period)])
            elif key in {"moving_average", "mean", "ma"}:
                window = max(1, min(int(self.window), len(hist)))
                pred = float(np.mean(np.asarray(hist[-window:], dtype=float)))
            else:
                pred = float(hist[-1])
            out.append(pred)
            hist.append(pred)
        return np.asarray(out, dtype=float)

    def predict(self, X: np.ndarray) -> np.ndarray:
        arr = np.asarray(X, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.ndim != 2 or arr.shape[1] <= 0:
            raise ValueError("X must be a 2D lag feature matrix")
        key = str(self.strategy or "naive").strip().lower()
        if key in {"seasonal_naive", "seasonal", "snaive"}:
            idx = min(max(0, int(self.seasonal_period) - 1), arr.shape[1] - 1)
            return arr[:, idx].reshape(-1)
        if key in {"moving_average", "mean", "ma"}:
            width = max(1, min(int(self.window), arr.shape[1]))
            return np.mean(arr[:, :width], axis=1).reshape(-1)
        return arr[:, 0].reshape(-1)

    def describe(self) -> dict[str, Any]:
        return {
            "model_type": "naive_forecast",
            "strategy": str(self.strategy),
            "window": int(self.window),
            "seasonal_period": int(self.seasonal_period),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class LinearAutoregressiveForecastModel:
    """Linear AR/ARX-style forecaster fitted on lagged windows."""

    intercept: float
    weights: np.ndarray
    window_config: TimeSeriesWindowConfig
    feature_names: Sequence[str] = tuple()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "weights", np.asarray(self.weights, dtype=float).reshape(-1))
        object.__setattr__(self, "feature_names", tuple(str(name) for name in self.feature_names))

    @classmethod
    def fit(
        cls,
        data: TimeSeriesDataView,
        config: TimeSeriesWindowConfig | Mapping[str, Any] | None = None,
        *,
        ridge: float = 0.0,
        metadata: Mapping[str, Any] | None = None,
    ) -> "LinearAutoregressiveForecastModel":
        cfg = config if isinstance(config, TimeSeriesWindowConfig) else TimeSeriesWindowConfig(**dict(config or {}))
        numeric = build_lagged_numeric_view(data, cfg)
        X = np.asarray(numeric.X_train, dtype=float)
        y = np.asarray(numeric.y_train, dtype=float).reshape(-1)
        design = np.column_stack([np.ones(X.shape[0], dtype=float), X])
        penalty = float(ridge) * np.eye(design.shape[1], dtype=float)
        penalty[0, 0] = 0.0
        if float(ridge) > 0.0:
            coef = np.linalg.solve(design.T @ design + penalty, design.T @ y)
        else:
            coef, *_ = np.linalg.lstsq(design, y, rcond=None)
        return cls(
            intercept=float(coef[0]),
            weights=np.asarray(coef[1:], dtype=float),
            window_config=cfg,
            feature_names=numeric.effective_feature_names,
            metadata={
                **dict(metadata or {}),
                "fit": "least_squares",
                "ridge": float(ridge),
                "n_train_windows": int(X.shape[0]),
                "source_series_length": int(data.n_obs),
            },
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        arr = np.asarray(X, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.ndim != 2:
            raise ValueError("X must be 2D")
        if arr.shape[1] != self.weights.shape[0]:
            raise ValueError(f"X has {arr.shape[1]} features but model expects {self.weights.shape[0]}")
        return np.asarray(float(self.intercept) + arr @ self.weights, dtype=float).reshape(-1)

    def forecast(
        self,
        history: Sequence[float] | np.ndarray,
        horizon: int,
        *,
        exogenous_future: np.ndarray | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> np.ndarray:
        _ = context
        return _recursive_lag_forecast(
            predictor=self.predict,
            history=history,
            horizon=horizon,
            config=self.window_config,
            exogenous_future=exogenous_future,
            n_features=int(self.weights.shape[0]),
        )

    def describe(self) -> dict[str, Any]:
        return {
            "model_type": "linear_autoregressive_forecast",
            "intercept": float(self.intercept),
            "weights": np.asarray(self.weights, dtype=float).reshape(-1).tolist(),
            "feature_names": tuple(self.feature_names),
            "window_config": self.window_config.describe(),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class LagEstimatorForecastModel:
    """Forecast wrapper around an already-fitted lag-feature estimator."""

    estimator: Any
    window_config: TimeSeriesWindowConfig
    feature_names: Sequence[str] = tuple()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def fit(
        cls,
        estimator: Any,
        data: TimeSeriesDataView,
        config: TimeSeriesWindowConfig | Mapping[str, Any] | None = None,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> "LagEstimatorForecastModel":
        cfg = config if isinstance(config, TimeSeriesWindowConfig) else TimeSeriesWindowConfig(**dict(config or {}))
        numeric = build_lagged_numeric_view(data, cfg)
        if not hasattr(estimator, "fit") or not callable(getattr(estimator, "fit")):
            raise TypeError("estimator must expose fit(X, y)")
        estimator.fit(numeric.X_train, numeric.y_train)
        return cls(
            estimator=estimator,
            window_config=cfg,
            feature_names=numeric.effective_feature_names,
            metadata={**dict(metadata or {}), "estimator_type": type(estimator).__name__},
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self.estimator, "predict") or not callable(getattr(self.estimator, "predict")):
            raise TypeError("estimator must expose predict(X)")
        return np.asarray(self.estimator.predict(np.asarray(X, dtype=float)), dtype=float).reshape(-1)

    def forecast(
        self,
        history: Sequence[float] | np.ndarray,
        horizon: int,
        *,
        exogenous_future: np.ndarray | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> np.ndarray:
        _ = context
        return _recursive_lag_forecast(
            predictor=self.predict,
            history=history,
            horizon=horizon,
            config=self.window_config,
            exogenous_future=exogenous_future,
            n_features=len(tuple(self.feature_names)) if self.feature_names else None,
        )

    def describe(self) -> dict[str, Any]:
        return {
            "model_type": "lag_estimator_forecast",
            "estimator_type": type(self.estimator).__name__,
            "feature_names": tuple(self.feature_names),
            "window_config": self.window_config.describe(),
            "metadata": dict(self.metadata),
        }



def _fit_arima_sarimax_statsmodels(data: TimeSeriesDataView, spec: ARIMASARIMAXSpec) -> StatsmodelsSARIMAXForecastModel:
    y = np.asarray(data.y, dtype=float).reshape(-1)
    exog = None
    if bool(spec.include_exogenous) and data.exogenous is not None:
        exog = np.asarray(data.exogenous, dtype=float)
        if exog.ndim == 1:
            exog = exog.reshape(-1, 1)
    result = _fit_statsmodels_sarimax_result(y, exog, spec)
    return StatsmodelsSARIMAXForecastModel(
        spec=spec,
        result=result,
        training_history=y,
        training_exogenous=exog,
        metadata={
            **dict(spec.metadata),
            "fit_route": "statsmodels_sarimax",
            "n_train_obs": int(y.shape[0]),
            "has_exogenous": exog is not None,
        },
    )


def _fit_statsmodels_sarimax_result(y: np.ndarray, exog: np.ndarray | None, spec: ARIMASARIMAXSpec) -> Any:
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
    except Exception as exc:  # pragma: no cover - depends on optional package.
        raise RuntimeError(
            "ARIMASARIMAXSpec.backend='statsmodels' requires optional dependency 'statsmodels'. "
            "Install with `pip install -e .[time_series]` or choose backend='numpy_fallback'."
        ) from exc

    metadata = dict(spec.metadata)
    model_kwargs = dict(metadata.get("statsmodels_model_kwargs", {}) or {})
    fit_kwargs = {
        "disp": False,
        **dict(metadata.get("statsmodels_fit_kwargs", {}) or {}),
    }
    model = SARIMAX(
        np.asarray(y, dtype=float).reshape(-1),
        exog=exog,
        order=spec.normalized_order(),
        seasonal_order=spec.normalized_seasonal_order(),
        trend=None if str(spec.trend).lower() in {"", "none", "n"} else str(spec.trend),
        enforce_stationarity=bool(metadata.get("enforce_stationarity", False)),
        enforce_invertibility=bool(metadata.get("enforce_invertibility", False)),
        **model_kwargs,
    )
    return model.fit(**fit_kwargs)


def _fit_arima_sarimax_numpy(data: TimeSeriesDataView, spec: ARIMASARIMAXSpec) -> ARIMASARIMAXForecastModel:
    p, d, q = spec.normalized_order()
    seasonal_p, seasonal_d, seasonal_q, period = spec.normalized_seasonal_order()
    if d > 1 or seasonal_d > 1:
        raise ValueError("numpy ARIMA fallback currently supports d and D in {0, 1}")
    if q or seasonal_q:
        # MA terms require residual recursion; keep the spec visible but do not
        # silently pretend this fallback is a full maximum-likelihood SARIMAX.
        ignored = {"q": int(q), "seasonal_q": int(seasonal_q)}
    else:
        ignored = {}
    y = np.asarray(data.y, dtype=float).reshape(-1)
    z = _difference_for_supported_orders(y, d=d, seasonal_d=seasonal_d, period=period)
    max_lag = max(int(p), int(seasonal_p) * max(1, int(period)), 0)
    if z.shape[0] <= max_lag:
        raise ValueError("series is too short for requested ARIMA/SARIMAX orders")
    exog = None
    if bool(spec.include_exogenous) and data.exogenous is not None:
        exog_full = np.asarray(data.exogenous, dtype=float)
        offset = y.shape[0] - z.shape[0]
        exog = exog_full[offset:, :]

    rows: list[list[float]] = []
    target: list[float] = []
    for idx in range(max_lag, z.shape[0]):
        row: list[float] = []
        if str(spec.trend).lower() in {"c", "constant", "intercept"}:
            row.append(1.0)
        for lag in range(1, p + 1):
            row.append(float(z[idx - lag]))
        for lag_idx in range(1, seasonal_p + 1):
            lag = max(1, int(period)) * lag_idx
            row.append(float(z[idx - lag]))
        if exog is not None:
            row.extend(float(v) for v in exog[idx, :])
        rows.append(row)
        target.append(float(z[idx]))
    X = np.asarray(rows, dtype=float)
    target_arr = np.asarray(target, dtype=float)
    if X.shape[1] == 0:
        X = np.zeros((target_arr.shape[0], 0), dtype=float)
        coef = np.zeros(0, dtype=float)
    else:
        penalty = float(spec.ridge) * np.eye(X.shape[1], dtype=float)
        coef = np.linalg.solve(X.T @ X + penalty, X.T @ target_arr)

    offset = 0
    intercept = 0.0
    if str(spec.trend).lower() in {"c", "constant", "intercept"}:
        intercept = float(coef[0])
        offset = 1
    ar_params = coef[offset : offset + p]
    offset += p
    seasonal_ar_params = coef[offset : offset + seasonal_p]
    offset += seasonal_p
    exog_params = coef[offset:]
    return ARIMASARIMAXForecastModel(
        spec=spec,
        intercept=intercept,
        ar_params=ar_params,
        seasonal_ar_params=seasonal_ar_params,
        exog_params=exog_params,
        training_history=y,
        differenced_history=z,
        metadata={
            **dict(spec.metadata),
            "fit_route": "numpy_least_squares_arx",
            "ignored_ma_terms": ignored,
            "n_train_rows": int(X.shape[0]),
            "n_design_features": int(X.shape[1]),
        },
    )


def _backend_key(value: str) -> str:
    return str(value or "numpy_fallback").strip().lower().replace("-", "_")


def _normalize_exog_future(value: np.ndarray | None, *, steps: int) -> np.ndarray | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.shape[0] < int(steps):
        raise ValueError("exogenous_future must have at least horizon rows")
    return arr[: int(steps), :]


def _same_history(left: np.ndarray, right: np.ndarray) -> bool:
    a = np.asarray(left, dtype=float).reshape(-1)
    b = np.asarray(right, dtype=float).reshape(-1)
    return a.shape == b.shape and bool(np.allclose(a, b, rtol=0.0, atol=1e-12))


def _matching_history_exog(
    history: np.ndarray,
    *,
    full_history: np.ndarray,
    full_exog: np.ndarray | None,
) -> np.ndarray | None:
    if full_exog is None:
        return None
    hist = np.asarray(history, dtype=float).reshape(-1)
    full = np.asarray(full_history, dtype=float).reshape(-1)
    exog = np.asarray(full_exog, dtype=float)
    if hist.shape[0] <= full.shape[0] and np.allclose(hist, full[: hist.shape[0]], rtol=0.0, atol=1e-12):
        return exog[: hist.shape[0], :]
    if hist.shape[0] == full.shape[0]:
        return exog
    raise ValueError("statsmodels SARIMAX refit requires exogenous training rows aligned to the provided history")


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except Exception:
        return None
    return out if np.isfinite(out) else None


def _difference_for_supported_orders(y: np.ndarray, *, d: int, seasonal_d: int, period: int) -> np.ndarray:
    arr = np.asarray(y, dtype=float).reshape(-1)
    if int(d) not in {0, 1} or int(seasonal_d) not in {0, 1}:
        raise ValueError("only d and D in {0, 1} are supported by the numpy fallback")
    if int(d) == 1:
        arr = np.diff(arr, n=1)
    if int(seasonal_d) == 1:
        lag = max(1, int(period))
        if arr.shape[0] <= lag:
            raise ValueError("series is too short for seasonal differencing")
        arr = arr[lag:] - arr[:-lag]
    return arr.reshape(-1)


def _invert_next_difference(history: list[float], z_pred: float, *, d: int, seasonal_d: int, period: int) -> float:
    if int(d) == 0 and int(seasonal_d) == 0:
        return float(z_pred)
    if int(d) == 1 and int(seasonal_d) == 0:
        return float(history[-1] + z_pred)
    lag = max(1, int(period))
    if len(history) <= lag:
        raise ValueError("history is too short for seasonal inverse differencing")
    if int(d) == 0 and int(seasonal_d) == 1:
        return float(history[-lag] + z_pred)
    if int(d) == 1 and int(seasonal_d) == 1:
        if len(history) <= lag + 1:
            raise ValueError("history is too short for combined inverse differencing")
        return float(z_pred + history[-1] + history[-lag] - history[-lag - 1])
    raise ValueError("only d and D in {0, 1} are supported by the numpy fallback")


def _recursive_lag_forecast(
    *,
    predictor: Any,
    history: Sequence[float] | np.ndarray,
    horizon: int,
    config: TimeSeriesWindowConfig,
    exogenous_future: np.ndarray | None,
    n_features: int | None,
) -> np.ndarray:
    hist = [float(value) for value in np.asarray(history, dtype=float).reshape(-1)]
    steps = int(horizon)
    if steps <= 0:
        raise ValueError("horizon must be positive")
    lags = config.normalized_lags()
    if len(hist) < max(lags):
        raise ValueError("history is shorter than required lags")
    exog = None if exogenous_future is None else np.asarray(exogenous_future, dtype=float)
    if exog is not None and exog.ndim == 1:
        exog = exog.reshape(-1, 1)
    if bool(config.include_exogenous) and exog is not None and exog.shape[0] < steps:
        raise ValueError("exogenous_future must have at least horizon rows")
    out: list[float] = []
    for step in range(steps):
        row = [float(hist[-lag]) for lag in lags]
        if bool(config.include_exogenous):
            if exog is not None:
                row.extend(float(value) for value in exog[step, :])
            elif n_features is not None and n_features > len(row) + int(bool(config.include_origin_index)):
                raise ValueError("model expects exogenous features but exogenous_future was not provided")
        if bool(config.include_origin_index):
            denom = max(1, len(hist) + steps - 1)
            row.append(float(len(hist) - 1) / float(denom))
        arr = np.asarray(row, dtype=float).reshape(1, -1)
        pred = float(np.asarray(predictor(arr), dtype=float).reshape(-1)[0])
        out.append(pred)
        hist.append(pred)
    return np.asarray(out, dtype=float)


__all__ = [
    "ARIMASARIMAXForecastModel",
    "ARIMASARIMAXProvider",
    "ARIMASARIMAXSpec",
    "ForecastResult",
    "LagEstimatorForecastModel",
    "LinearAutoregressiveForecastModel",
    "NaiveForecastModel",
    "StatsmodelsSARIMAXForecastModel",
]
