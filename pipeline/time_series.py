from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from mlblack.core.contracts import ComponentContract, ContractMixin
from mlblack.pipeline.data_views import NumericDataView, TimeSeriesDataView


@dataclass(frozen=True)
class SeasonalDecompositionConfig:
    """STL-like additive seasonal decomposition configuration."""

    period: int
    trend_window: int | None = None
    model: str = "additive"
    residual_target: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def normalized_period(self) -> int:
        period = int(self.period)
        if period <= 1:
            raise ValueError("SeasonalDecompositionConfig.period must be greater than 1")
        return period

    def normalized_trend_window(self) -> int:
        period = self.normalized_period()
        window = int(self.trend_window or period)
        window = max(3, window)
        if window % 2 == 0:
            window += 1
        return window

    def describe(self) -> dict[str, Any]:
        return {
            "period": int(self.normalized_period()),
            "trend_window": int(self.normalized_trend_window()),
            "model": str(self.model),
            "residual_target": bool(self.residual_target),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class SeasonalDecompositionResult:
    """Trend, seasonal and residual arrays aligned to the original series."""

    original: np.ndarray
    trend: np.ndarray
    seasonal: np.ndarray
    resid: np.ndarray
    period: int
    model: str = "additive"
    time_index: Sequence[Any] = tuple()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        original = np.asarray(self.original, dtype=float).reshape(-1)
        trend = np.asarray(self.trend, dtype=float).reshape(-1)
        seasonal = np.asarray(self.seasonal, dtype=float).reshape(-1)
        resid = np.asarray(self.resid, dtype=float).reshape(-1)
        if not (original.shape == trend.shape == seasonal.shape == resid.shape):
            raise ValueError("decomposition arrays must have the same shape")
        object.__setattr__(self, "original", original)
        object.__setattr__(self, "trend", trend)
        object.__setattr__(self, "seasonal", seasonal)
        object.__setattr__(self, "resid", resid)
        object.__setattr__(self, "period", int(self.period))
        object.__setattr__(self, "time_index", tuple(self.time_index) or tuple(range(original.shape[0])))

    def component_values(self, component: str) -> np.ndarray:
        key = str(component or "resid").strip().lower()
        if key in {"resid", "residual", "remainder"}:
            return np.asarray(self.resid, dtype=float).reshape(-1)
        if key == "trend":
            return np.asarray(self.trend, dtype=float).reshape(-1)
        if key in {"seasonal", "seasonality"}:
            return np.asarray(self.seasonal, dtype=float).reshape(-1)
        if key in {"original", "observed", "y"}:
            return np.asarray(self.original, dtype=float).reshape(-1)
        raise ValueError(f"unknown decomposition component: {component}")

    def to_data_view(self, *, component: str = "resid", target_name: str | None = None) -> TimeSeriesDataView:
        values = self.component_values(component)
        return TimeSeriesDataView(
            y=values,
            time_index=self.time_index,
            target_name=target_name or str(component),
            frequency=str(self.metadata.get("frequency", "")),
            metadata={
                **dict(self.metadata),
                "time_series.decomposition_component": str(component),
                "time_series.decomposition_period": int(self.period),
                "time_series.decomposition_model": str(self.model),
            },
        )

    def describe(self) -> dict[str, Any]:
        return {
            "name": "seasonal_decomposition_result",
            "period": int(self.period),
            "model": str(self.model),
            "n_obs": int(self.original.shape[0]),
            "resid_std": float(np.std(self.resid)),
            "seasonal_std": float(np.std(self.seasonal)),
            "trend_std": float(np.std(self.trend)),
            "metadata": dict(self.metadata),
        }


class STLSeasonalDecompositionComponent(ContractMixin):
    """STL-like deterministic seasonal decomposition for TimeSeriesDataView."""

    name = "stl_seasonal_decomposition"
    context_requires = ("data.time_series_view",)
    context_optional = ("trainer.context",)
    context_provides = ("time_series.decomposition", "data.time_series_view", "pipeline.component_state")
    context_mutates = ("pipeline.component_state",)
    context_cache = ()
    requires_metrics = ()
    metrics_fallback = "strict"
    context_notes = "Builds trend/seasonal/residual components without owning training orchestration."
    contract = ComponentContract(
        name=name,
        requires=("data.time_series_view",),
        optional=("trainer.context",),
        provides=("time_series.decomposition", "data.time_series_view", "pipeline.component_state"),
        mutates=("pipeline.component_state",),
        supports_batch=False,
        supports_resume=True,
        metadata={"layer": "pipeline", "component": "seasonal_decomposition", "route": "stl_like"},
    )

    def __init__(self, config: SeasonalDecompositionConfig | Mapping[str, Any]) -> None:
        self.config = config if isinstance(config, SeasonalDecompositionConfig) else SeasonalDecompositionConfig(**dict(config))

    def fit(self, data: TimeSeriesDataView, context: Mapping[str, Any] | None = None) -> Mapping[str, Any]:
        _ = context
        result = seasonal_decompose(data, self.config)
        return {
            "config": self.config.describe(),
            "period": int(result.period),
            "n_obs": int(result.original.shape[0]),
            "resid_std": float(np.std(result.resid)),
            "seasonal_std": float(np.std(result.seasonal)),
        }

    def transform(
        self,
        data: TimeSeriesDataView,
        state: Mapping[str, Any] | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> SeasonalDecompositionResult | TimeSeriesDataView:
        _ = state
        _ = context
        result = seasonal_decompose(data, self.config)
        if bool(self.config.residual_target):
            return result.to_data_view(component="resid", target_name=f"{data.target_name}.resid")
        return result

    def fit_transform(
        self,
        data: TimeSeriesDataView,
        context: Mapping[str, Any] | None = None,
    ) -> tuple[SeasonalDecompositionResult | TimeSeriesDataView, Mapping[str, Any]]:
        state = self.fit(data, context)
        return self.transform(data, state, context), state

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "config": self.config.describe()}


@dataclass(frozen=True)
class TimeSeriesWindowConfig:
    """Lag-window configuration for converting a series into supervised rows.

    format="flat" produces 2D [n_samples, lagged_features] for classical models.
    format="sequence" produces 3D [n_samples, sequence_length, input_dim] for
    neural temporal models.  In sequence mode lags must be a contiguous range
    [1, 2, ..., sequence_length].

    input_dim accounts for the univariate target plus any exogenous columns.
    """

    lags: Sequence[int] = (1,)
    horizon: int = 1
    valid_size: int | float = 0.2
    include_exogenous: bool = True
    include_origin_index: bool = False
    format: str = "flat"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def normalized_lags(self) -> tuple[int, ...]:
        lags = tuple(sorted({int(lag) for lag in self.lags if int(lag) > 0}))
        if not lags:
            raise ValueError("TimeSeriesWindowConfig requires at least one positive lag")
        return lags

    def normalized_format(self) -> str:
        fmt = str(self.format or "flat").strip().lower()
        if fmt not in {"flat", "sequence"}:
            raise ValueError("format must be 'flat' or 'sequence'")
        return fmt

    def describe(self) -> dict[str, Any]:
        return {
            "lags": tuple(int(lag) for lag in self.normalized_lags()),
            "horizon": int(self.horizon),
            "valid_size": self.valid_size,
            "include_exogenous": bool(self.include_exogenous),
            "include_origin_index": bool(self.include_origin_index),
            "format": self.normalized_format(),
            "metadata": dict(self.metadata),
        }


class TimeSeriesWindowingComponent(ContractMixin):
    """Convert TimeSeriesDataView into a lagged NumericDataView."""

    name = "time_series_windowing"
    context_requires = ("data.time_series_view",)
    context_optional = ("trainer.context",)
    context_provides = ("data.numeric_view", "pipeline.feature_space", "time_series.window_config")
    context_mutates = ("pipeline.component_state",)
    context_cache = ()
    requires_metrics = ()
    metrics_fallback = "strict"
    context_notes = "Builds supervised lag/window rows from ordered time-series data."
    contract = ComponentContract(
        name=name,
        requires=("data.time_series_view",),
        optional=("trainer.context",),
        provides=("data.numeric_view", "pipeline.feature_space", "time_series.window_config"),
        mutates=("pipeline.component_state",),
        supports_batch=True,
        supports_resume=True,
        metadata={"layer": "pipeline", "component": "time_series_windowing"},
    )

    def __init__(self, config: TimeSeriesWindowConfig | Mapping[str, Any] | None = None) -> None:
        self.config = config if isinstance(config, TimeSeriesWindowConfig) else TimeSeriesWindowConfig(**dict(config or {}))

    def fit(self, data: TimeSeriesDataView, context: Mapping[str, Any] | None = None) -> Mapping[str, Any]:
        _ = context
        numeric = build_lagged_numeric_view(data, self.config)
        return {
            "n_train": int(numeric.X_train.shape[0]),
            "n_valid": 0 if numeric.X_valid is None else int(numeric.X_valid.shape[0]),
            "n_features": int(numeric.X_train.shape[1]),
            "config": self.config.describe(),
        }

    def transform(
        self,
        data: TimeSeriesDataView,
        state: Mapping[str, Any] | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> NumericDataView:
        _ = state
        _ = context
        return build_lagged_numeric_view(data, self.config)

    def fit_transform(
        self,
        data: TimeSeriesDataView,
        context: Mapping[str, Any] | None = None,
    ) -> tuple[NumericDataView, Mapping[str, Any]]:
        state = self.fit(data, context)
        return self.transform(data, state, context), state

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "config": self.config.describe()}


def build_lagged_numeric_view(
    data: TimeSeriesDataView,
    config: TimeSeriesWindowConfig | Mapping[str, Any] | None = None,
) -> NumericDataView:
    cfg = config if isinstance(config, TimeSeriesWindowConfig) else TimeSeriesWindowConfig(**dict(config or {}))
    lags = cfg.normalized_lags()
    horizon = int(cfg.horizon)
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    y = np.asarray(data.y, dtype=float).reshape(-1)
    max_lag = max(lags)
    first_origin = max_lag - 1
    last_origin = y.shape[0] - horizon - 1
    if last_origin < first_origin:
        raise ValueError("series is too short for requested lags and horizon")
    exog = None if data.exogenous is None else np.asarray(data.exogenous, dtype=float)
    fmt = cfg.normalized_format()

    if fmt == "sequence":
        return _build_sequence_view(data, cfg, y, exog, lags, horizon, first_origin, last_origin)

    rows: list[list[float]] = []
    targets: list[float | list[float]] = []
    origin_indices: list[int] = []
    target_indices: list[int] = []
    for origin in range(first_origin, last_origin + 1):
        row = [float(y[origin + 1 - lag]) for lag in lags]
        if bool(cfg.include_exogenous) and exog is not None:
            row.extend(float(value) for value in exog[origin, :])
        if bool(cfg.include_origin_index):
            denom = max(1, y.shape[0] - 1)
            row.append(float(origin) / float(denom))
        rows.append(row)
        if horizon == 1:
            targets.append(float(y[origin + 1]))
            target_indices.append(int(origin + 1))
        else:
            targets.append([float(y[origin + step]) for step in range(1, horizon + 1)])
            target_indices.append(int(origin + horizon))
        origin_indices.append(int(origin))

    X = np.asarray(rows, dtype=float)
    target = np.asarray(targets, dtype=float)
    if target.ndim == 1:
        target = target.reshape(-1)
    valid_count = _resolve_valid_size(cfg.valid_size, X.shape[0])
    if valid_count <= 0:
        split = X.shape[0]
        X_valid = None
        y_valid = None
    else:
        split = X.shape[0] - valid_count
        if split <= 0:
            raise ValueError("valid_size leaves no training windows")
        X_valid = X[split:]
        y_valid = target[split:]

    feature_names = [f"lag_{lag}" for lag in lags]
    if bool(cfg.include_exogenous) and exog is not None:
        feature_names.extend(f"{name}@t" for name in data.exogenous_names)
    if bool(cfg.include_origin_index):
        feature_names.append("origin_index")

    return NumericDataView(
        X_train=X[:split],
        y_train=target[:split],
        X_valid=X_valid,
        y_valid=y_valid,
        feature_names=tuple(feature_names),
        target_name=f"{data.target_name}.h{horizon}",
        metadata={
            **dict(data.metadata),
            "time_series": True,
            "time_series.window_config": cfg.describe(),
            "time_series.forecast_horizon": int(horizon),
            "time_series.origin_indices": tuple(origin_indices),
            "time_series.target_indices": tuple(target_indices),
            "time_series.target_time_index": tuple(
                data.time_index[min(idx, len(data.time_index) - 1)] for idx in target_indices
            ),
        },
    )


def _build_sequence_view(
    data: TimeSeriesDataView,
    cfg: TimeSeriesWindowConfig,
    y: np.ndarray,
    exog: np.ndarray | None,
    lags: tuple[int, ...],
    horizon: int,
    first_origin: int,
    last_origin: int,
) -> NumericDataView:
    expected = tuple(range(min(lags), max(lags) + 1))
    if lags != expected:
        raise ValueError("sequence format requires consecutive lags, e.g. [1, 2, 3, 4]")
    seq_len = len(lags)
    input_dim = 1
    exog_dim = 0
    if bool(cfg.include_exogenous) and exog is not None:
        exog_dim = exog.shape[1]
        input_dim += exog_dim

    sequences: list[np.ndarray] = []
    targets: list[float | list[float]] = []
    origin_indices: list[int] = []
    target_indices: list[int] = []
    for origin in range(first_origin, last_origin + 1):
        seq = np.zeros((seq_len, input_dim), dtype=float)
        for idx, lag in enumerate(lags):
            t = origin + 1 - lag
            seq[idx, 0] = float(y[t])
            if exog_dim:
                for j in range(exog_dim):
                    seq[idx, 1 + j] = float(exog[t, j])
        sequences.append(seq)
        if horizon == 1:
            targets.append(float(y[origin + 1]))
            target_indices.append(int(origin + 1))
        else:
            targets.append([float(y[origin + step]) for step in range(1, horizon + 1)])
            target_indices.append(int(origin + horizon))
        origin_indices.append(int(origin))

    X = np.stack(sequences, axis=0)
    n_samples = X.shape[0]
    X_flat = X.reshape(n_samples, seq_len * input_dim)
    target = np.asarray(targets, dtype=float)
    if target.ndim == 1:
        target = target.reshape(-1)
    valid_count = _resolve_valid_size(cfg.valid_size, n_samples)
    if valid_count <= 0:
        split = n_samples
        X_valid = None
        y_valid = None
    else:
        split = n_samples - valid_count
        if split <= 0:
            raise ValueError("valid_size leaves no training windows")
        X_valid = X_flat[split:]
        y_valid = target[split:]

    return NumericDataView(
        X_train=X_flat[:split],
        y_train=target[:split],
        X_valid=X_valid,
        y_valid=y_valid,
        feature_names=None,
        target_name=f"{data.target_name}.h{horizon}",
        metadata={
            **dict(data.metadata),
            "time_series": True,
            "time_series.format": "sequence",
            "time_series.sequence_length": seq_len,
            "time_series.input_dim": input_dim,
            "time_series.forecast_horizon": int(horizon),
            "time_series.window_config": cfg.describe(),
            "time_series.origin_indices": tuple(origin_indices),
            "time_series.target_indices": tuple(target_indices),
            "time_series.target_time_index": tuple(
                data.time_index[min(idx, len(data.time_index) - 1)] for idx in target_indices
            ),
        },
    )


def seasonal_decompose(
    data: TimeSeriesDataView,
    config: SeasonalDecompositionConfig | Mapping[str, Any],
) -> SeasonalDecompositionResult:
    cfg = config if isinstance(config, SeasonalDecompositionConfig) else SeasonalDecompositionConfig(**dict(config))
    y = np.asarray(data.y, dtype=float).reshape(-1)
    period = cfg.normalized_period()
    if y.shape[0] < period * 2:
        raise ValueError("seasonal decomposition requires at least two full periods")
    key = str(cfg.model or "additive").strip().lower()
    if key not in {"additive", "stl", "stl_like"}:
        raise ValueError("only additive/stl-like seasonal decomposition is currently supported")

    trend = _centered_moving_average(y, cfg.normalized_trend_window())
    detrended = y - trend
    phase_means = np.zeros(period, dtype=float)
    for phase in range(period):
        values = detrended[phase::period]
        phase_means[phase] = float(np.mean(values)) if values.size else 0.0
    phase_means = phase_means - float(np.mean(phase_means))
    seasonal = np.asarray([phase_means[idx % period] for idx in range(y.shape[0])], dtype=float)
    resid = y - trend - seasonal
    return SeasonalDecompositionResult(
        original=y,
        trend=trend,
        seasonal=seasonal,
        resid=resid,
        period=period,
        model="additive",
        time_index=data.time_index,
        metadata={
            **dict(data.metadata),
            **dict(cfg.metadata),
            "frequency": str(data.frequency),
            "route": "stl_like",
            "trend_window": int(cfg.normalized_trend_window()),
        },
    )


def _centered_moving_average(values: np.ndarray, window: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    width = int(window)
    if width <= 1:
        return arr.copy()
    if width % 2 == 0:
        width += 1
    pad = width // 2
    padded = np.pad(arr, (pad, pad), mode="edge")
    kernel = np.ones(width, dtype=float) / float(width)
    return np.convolve(padded, kernel, mode="valid").reshape(-1)


def _resolve_valid_size(value: int | float, n_rows: int) -> int:
    if isinstance(value, float) and 0.0 < float(value) < 1.0:
        return max(1, int(round(float(value) * float(n_rows)))) if n_rows > 1 else 0
    count = int(value)
    if count < 0:
        raise ValueError("valid_size must be non-negative")
    if count >= n_rows:
        return max(0, n_rows - 1)
    return count


__all__ = [
    "SeasonalDecompositionConfig",
    "SeasonalDecompositionResult",
    "STLSeasonalDecompositionComponent",
    "TimeSeriesWindowConfig",
    "TimeSeriesWindowingComponent",
    "build_lagged_numeric_view",
    "seasonal_decompose",
]
