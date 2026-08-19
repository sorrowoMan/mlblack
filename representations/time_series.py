from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from blackbase.contracts import ComponentContract
from mlblack.core.representation import ModelRepresentation
from mlblack.core.types import UnknownState
from mlblack.models.time_series import NaiveForecastModel


@dataclass(frozen=True)
class BaselineForecastSearchConfig:
    """Search space for simple univariate forecasting baselines."""

    strategies: Sequence[str] = ("naive", "seasonal_naive", "moving_average")
    window_bounds: tuple[int, int] = (2, 12)
    seasonal_period_bounds: tuple[int, int] = (1, 24)
    random_seed: int = 42
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def normalized_strategies(self) -> tuple[str, ...]:
        strategies = tuple(str(item).strip() for item in self.strategies if str(item).strip())
        if not strategies:
            raise ValueError("BaselineForecastSearchConfig requires at least one strategy")
        return strategies

    def describe(self) -> dict[str, Any]:
        return {
            "strategies": self.normalized_strategies(),
            "window_bounds": tuple(int(value) for value in self.window_bounds),
            "seasonal_period_bounds": tuple(int(value) for value in self.seasonal_period_bounds),
            "random_seed": int(self.random_seed),
            "metadata": dict(self.metadata),
        }


class BaselineForecastRepresentation(ModelRepresentation):
    """UnknownState -> Naive/seasonal-naive/moving-average forecast model."""

    name = "baseline_forecast"
    context_requires = ("candidate.unknown_state",)
    context_optional = ("time_series.search_space",)
    context_provides = ("candidate.forecast_model",)
    context_mutates = ("candidate.repaired_state",)
    context_cache = ()
    requires_metrics = ()
    metrics_fallback = "strict"
    context_notes = "Decodes a small numeric state into a baseline forecast model."
    contract = ComponentContract(
        name=name,
        requires=("candidate.unknown_state",),
        optional=("time_series.search_space",),
        provides=("candidate.forecast_model",),
        mutates=("candidate.repaired_state",),
        supports_gradient=False,
        supports_batch=True,
        supports_resume=True,
        metadata={"family": "time_series", "representation": "baseline_forecast"},
    )

    def __init__(self, config: BaselineForecastSearchConfig | Mapping[str, Any] | None = None) -> None:
        self.config = config if isinstance(config, BaselineForecastSearchConfig) else BaselineForecastSearchConfig(**dict(config or {}))
        self.dimension = 3
        self._rng = np.random.default_rng(int(self.config.random_seed))

    def init(self, context: Mapping[str, Any]) -> UnknownState:
        _ = context
        strategies = self.config.normalized_strategies()
        window_low, window_high = _bounds(self.config.window_bounds, lower_floor=1)
        season_low, season_high = _bounds(self.config.seasonal_period_bounds, lower_floor=1)
        values = np.asarray(
            [
                float(self._rng.integers(0, len(strategies))),
                float(self._rng.integers(window_low, window_high + 1)),
                float(self._rng.integers(season_low, season_high + 1)),
            ],
            dtype=float,
        )
        return UnknownState(values=values, metadata={"source": f"{self.name}_init"})

    def decode(self, state: UnknownState, context: Mapping[str, Any] | None = None) -> NaiveForecastModel:
        _ = context
        repaired = self.repair(state)
        values = repaired.as_array()
        strategies = self.config.normalized_strategies()
        strategy_index = int(round(float(values[0])))
        strategy = strategies[min(max(0, strategy_index), len(strategies) - 1)]
        return NaiveForecastModel(
            strategy=strategy,
            window=int(round(float(values[1]))),
            seasonal_period=int(round(float(values[2]))),
            metadata={"representation": self.name, "state": values.tolist()},
        )

    def repair(self, state: UnknownState, context: Mapping[str, Any] | None = None) -> UnknownState:
        _ = context
        values = state.as_array()
        if values.shape[0] != self.dimension:
            raise ValueError(f"state dimension {values.shape[0]} does not match {self.dimension}")
        strategies = self.config.normalized_strategies()
        window_low, window_high = _bounds(self.config.window_bounds, lower_floor=1)
        season_low, season_high = _bounds(self.config.seasonal_period_bounds, lower_floor=1)
        repaired = np.asarray(
            [
                float(np.clip(round(float(values[0])), 0, len(strategies) - 1)),
                float(np.clip(round(float(values[1])), window_low, window_high)),
                float(np.clip(round(float(values[2])), season_low, season_high)),
            ],
            dtype=float,
        )
        return state.with_values(repaired, repaired=True)

    def describe(self) -> Mapping[str, Any]:
        return {
            "name": self.name,
            "family": "time_series",
            "dimension": int(self.dimension),
            "config": self.config.describe(),
        }


class ForecastModelSpecRepresentation(ModelRepresentation):
    """Single-candidate Codec for one declarative forecast fit specification."""

    name = "forecast_model_spec"
    context_requires = ("candidate.unknown_state", "candidate.model_spec")
    context_optional = ()
    context_provides = ("candidate.model_spec",)
    context_mutates = ()
    context_cache = ()
    requires_metrics = ()
    metrics_fallback = "strict"
    context_notes = "Decodes a declarative fit specification; the Problem/Provider owns fitting."
    contract = ComponentContract(
        name=name,
        requires=("candidate.unknown_state", "candidate.model_spec"),
        provides=("candidate.model_spec",),
        supports_gradient=False,
        supports_batch=True,
        supports_resume=True,
        metadata={"family": "time_series", "representation": "forecast_model_spec"},
    )

    def __init__(self, spec: Any, *, metadata: Mapping[str, Any] | None = None) -> None:
        self.spec = spec
        self.metadata = dict(metadata or {})
        # NSGABlack candidates are non-empty numeric states.  This sentinel is
        # a stable identity token, not a fake trainable hyperparameter.
        self.dimension = 1

    def init(self, context: Mapping[str, Any]) -> UnknownState:
        _ = context
        return UnknownState(values=np.zeros(1, dtype=float), metadata={"source": f"{self.name}_init"})

    def encode(self, model: Any, context: Mapping[str, Any] | None = None) -> UnknownState:
        _ = context
        if model is not self.spec and model != self.spec:
            raise ValueError("ForecastModelSpecRepresentation only encodes its configured spec")
        return self.init({})

    def decode(self, state: UnknownState, context: Mapping[str, Any] | None = None) -> Any:
        values = np.asarray(state.as_array(), dtype=float).reshape(-1)
        if values.shape != (1,):
            raise ValueError("forecast model spec state must contain one identity token")
        _ = context
        return self.spec

    def repair(self, state: UnknownState, context: Mapping[str, Any] | None = None) -> UnknownState:
        del state, context
        return self.init({})

    def describe(self) -> Mapping[str, Any]:
        describe = getattr(self.spec, "describe", None)
        spec_payload = describe() if callable(describe) else {"spec_type": type(self.spec).__name__}
        return {
            "name": self.name,
            "family": "time_series",
            "dimension": int(self.dimension),
            "spec": spec_payload,
            "metadata": dict(self.metadata),
        }


def _bounds(bounds: Sequence[int], *, lower_floor: int) -> tuple[int, int]:
    values = tuple(int(value) for value in bounds)
    if len(values) != 2:
        raise ValueError("bounds must contain exactly two values")
    low, high = min(values), max(values)
    low = max(int(lower_floor), low)
    high = max(low, high)
    return low, high


__all__ = [
    "BaselineForecastRepresentation",
    "BaselineForecastSearchConfig",
    "ForecastModelSpecRepresentation",
]
