from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from mlblack.core.contracts import ComponentContract
from mlblack.core.representation import ModelRepresentation
from mlblack.core.types import UnknownState
from mlblack.models import EstimatorFactory, EstimatorSpecModel
from mlblack.representations.codecs import (
    BoostingMechanismSpec,
    EstimatorCodecConfig,
    EstimatorSpecCodec,
    TreeMechanismSpec,
    TunableParameterSpec,
    tunables_from_legacy,
)


@dataclass(frozen=True)
class EstimatorRouteConfig:
    family: str
    route: str
    params: Mapping[str, Any] = field(default_factory=dict)
    tunable_params: tuple[str, ...] = tuple()
    bounds: Mapping[str, tuple[float, float]] = field(default_factory=dict)
    integer_params: tuple[str, ...] = tuple()
    factory: EstimatorFactory | None = None
    tunables: tuple[TunableParameterSpec, ...] = tuple()
    mechanisms: Mapping[str, Any] = field(default_factory=dict)


class EstimatorSpecRepresentation(ModelRepresentation):
    """Unknown vector -> external estimator specification."""

    name = "estimator_spec"
    context_requires = ('candidate.unknown_state',)
    context_optional = ()
    context_provides = ('candidate.model_spec', 'estimator.factory')
    context_mutates = ('candidate.repaired_state',)
    context_cache = ()
    requires_metrics = ()
    metrics_fallback = "strict"
    context_notes = 'Reads context fields: candidate.unknown_state; provides candidate.model_spec, estimator.factory; mutates candidate.repaired_state.'
    contract = ComponentContract(
        name=name,
        requires=("candidate.unknown_state",),
        provides=("candidate.model_spec", "estimator.factory"),
        mutates=("candidate.repaired_state",),
        supports_gradient=False,
        supports_batch=True,
        supports_resume=True,
    )

    def __init__(self, config: EstimatorRouteConfig) -> None:
        self.config = config
        tunables = tuple(config.tunables) or tunables_from_legacy(
            tuple(config.tunable_params),
            dict(config.bounds),
            tuple(config.integer_params),
        )
        self.codec = EstimatorSpecCodec(
            EstimatorCodecConfig(
                family=str(config.family),
                route=str(config.route),
                params=dict(config.params),
                tunables=tunables,
                factory=config.factory,
                mechanisms=dict(config.mechanisms),
            )
        )
        self.dimension = int(self.codec.base_dimension)

    def init(self, context: Mapping[str, Any]) -> UnknownState:
        _ = context
        return UnknownState(values=self.codec.init_values(), metadata={"source": f"{self.name}_init"})

    def encode(self, model: Any, context: Mapping[str, Any] | None = None) -> UnknownState:
        _ = context
        if not isinstance(model, EstimatorSpecModel):
            raise TypeError("EstimatorSpecRepresentation can only encode EstimatorSpecModel")
        values = self.codec.encode(model)
        return UnknownState(values=np.asarray(values, dtype=float), metadata={"source": "encoded_estimator_spec"})

    def decode(self, state: UnknownState, context: Mapping[str, Any] | None = None) -> EstimatorSpecModel:
        _ = context
        values = np.asarray(state.values, dtype=float).reshape(-1)
        if values.shape[0] != self.dimension:
            raise ValueError(f"state dimension {values.shape[0]} does not match representation dimension {self.dimension}")
        return self.codec.decode(values, metadata={"state_metadata": dict(state.metadata)})

    def repair(self, state: UnknownState, context: Mapping[str, Any] | None = None) -> UnknownState:
        _ = context
        arr = self.codec.repair_values(state.values)
        return state.with_values(arr)

    def describe(self) -> Mapping[str, Any]:
        return {
            "name": self.name,
            "dimension": int(self.dimension),
            "codec": self.codec.describe(),
        }


def build_tree_estimator_representation(
    *,
    route: str = "random_forest",
    params: Mapping[str, Any] | None = None,
    tunable_params: tuple[str, ...] = tuple(),
    bounds: Mapping[str, tuple[float, float]] | None = None,
    integer_params: tuple[str, ...] = tuple(),
    factory: EstimatorFactory | None = None,
    mechanism: TreeMechanismSpec | None = None,
) -> EstimatorSpecRepresentation:
    base_params = dict(params or {})
    mechanism_payload = {}
    if mechanism is not None:
        base_params = {**mechanism.to_params(route=route), **base_params}
        mechanism_payload["tree"] = mechanism.as_dict()
    return EstimatorSpecRepresentation(
        EstimatorRouteConfig(
            family="tree",
            route=route,
            params=base_params,
            tunable_params=tuple(tunable_params),
            bounds=dict(bounds or {}),
            integer_params=tuple(integer_params),
            factory=factory,
            mechanisms=mechanism_payload,
        )
    )


def make_sklearn_tree_factory(route: str) -> EstimatorFactory:
    route_key = str(route or "random_forest").strip().lower()

    def factory(params: Mapping[str, Any]) -> Any:
        try:
            from sklearn.ensemble import AdaBoostRegressor, BaggingRegressor, ExtraTreesRegressor, RandomForestRegressor
            from sklearn.tree import DecisionTreeRegressor
        except Exception as exc:
            raise RuntimeError("sklearn is required for sklearn tree estimator factories") from exc

        raw = dict(params)
        if route_key in {"decision_tree", "cart"}:
            return DecisionTreeRegressor(**raw)
        if route_key == "extra_trees":
            return ExtraTreesRegressor(**raw)
        if route_key == "bagging":
            return BaggingRegressor(**raw)
        if route_key == "adaboost":
            return AdaBoostRegressor(**raw)
        if route_key in {"random_forest", "rf"}:
            return RandomForestRegressor(**raw)
        raise ValueError(f"unsupported sklearn tree route: {route}")

    return factory


def build_tree_boosting_estimator_representation(
    *,
    route: str = "xgboost",
    params: Mapping[str, Any] | None = None,
    tunable_params: tuple[str, ...] = tuple(),
    bounds: Mapping[str, tuple[float, float]] | None = None,
    integer_params: tuple[str, ...] = tuple(),
    factory: EstimatorFactory | None = None,
    mechanism: BoostingMechanismSpec | None = None,
) -> EstimatorSpecRepresentation:
    base_params = dict(params or {})
    mechanism_payload = {}
    if mechanism is not None:
        base_params = {**mechanism.to_params(), **base_params}
        mechanism_payload["boosting"] = mechanism.as_dict()
    return EstimatorSpecRepresentation(
        EstimatorRouteConfig(
            family="tree_boosting",
            route=route,
            params=base_params,
            tunable_params=tuple(tunable_params),
            bounds=dict(bounds or {}),
            integer_params=tuple(integer_params),
            factory=factory,
            mechanisms=mechanism_payload,
        )
    )


def make_xgboost_factory() -> EstimatorFactory:
    def factory(params: Mapping[str, Any]) -> Any:
        try:
            from xgboost import XGBRegressor
        except Exception as exc:
            raise RuntimeError("xgboost is required for xgboost estimator factories") from exc
        return XGBRegressor(**dict(params))

    return factory


def build_neural_estimator_representation(
    *,
    route: str = "sklearn_mlp",
    params: Mapping[str, Any] | None = None,
    tunable_params: tuple[str, ...] = tuple(),
    bounds: Mapping[str, tuple[float, float]] | None = None,
    integer_params: tuple[str, ...] = tuple(),
    factory: EstimatorFactory | None = None,
    mechanisms: Mapping[str, Any] | None = None,
) -> EstimatorSpecRepresentation:
    return EstimatorSpecRepresentation(
        EstimatorRouteConfig(
            family="neural",
            route=route,
            params=dict(params or {}),
            tunable_params=tuple(tunable_params),
            bounds=dict(bounds or {}),
            integer_params=tuple(integer_params),
            factory=factory,
            mechanisms=dict(mechanisms or {}),
        )
    )


def make_sklearn_mlp_factory() -> EstimatorFactory:
    def factory(params: Mapping[str, Any]) -> Any:
        try:
            from sklearn.neural_network import MLPRegressor
        except Exception as exc:
            raise RuntimeError("sklearn is required for sklearn MLP estimator factories") from exc
        return MLPRegressor(**dict(params))

    return factory

