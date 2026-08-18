from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from mlblack.models import EstimatorFactory, EstimatorSpecModel


@dataclass(frozen=True)
class TunableParameterSpec:
    name: str
    low: float = 0.0
    high: float = 1.0
    integer: bool = False

    def initial_value(self) -> float:
        return (float(self.low) + float(self.high)) / 2.0

    def repair(self, value: float) -> float | int:
        fixed = float(np.clip(float(value), float(self.low), float(self.high)))
        return int(round(fixed)) if self.integer else fixed

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "low": float(self.low), "high": float(self.high), "integer": bool(self.integer)}


@dataclass(frozen=True)
class TreeSplitMechanismSpec:
    criterion: str = "squared_error"
    splitter: str = "best"
    max_depth: int | None = None
    min_samples_split: int | float = 2
    min_samples_leaf: int | float = 1
    min_impurity_decrease: float = 0.0
    ccp_alpha: float = 0.0

    def to_params(self) -> dict[str, Any]:
        return {
            "criterion": self.criterion,
            "splitter": self.splitter,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "min_impurity_decrease": float(self.min_impurity_decrease),
            "ccp_alpha": float(self.ccp_alpha),
        }


@dataclass(frozen=True)
class TreeSamplingMechanismSpec:
    bootstrap: bool = True
    max_samples: int | float | None = None
    max_features: int | float | str | None = 1.0
    random_state: int = 42
    n_jobs: int = -1

    def to_params(self) -> dict[str, Any]:
        return {
            "bootstrap": bool(self.bootstrap),
            "max_samples": self.max_samples,
            "max_features": self.max_features,
            "random_state": int(self.random_state),
            "n_jobs": int(self.n_jobs),
        }


@dataclass(frozen=True)
class TreePruningMechanismSpec:
    """Tree pruning / regularization mechanism."""

    ccp_alpha: float = 0.0
    min_impurity_decrease: float = 0.0
    max_leaf_nodes: int | None = None

    def to_params(self) -> dict[str, Any]:
        return {
            "ccp_alpha": float(self.ccp_alpha),
            "min_impurity_decrease": float(self.min_impurity_decrease),
            "max_leaf_nodes": self.max_leaf_nodes,
        }

    def as_dict(self) -> dict[str, Any]:
        return {key: value for key, value in self.to_params().items() if value is not None}


@dataclass(frozen=True)
class TreeContinuationSpec:
    warm_start: bool = False
    trainer_state_enabled: bool = True
    supports_resume: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_params(self) -> dict[str, Any]:
        return {"warm_start": bool(self.warm_start)}


@dataclass(frozen=True)
class TreeMechanismSpec:
    n_estimators: int = 100
    split: TreeSplitMechanismSpec = field(default_factory=TreeSplitMechanismSpec)
    sampling: TreeSamplingMechanismSpec = field(default_factory=TreeSamplingMechanismSpec)
    pruning: TreePruningMechanismSpec = field(default_factory=TreePruningMechanismSpec)
    continuation: TreeContinuationSpec = field(default_factory=TreeContinuationSpec)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_params(self, *, route: str) -> dict[str, Any]:
        route_key = str(route).strip().lower()
        params: dict[str, Any] = {}
        if route_key not in {"decision_tree", "cart"}:
            params["n_estimators"] = int(self.n_estimators)
            params.update(self.sampling.to_params())
            params.update(self.continuation.to_params())
        params.update(self.split.to_params())
        params.update(self.pruning.to_params())
        if route_key in {"extra_trees", "random_forest", "rf"}:
            params.pop("splitter", None)
        if route_key in {"bagging", "adaboost"}:
            for key in ("criterion", "splitter", "min_impurity_decrease", "ccp_alpha", "max_leaf_nodes"):
                params.pop(key, None)
        return {key: value for key, value in params.items() if value is not None}

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_estimators": int(self.n_estimators),
            "split": self.split.to_params(),
            "sampling": self.sampling.to_params(),
            "pruning": self.pruning.as_dict(),
            "continuation": {
                "warm_start": self.continuation.warm_start,
                "trainer_state_enabled": self.continuation.trainer_state_enabled,
                "supports_resume": self.continuation.supports_resume,
                "metadata": dict(self.continuation.metadata),
            },
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class BoostingEarlyStoppingSpec:
    rounds: int | None = None
    eval_metric: str | None = None
    use_validation: bool = True
    strict: bool = False

    def to_params(self) -> dict[str, Any]:
        params: dict[str, Any] = {}
        if self.rounds is not None:
            params["early_stopping_rounds"] = int(self.rounds)
        if self.eval_metric:
            params["eval_metric"] = str(self.eval_metric)
        return params

    def as_dict(self) -> dict[str, Any]:
        return {
            "rounds": self.rounds,
            "eval_metric": self.eval_metric,
            "use_validation": bool(self.use_validation),
            "strict": bool(self.strict),
        }


@dataclass(frozen=True)
class BoostingContinuationSpec:
    mode: str = "xgb_model"
    warm_start: bool = True
    trainer_state_enabled: bool = True
    artifact_resume_key: str = "booster"

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "warm_start": bool(self.warm_start),
            "trainer_state_enabled": bool(self.trainer_state_enabled),
            "artifact_resume_key": self.artifact_resume_key,
        }


@dataclass(frozen=True)
class BoostingMechanismSpec:
    n_estimators: int = 200
    learning_rate: float = 0.05
    objective: str = "reg:squarederror"
    tree_method: str = "hist"
    max_depth: int = 6
    min_child_weight: float = 1.0
    gamma: float = 0.0
    reg_lambda: float = 1.0
    reg_alpha: float = 0.0
    subsample: float = 0.9
    colsample_bytree: float = 0.9
    early_stopping_rounds: int | None = None
    early_stopping: BoostingEarlyStoppingSpec | None = None
    continuation_mode: str = "xgb_model"
    continuation: BoostingContinuationSpec | None = None
    warm_start: bool = True
    trainer_state_enabled: bool = True
    random_state: int = 42
    n_jobs: int = -1
    verbosity: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_params(self) -> dict[str, Any]:
        params = {
            "n_estimators": int(self.n_estimators),
            "learning_rate": float(self.learning_rate),
            "objective": str(self.objective),
            "tree_method": str(self.tree_method),
            "max_depth": int(self.max_depth),
            "min_child_weight": float(self.min_child_weight),
            "gamma": float(self.gamma),
            "reg_lambda": float(self.reg_lambda),
            "reg_alpha": float(self.reg_alpha),
            "subsample": float(self.subsample),
            "colsample_bytree": float(self.colsample_bytree),
            "random_state": int(self.random_state),
            "n_jobs": int(self.n_jobs),
            "verbosity": int(self.verbosity),
        }
        if self.early_stopping_rounds is not None:
            params["early_stopping_rounds"] = int(self.early_stopping_rounds)
        if self.early_stopping is not None:
            params.update(self.early_stopping.to_params())
        return params

    def as_dict(self) -> dict[str, Any]:
        continuation = self.continuation or BoostingContinuationSpec(
            mode=self.continuation_mode,
            warm_start=self.warm_start,
            trainer_state_enabled=self.trainer_state_enabled,
        )
        return {
            **self.to_params(),
            "early_stopping": None if self.early_stopping is None else self.early_stopping.as_dict(),
            "continuation": continuation.as_dict(),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class EstimatorCodecConfig:
    family: str
    route: str
    params: Mapping[str, Any] = field(default_factory=dict)
    tunables: tuple[TunableParameterSpec, ...] = tuple()
    factory: EstimatorFactory | None = None
    mechanisms: Mapping[str, Any] = field(default_factory=dict)


class EstimatorSpecCodec:
    """External estimator-spec codec with typed mechanism metadata."""

    def __init__(self, config: EstimatorCodecConfig) -> None:
        self.config = config
        self.base_dimension = int(len(config.tunables))

    def init_values(self) -> np.ndarray:
        return np.asarray([spec.initial_value() for spec in self.config.tunables], dtype=float)

    def encode(self, model: EstimatorSpecModel) -> np.ndarray:
        return np.asarray([float(model.params[spec.name]) for spec in self.config.tunables], dtype=float)

    def decode(self, values: np.ndarray, metadata: Mapping[str, Any] | None = None) -> EstimatorSpecModel:
        arr = self.repair_values(values)
        params = dict(self.config.params)
        for idx, spec in enumerate(self.config.tunables):
            params[spec.name] = spec.repair(float(arr[idx]))
        return EstimatorSpecModel(
            family=str(self.config.family),
            route=str(self.config.route),
            params=params,
            factory=self.config.factory,
            metadata={
                "representation": "estimator_spec",
                "mechanisms": dict(self.config.mechanisms),
                **dict(metadata or {}),
            },
        )

    def repair_values(self, values: np.ndarray) -> np.ndarray:
        arr = np.asarray(values, dtype=float).reshape(-1)
        if arr.shape[0] != self.base_dimension:
            fixed = np.zeros(self.base_dimension, dtype=float)
            fixed[: min(self.base_dimension, arr.shape[0])] = arr[: min(self.base_dimension, arr.shape[0])]
            arr = fixed
        repaired = [float(spec.repair(float(arr[idx]))) for idx, spec in enumerate(self.config.tunables)]
        return np.asarray(repaired, dtype=float)

    def describe(self) -> dict[str, Any]:
        return {
            "codec": "estimator_spec",
            "family": self.config.family,
            "route": self.config.route,
            "base_dimension": int(self.base_dimension),
            "params": dict(self.config.params),
            "tunables": [spec.as_dict() for spec in self.config.tunables],
            "mechanisms": dict(self.config.mechanisms),
        }


def build_tunable_parameter_specs(
    names: tuple[str, ...],
    bounds: Mapping[str, tuple[float, float]],
    integer_params: tuple[str, ...],
) -> tuple[TunableParameterSpec, ...]:
    integers = {str(name) for name in integer_params}
    return tuple(
        TunableParameterSpec(
            name=str(name),
            low=float(bounds.get(name, (0.0, 1.0))[0]),
            high=float(bounds.get(name, (0.0, 1.0))[1]),
            integer=str(name) in integers,
        )
        for name in names
    )

