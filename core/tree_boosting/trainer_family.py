from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Mapping

from core.common.family_router import FamilyRouteSpec, resolve_family_route_spec, serialize_family_route_registry
from core.mechanisms import (
    MechanismProtocolBase,
    build_tree_boosting_family_mechanism_bindings,
    serialize_family_bindings,
)

TREE_BOOSTING_FORMAL_PRESET_KEY: str = "tree_boosting"
TREE_BOOSTING_ROUTE_KEYS: tuple[str, ...] = ("xgboost",)


def _normalize_name(value: str | None, default: str) -> str:
    text = str(value or "").strip().lower()
    return text or str(default)


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in sorted(dict(value).items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(v) for v in value]
    return str(value)


def _stable_hash(payload: Any) -> str | None:
    if payload is None:
        return None
    encoded = json.dumps(_jsonable(payload), sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    if encoded in {"null", "{}", "[]", "\"\""}:
        return None
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class TreeBoostingBackendSpec:
    backend: str = "xgboost"
    booster: str = "gbtree"
    trainer_kind: str = "gradient_boosted_trees"
    continuation_mode: str = "xgb_model"
    trainer_state_enabled: bool = True
    supports_resume: bool = True
    supports_warm_start: bool = True
    supports_incremental: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "backend", _normalize_name(self.backend, "xgboost"))
        object.__setattr__(self, "booster", _normalize_name(self.booster, "gbtree"))
        object.__setattr__(self, "trainer_kind", _normalize_name(self.trainer_kind, "gradient_boosted_trees"))
        object.__setattr__(self, "continuation_mode", _normalize_name(self.continuation_mode, "xgb_model"))
        object.__setattr__(self, "trainer_state_enabled", bool(self.trainer_state_enabled))
        object.__setattr__(self, "supports_resume", bool(self.supports_resume))
        object.__setattr__(self, "supports_warm_start", bool(self.supports_warm_start))
        object.__setattr__(self, "supports_incremental", bool(self.supports_incremental))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def as_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "booster": self.booster,
            "trainer_kind": self.trainer_kind,
            "continuation_mode": self.continuation_mode,
            "trainer_state_enabled": bool(self.trainer_state_enabled),
            "supports_resume": bool(self.supports_resume),
            "supports_warm_start": bool(self.supports_warm_start),
            "supports_incremental": bool(self.supports_incremental),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class TreeBoostingEnsembleSpec:
    n_estimators: int = 400
    learning_rate: float = 0.05
    objective: str = "reg:squarederror"
    tree_method: str = "hist"
    verbosity: int = 0
    aggregation: str = "additive"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "n_estimators", max(1, int(self.n_estimators)))
        object.__setattr__(self, "learning_rate", max(0.0, float(self.learning_rate)))
        object.__setattr__(self, "objective", str(self.objective or "reg:squarederror").strip() or "reg:squarederror")
        object.__setattr__(self, "tree_method", _normalize_name(self.tree_method, "hist"))
        object.__setattr__(self, "verbosity", int(self.verbosity))
        object.__setattr__(self, "aggregation", _normalize_name(self.aggregation, "additive"))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_estimators": int(self.n_estimators),
            "learning_rate": float(self.learning_rate),
            "objective": self.objective,
            "tree_method": self.tree_method,
            "verbosity": int(self.verbosity),
            "aggregation": self.aggregation,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class TreeBoostingSamplingSpec:
    subsample: float = 0.9
    colsample_bytree: float = 0.9
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "subsample", max(0.0, float(self.subsample)))
        object.__setattr__(self, "colsample_bytree", max(0.0, float(self.colsample_bytree)))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def as_dict(self) -> dict[str, Any]:
        return {
            "subsample": float(self.subsample),
            "colsample_bytree": float(self.colsample_bytree),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class TreeBoostingRegularizationSpec:
    max_depth: int = 6
    min_child_weight: float = 1.0
    gamma: float = 0.0
    reg_lambda: float = 1.0
    reg_alpha: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "max_depth", max(1, int(self.max_depth)))
        object.__setattr__(self, "min_child_weight", max(0.0, float(self.min_child_weight)))
        object.__setattr__(self, "gamma", max(0.0, float(self.gamma)))
        object.__setattr__(self, "reg_lambda", max(0.0, float(self.reg_lambda)))
        object.__setattr__(self, "reg_alpha", max(0.0, float(self.reg_alpha)))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def as_dict(self) -> dict[str, Any]:
        return {
            "max_depth": int(self.max_depth),
            "min_child_weight": float(self.min_child_weight),
            "gamma": float(self.gamma),
            "reg_lambda": float(self.reg_lambda),
            "reg_alpha": float(self.reg_alpha),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class TreeBoostingExecutionSpec:
    n_jobs: int = -1
    random_seed: int = 42
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "n_jobs", int(self.n_jobs))
        object.__setattr__(self, "random_seed", int(self.random_seed))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_jobs": int(self.n_jobs),
            "random_seed": int(self.random_seed),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class TreeBoostingTaskHeadSpec:
    task: str = "point"
    objective_family: str = "regression"
    outputs: tuple[str, ...] = ("mean",)
    uncertainty_mode: str = "residual_std"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "task", _normalize_name(self.task, "point"))
        object.__setattr__(self, "objective_family", _normalize_name(self.objective_family, "regression"))
        outs = tuple(str(v).strip().lower() for v in tuple(self.outputs) if str(v).strip()) or ("mean",)
        object.__setattr__(self, "outputs", outs)
        object.__setattr__(self, "uncertainty_mode", _normalize_name(self.uncertainty_mode, "residual_std"))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def as_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "objective_family": self.objective_family,
            "outputs": tuple(self.outputs),
            "uncertainty_mode": self.uncertainty_mode,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class TreeBoostingTrainerFamilySpec:
    trainer_key: str = "xgboost"
    backend: TreeBoostingBackendSpec = field(default_factory=TreeBoostingBackendSpec)
    boosting: TreeBoostingEnsembleSpec = field(default_factory=TreeBoostingEnsembleSpec)
    sampling: TreeBoostingSamplingSpec = field(default_factory=TreeBoostingSamplingSpec)
    regularization: TreeBoostingRegularizationSpec = field(default_factory=TreeBoostingRegularizationSpec)
    execution: TreeBoostingExecutionSpec = field(default_factory=TreeBoostingExecutionSpec)
    task_head: TreeBoostingTaskHeadSpec = field(default_factory=TreeBoostingTaskHeadSpec)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "trainer_key", _normalize_name(self.trainer_key, "xgboost"))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def as_dict(self) -> dict[str, Any]:
        return {
            "trainer_key": self.trainer_key,
            "backend": self.backend.as_dict(),
            "boosting": self.boosting.as_dict(),
            "sampling": self.sampling.as_dict(),
            "regularization": self.regularization.as_dict(),
            "execution": self.execution.as_dict(),
            "task_head": self.task_head.as_dict(),
            "metadata": dict(self.metadata),
        }

    def mechanism_bindings(self) -> tuple[MechanismProtocolBase, ...]:
        return build_tree_boosting_family_mechanism_bindings()

    def mechanism_binding_payload(self) -> list[dict[str, object]]:
        return serialize_family_bindings(self.mechanism_bindings())

    def description_dict(self) -> dict[str, Any]:
        payload = self.as_dict()
        payload["mechanism_bindings"] = self.mechanism_binding_payload()
        return payload

    def family_signature(self) -> str | None:
        payload = self.as_dict()
        backend = dict(payload.get("backend", {}) or {})
        for key in ("trainer_state_enabled", "supports_resume", "supports_warm_start", "supports_incremental"):
            backend.pop(key, None)
        payload["backend"] = backend

        boosting = dict(payload.get("boosting", {}) or {})
        for key in ("n_estimators", "verbosity"):
            boosting.pop(key, None)
        payload["boosting"] = boosting

        execution = dict(payload.get("execution", {}) or {})
        for key in ("n_jobs", "random_seed"):
            execution.pop(key, None)
        payload["execution"] = execution
        return _stable_hash(payload)


def coerce_tree_boosting_backend_spec(
    value: TreeBoostingBackendSpec | Mapping[str, Any] | None,
    *,
    default: TreeBoostingBackendSpec | Mapping[str, Any] | None = None,
) -> TreeBoostingBackendSpec:
    if value is None:
        value = TreeBoostingBackendSpec() if default is None else default
    if isinstance(value, TreeBoostingBackendSpec):
        return value
    return TreeBoostingBackendSpec(**dict(value))


def coerce_tree_boosting_ensemble_spec(
    value: TreeBoostingEnsembleSpec | Mapping[str, Any] | None,
    *,
    default: TreeBoostingEnsembleSpec | Mapping[str, Any] | None = None,
) -> TreeBoostingEnsembleSpec:
    if value is None:
        value = TreeBoostingEnsembleSpec() if default is None else default
    if isinstance(value, TreeBoostingEnsembleSpec):
        return value
    return TreeBoostingEnsembleSpec(**dict(value))


def coerce_tree_boosting_sampling_spec(
    value: TreeBoostingSamplingSpec | Mapping[str, Any] | None,
    *,
    default: TreeBoostingSamplingSpec | Mapping[str, Any] | None = None,
) -> TreeBoostingSamplingSpec:
    if value is None:
        value = TreeBoostingSamplingSpec() if default is None else default
    if isinstance(value, TreeBoostingSamplingSpec):
        return value
    return TreeBoostingSamplingSpec(**dict(value))


def coerce_tree_boosting_regularization_spec(
    value: TreeBoostingRegularizationSpec | Mapping[str, Any] | None,
    *,
    default: TreeBoostingRegularizationSpec | Mapping[str, Any] | None = None,
) -> TreeBoostingRegularizationSpec:
    if value is None:
        value = TreeBoostingRegularizationSpec() if default is None else default
    if isinstance(value, TreeBoostingRegularizationSpec):
        return value
    return TreeBoostingRegularizationSpec(**dict(value))


def coerce_tree_boosting_execution_spec(
    value: TreeBoostingExecutionSpec | Mapping[str, Any] | None,
    *,
    default: TreeBoostingExecutionSpec | Mapping[str, Any] | None = None,
) -> TreeBoostingExecutionSpec:
    if value is None:
        value = TreeBoostingExecutionSpec() if default is None else default
    if isinstance(value, TreeBoostingExecutionSpec):
        return value
    return TreeBoostingExecutionSpec(**dict(value))


def coerce_tree_boosting_task_head_spec(
    value: TreeBoostingTaskHeadSpec | Mapping[str, Any] | None,
    *,
    default: TreeBoostingTaskHeadSpec | Mapping[str, Any] | None = None,
) -> TreeBoostingTaskHeadSpec:
    if value is None:
        value = TreeBoostingTaskHeadSpec() if default is None else default
    if isinstance(value, TreeBoostingTaskHeadSpec):
        return value
    return TreeBoostingTaskHeadSpec(**dict(value))


def build_xgboost_family_spec(
    *,
    trainer_key: str = "xgboost",
    n_estimators: int = 400,
    max_depth: int = 6,
    learning_rate: float = 0.05,
    subsample: float = 0.9,
    colsample_bytree: float = 0.9,
    min_child_weight: float = 1.0,
    gamma: float = 0.0,
    reg_lambda: float = 1.0,
    reg_alpha: float = 0.0,
    objective: str = "reg:squarederror",
    tree_method: str = "hist",
    n_jobs: int = -1,
    random_seed: int = 42,
    verbosity: int = 0,
    uncertainty_mode: str = "residual_std",
    metadata: Mapping[str, Any] | None = None,
) -> TreeBoostingTrainerFamilySpec:
    return TreeBoostingTrainerFamilySpec(
        trainer_key=trainer_key,
        backend=TreeBoostingBackendSpec(
            backend="xgboost",
            booster="gbtree",
            trainer_kind="gradient_boosted_trees",
            continuation_mode="xgb_model",
            trainer_state_enabled=True,
            supports_resume=True,
            supports_warm_start=True,
            supports_incremental=True,
        ),
        boosting=TreeBoostingEnsembleSpec(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            objective=objective,
            tree_method=tree_method,
            verbosity=verbosity,
            aggregation="additive",
        ),
        sampling=TreeBoostingSamplingSpec(
            subsample=subsample,
            colsample_bytree=colsample_bytree,
        ),
        regularization=TreeBoostingRegularizationSpec(
            max_depth=max_depth,
            min_child_weight=min_child_weight,
            gamma=gamma,
            reg_lambda=reg_lambda,
            reg_alpha=reg_alpha,
        ),
        execution=TreeBoostingExecutionSpec(
            n_jobs=n_jobs,
            random_seed=random_seed,
        ),
        task_head=TreeBoostingTaskHeadSpec(
            task="point",
            objective_family="regression",
            outputs=("mean",),
            uncertainty_mode=uncertainty_mode,
        ),
        metadata={} if metadata is None else dict(metadata),
    )


def build_unified_tree_boosting_family_spec(
    *,
    trainer_key: str = TREE_BOOSTING_FORMAL_PRESET_KEY,
    n_estimators: int = 400,
    max_depth: int = 6,
    learning_rate: float = 0.05,
    subsample: float = 0.9,
    colsample_bytree: float = 0.9,
    min_child_weight: float = 1.0,
    gamma: float = 0.0,
    reg_lambda: float = 1.0,
    reg_alpha: float = 0.0,
    objective: str = "reg:squarederror",
    tree_method: str = "hist",
    n_jobs: int = -1,
    random_seed: int = 42,
    verbosity: int = 0,
    uncertainty_mode: str = "residual_std",
    metadata: Mapping[str, Any] | None = None,
) -> TreeBoostingTrainerFamilySpec:
    return build_xgboost_family_spec(
        trainer_key=trainer_key,
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        min_child_weight=min_child_weight,
        gamma=gamma,
        reg_lambda=reg_lambda,
        reg_alpha=reg_alpha,
        objective=objective,
        tree_method=tree_method,
        n_jobs=n_jobs,
        random_seed=random_seed,
        verbosity=verbosity,
        uncertainty_mode=uncertainty_mode,
        metadata={
            "preset_kind": "formal_family",
            "surface_status": "formal",
            "route_family": "tree_boosting",
            **dict(metadata or {}),
        },
    )


def tree_boosting_route_registry() -> tuple[FamilyRouteSpec, ...]:
    return (
        FamilyRouteSpec(
            family_key=TREE_BOOSTING_FORMAL_PRESET_KEY,
            route_key="xgboost",
            match_fields={
                "backend.backend": ("xgboost",),
                "backend.booster": ("gbtree",),
                "task_head.task": ("point",),
            },
            status="stable",
            summary="XGBoost route for the tree_boosting family.",
        ),
    )


def resolve_tree_boosting_route_spec(
    family_spec: TreeBoostingTrainerFamilySpec | Mapping[str, Any] | None,
) -> FamilyRouteSpec:
    spec = coerce_tree_boosting_family_spec(family_spec, trainer_key=TREE_BOOSTING_FORMAL_PRESET_KEY)
    return resolve_family_route_spec(
        tree_boosting_route_registry(),
        spec,
        family_key=TREE_BOOSTING_FORMAL_PRESET_KEY,
    )


def resolve_tree_boosting_router_target(
    family_spec: TreeBoostingTrainerFamilySpec | Mapping[str, Any] | None,
) -> str:
    return resolve_tree_boosting_route_spec(family_spec).route_key


def tree_boosting_surface_contract() -> dict[str, Any]:
    return {
        "formal_preset": TREE_BOOSTING_FORMAL_PRESET_KEY,
        "route_keys": TREE_BOOSTING_ROUTE_KEYS,
        "route_registry": serialize_family_route_registry(tree_boosting_route_registry()),
        "surface_status": {
            TREE_BOOSTING_FORMAL_PRESET_KEY: "formal",
            "xgboost": "route_target",
        },
    }


def coerce_tree_boosting_family_spec(
    value: TreeBoostingTrainerFamilySpec | Mapping[str, Any] | None,
    *,
    trainer_key: str = "xgboost",
) -> TreeBoostingTrainerFamilySpec:
    if value is None:
        if _normalize_name(trainer_key, "xgboost") == TREE_BOOSTING_FORMAL_PRESET_KEY:
            return build_unified_tree_boosting_family_spec(trainer_key=trainer_key)
        return build_xgboost_family_spec(trainer_key=trainer_key)
    if isinstance(value, TreeBoostingTrainerFamilySpec):
        return value
    raw = dict(value)
    backend = coerce_tree_boosting_backend_spec(raw.get("backend"))
    boosting = coerce_tree_boosting_ensemble_spec(raw.get("boosting"))
    sampling = coerce_tree_boosting_sampling_spec(raw.get("sampling"))
    regularization = coerce_tree_boosting_regularization_spec(raw.get("regularization"))
    execution = coerce_tree_boosting_execution_spec(raw.get("execution"))
    task_head = coerce_tree_boosting_task_head_spec(raw.get("task_head"))
    return TreeBoostingTrainerFamilySpec(
        trainer_key=str(raw.get("trainer_key", trainer_key)),
        backend=backend,
        boosting=boosting,
        sampling=sampling,
        regularization=regularization,
        execution=execution,
        task_head=task_head,
        metadata=dict(raw.get("metadata", {}) or {}),
    )


__all__ = [
    "TREE_BOOSTING_FORMAL_PRESET_KEY",
    "TREE_BOOSTING_ROUTE_KEYS",
    "TreeBoostingBackendSpec",
    "TreeBoostingEnsembleSpec",
    "TreeBoostingExecutionSpec",
    "TreeBoostingRegularizationSpec",
    "TreeBoostingSamplingSpec",
    "TreeBoostingTaskHeadSpec",
    "TreeBoostingTrainerFamilySpec",
    "build_unified_tree_boosting_family_spec",
    "build_xgboost_family_spec",
    "coerce_tree_boosting_backend_spec",
    "coerce_tree_boosting_ensemble_spec",
    "coerce_tree_boosting_execution_spec",
    "coerce_tree_boosting_family_spec",
    "coerce_tree_boosting_regularization_spec",
    "coerce_tree_boosting_sampling_spec",
    "coerce_tree_boosting_task_head_spec",
    "resolve_tree_boosting_route_spec",
    "resolve_tree_boosting_router_target",
    "tree_boosting_route_registry",
    "tree_boosting_surface_contract",
]
