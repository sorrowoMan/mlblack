from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Mapping

from core.common.family_router import FamilyRouteSpec, resolve_family_route_spec, serialize_family_route_registry
from core.mechanisms import (
    MechanismProtocolBase,
    build_adaboost_mechanism_bindings,
    build_bagging_mechanism_bindings,
    build_extra_trees_mechanism_bindings,
    build_random_forest_mechanism_bindings,
    build_tree_boosting_family_mechanism_bindings,
    serialize_family_bindings,
)

TREE_ENSEMBLE_FORMAL_PRESET_KEY: str = "tree_ensemble"
TREE_ENSEMBLE_ROUTE_KEYS: tuple[str, ...] = ("random_forest", "extra_trees", "bagging", "adaboost")


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
class TreeEnsembleSpec:
    ensemble_kind: str = "random_forest"
    backend: str = "sklearn"
    n_estimators: int = 200
    aggregation: str = "mean"
    oob_score: bool = False
    n_jobs: int = -1
    random_seed: int = 42
    warm_start_enabled: bool = True
    trainer_state_enabled: bool = True
    supports_resume: bool = True
    supports_warm_start: bool = True
    supports_incremental: bool = True
    learning_rate: float = 1.0
    loss: str = "linear"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "ensemble_kind", _normalize_name(self.ensemble_kind, "random_forest"))
        object.__setattr__(self, "backend", _normalize_name(self.backend, "sklearn"))
        object.__setattr__(self, "n_estimators", max(1, int(self.n_estimators)))
        object.__setattr__(self, "aggregation", _normalize_name(self.aggregation, "mean"))
        object.__setattr__(self, "oob_score", bool(self.oob_score))
        object.__setattr__(self, "n_jobs", int(self.n_jobs))
        object.__setattr__(self, "random_seed", int(self.random_seed))
        object.__setattr__(self, "warm_start_enabled", bool(self.warm_start_enabled))
        object.__setattr__(self, "trainer_state_enabled", bool(self.trainer_state_enabled))
        object.__setattr__(self, "supports_resume", bool(self.supports_resume))
        object.__setattr__(self, "supports_warm_start", bool(self.supports_warm_start))
        object.__setattr__(self, "supports_incremental", bool(self.supports_incremental))
        object.__setattr__(self, "learning_rate", max(0.0, float(self.learning_rate)))
        object.__setattr__(self, "loss", _normalize_name(self.loss, "linear"))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def as_dict(self) -> dict[str, Any]:
        return {
            "ensemble_kind": self.ensemble_kind,
            "backend": self.backend,
            "n_estimators": int(self.n_estimators),
            "aggregation": self.aggregation,
            "oob_score": bool(self.oob_score),
            "n_jobs": int(self.n_jobs),
            "random_seed": int(self.random_seed),
            "warm_start_enabled": bool(self.warm_start_enabled),
            "trainer_state_enabled": bool(self.trainer_state_enabled),
            "supports_resume": bool(self.supports_resume),
            "supports_warm_start": bool(self.supports_warm_start),
            "supports_incremental": bool(self.supports_incremental),
            "learning_rate": float(self.learning_rate),
            "loss": self.loss,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class TreeSamplingSpec:
    bootstrap: bool = True
    bootstrap_features: bool = False
    max_samples: int | float | None = None
    max_features: int | float | str | None = 1.0
    class_weight: Any | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "bootstrap", bool(self.bootstrap))
        object.__setattr__(self, "bootstrap_features", bool(self.bootstrap_features))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def as_dict(self) -> dict[str, Any]:
        return {
            "bootstrap": bool(self.bootstrap),
            "bootstrap_features": bool(self.bootstrap_features),
            "max_samples": self.max_samples,
            "max_features": self.max_features,
            "class_weight": self.class_weight,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class TreeSplitSpec:
    criterion: str = "squared_error"
    splitter: str = "best"
    min_impurity_decrease: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "criterion", _normalize_name(self.criterion, "squared_error"))
        object.__setattr__(self, "splitter", _normalize_name(self.splitter, "best"))
        object.__setattr__(self, "min_impurity_decrease", max(0.0, float(self.min_impurity_decrease)))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def as_dict(self) -> dict[str, Any]:
        return {
            "criterion": self.criterion,
            "splitter": self.splitter,
            "min_impurity_decrease": float(self.min_impurity_decrease),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class TreeRegularizationSpec:
    max_depth: int | None = None
    min_samples_split: int | float = 2
    min_samples_leaf: int | float = 1
    min_weight_fraction_leaf: float = 0.0
    max_leaf_nodes: int | None = None
    ccp_alpha: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "max_depth", None if self.max_depth is None else max(1, int(self.max_depth)))
        object.__setattr__(self, "min_samples_split", self.min_samples_split)
        object.__setattr__(self, "min_samples_leaf", self.min_samples_leaf)
        object.__setattr__(self, "min_weight_fraction_leaf", max(0.0, float(self.min_weight_fraction_leaf)))
        object.__setattr__(self, "max_leaf_nodes", None if self.max_leaf_nodes is None else max(2, int(self.max_leaf_nodes)))
        object.__setattr__(self, "ccp_alpha", max(0.0, float(self.ccp_alpha)))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def as_dict(self) -> dict[str, Any]:
        return {
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "min_weight_fraction_leaf": float(self.min_weight_fraction_leaf),
            "max_leaf_nodes": self.max_leaf_nodes,
            "ccp_alpha": float(self.ccp_alpha),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class TreeTaskHeadSpec:
    task: str = "point"
    objective_family: str = "regression"
    outputs: tuple[str, ...] = ("mean",)
    uncertainty_mode: str = "ensemble_std"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "task", _normalize_name(self.task, "point"))
        object.__setattr__(self, "objective_family", _normalize_name(self.objective_family, "regression"))
        outs = tuple(str(v).strip().lower() for v in tuple(self.outputs) if str(v).strip()) or ("mean",)
        object.__setattr__(self, "outputs", outs)
        object.__setattr__(self, "uncertainty_mode", _normalize_name(self.uncertainty_mode, "ensemble_std"))
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
class TreeTrainerFamilySpec:
    trainer_key: str = "random_forest"
    ensemble: TreeEnsembleSpec = field(default_factory=TreeEnsembleSpec)
    sampling: TreeSamplingSpec = field(default_factory=TreeSamplingSpec)
    splitter: TreeSplitSpec = field(default_factory=TreeSplitSpec)
    regularization: TreeRegularizationSpec = field(default_factory=TreeRegularizationSpec)
    task_head: TreeTaskHeadSpec = field(default_factory=TreeTaskHeadSpec)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "trainer_key", _normalize_name(self.trainer_key, "random_forest"))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def as_dict(self) -> dict[str, Any]:
        return {
            "trainer_key": self.trainer_key,
            "ensemble": self.ensemble.as_dict(),
            "sampling": self.sampling.as_dict(),
            "splitter": self.splitter.as_dict(),
            "regularization": self.regularization.as_dict(),
            "task_head": self.task_head.as_dict(),
            "metadata": dict(self.metadata),
        }

    def mechanism_bindings(self) -> tuple[MechanismProtocolBase, ...]:
        ensemble_kind = str(self.ensemble.ensemble_kind).strip().lower()
        if ensemble_kind in {"random_forest", "extra_trees", "bagging", "bagged_trees"}:
            if ensemble_kind == "extra_trees":
                return build_extra_trees_mechanism_bindings()
            if ensemble_kind in {"bagging", "bagged_trees"}:
                return build_bagging_mechanism_bindings()
            return build_random_forest_mechanism_bindings()
        if ensemble_kind == "adaboost":
            return build_adaboost_mechanism_bindings()
        if ensemble_kind in {"tree_boosting", "gradient_boosting", "gbdt", "xgboost"}:
            return build_tree_boosting_family_mechanism_bindings()
        if "boost" in ensemble_kind:
            return build_tree_boosting_family_mechanism_bindings()
        return build_random_forest_mechanism_bindings()

    def mechanism_binding_payload(self) -> list[dict[str, object]]:
        return serialize_family_bindings(self.mechanism_bindings())

    def description_dict(self) -> dict[str, Any]:
        payload = self.as_dict()
        payload["mechanism_bindings"] = self.mechanism_binding_payload()
        return payload

    def family_signature(self) -> str | None:
        payload = self.as_dict()
        ensemble = dict(payload.get("ensemble", {}) or {})
        for key in (
            "n_estimators",
            "n_jobs",
            "oob_score",
            "warm_start_enabled",
            "trainer_state_enabled",
            "supports_resume",
            "supports_warm_start",
            "supports_incremental",
        ):
            ensemble.pop(key, None)
        payload["ensemble"] = ensemble
        return _stable_hash(payload)


def coerce_tree_ensemble_spec(
    value: TreeEnsembleSpec | Mapping[str, Any] | None,
    *,
    default: TreeEnsembleSpec | Mapping[str, Any] | None = None,
) -> TreeEnsembleSpec:
    if value is None:
        value = TreeEnsembleSpec() if default is None else default
    if isinstance(value, TreeEnsembleSpec):
        return value
    return TreeEnsembleSpec(**dict(value))


def coerce_tree_sampling_spec(
    value: TreeSamplingSpec | Mapping[str, Any] | None,
    *,
    default: TreeSamplingSpec | Mapping[str, Any] | None = None,
) -> TreeSamplingSpec:
    if value is None:
        value = TreeSamplingSpec() if default is None else default
    if isinstance(value, TreeSamplingSpec):
        return value
    return TreeSamplingSpec(**dict(value))


def coerce_tree_split_spec(
    value: TreeSplitSpec | Mapping[str, Any] | None,
    *,
    default: TreeSplitSpec | Mapping[str, Any] | None = None,
) -> TreeSplitSpec:
    if value is None:
        value = TreeSplitSpec() if default is None else default
    if isinstance(value, TreeSplitSpec):
        return value
    return TreeSplitSpec(**dict(value))


def coerce_tree_regularization_spec(
    value: TreeRegularizationSpec | Mapping[str, Any] | None,
    *,
    default: TreeRegularizationSpec | Mapping[str, Any] | None = None,
) -> TreeRegularizationSpec:
    if value is None:
        value = TreeRegularizationSpec() if default is None else default
    if isinstance(value, TreeRegularizationSpec):
        return value
    return TreeRegularizationSpec(**dict(value))


def coerce_tree_task_head_spec(
    value: TreeTaskHeadSpec | Mapping[str, Any] | None,
    *,
    default: TreeTaskHeadSpec | Mapping[str, Any] | None = None,
) -> TreeTaskHeadSpec:
    if value is None:
        value = TreeTaskHeadSpec() if default is None else default
    if isinstance(value, TreeTaskHeadSpec):
        return value
    return TreeTaskHeadSpec(**dict(value))


def build_random_forest_family_spec(
    *,
    trainer_key: str = "random_forest",
    n_estimators: int = 200,
    max_depth: int | None = None,
    max_features: int | float | str | None = 1.0,
    criterion: str = "squared_error",
    bootstrap: bool = True,
    bootstrap_features: bool = False,
    max_samples: int | float | None = None,
    n_jobs: int = -1,
    random_seed: int = 42,
    oob_score: bool = False,
    uncertainty_mode: str = "ensemble_std",
    metadata: Mapping[str, Any] | None = None,
) -> TreeTrainerFamilySpec:
    return TreeTrainerFamilySpec(
        trainer_key=trainer_key,
        ensemble=TreeEnsembleSpec(
            ensemble_kind="random_forest",
            backend="sklearn",
            n_estimators=int(n_estimators),
            oob_score=bool(oob_score),
            n_jobs=int(n_jobs),
            random_seed=int(random_seed),
            warm_start_enabled=True,
            trainer_state_enabled=True,
            supports_resume=True,
            supports_warm_start=True,
            supports_incremental=True,
        ),
        sampling=TreeSamplingSpec(
            bootstrap=bool(bootstrap),
            bootstrap_features=bool(bootstrap_features),
            max_samples=max_samples,
            max_features=max_features,
        ),
        splitter=TreeSplitSpec(
            criterion=criterion,
            splitter="best",
        ),
        regularization=TreeRegularizationSpec(
            max_depth=max_depth,
        ),
        task_head=TreeTaskHeadSpec(
            task="point",
            objective_family="regression",
            outputs=("mean",),
            uncertainty_mode=uncertainty_mode,
        ),
        metadata={} if metadata is None else dict(metadata),
    )


def build_extra_trees_family_spec(
    *,
    trainer_key: str = "extra_trees",
    n_estimators: int = 200,
    max_depth: int | None = None,
    max_features: int | float | str | None = 1.0,
    criterion: str = "squared_error",
    bootstrap: bool = False,
    max_samples: int | float | None = None,
    n_jobs: int = -1,
    random_seed: int = 42,
    oob_score: bool = False,
    uncertainty_mode: str = "ensemble_std",
    metadata: Mapping[str, Any] | None = None,
) -> TreeTrainerFamilySpec:
    return TreeTrainerFamilySpec(
        trainer_key=trainer_key,
        ensemble=TreeEnsembleSpec(
            ensemble_kind="extra_trees",
            backend="sklearn",
            n_estimators=int(n_estimators),
            aggregation="mean",
            oob_score=bool(oob_score),
            n_jobs=int(n_jobs),
            random_seed=int(random_seed),
            warm_start_enabled=True,
            trainer_state_enabled=True,
            supports_resume=True,
            supports_warm_start=True,
            supports_incremental=True,
        ),
        sampling=TreeSamplingSpec(
            bootstrap=bool(bootstrap),
            bootstrap_features=False,
            max_samples=max_samples,
            max_features=max_features,
        ),
        splitter=TreeSplitSpec(
            criterion=criterion,
            splitter="random",
        ),
        regularization=TreeRegularizationSpec(
            max_depth=max_depth,
        ),
        task_head=TreeTaskHeadSpec(
            task="point",
            objective_family="regression",
            outputs=("mean",),
            uncertainty_mode=uncertainty_mode,
        ),
        metadata={} if metadata is None else dict(metadata),
    )


def build_bagging_family_spec(
    *,
    trainer_key: str = "bagging",
    n_estimators: int = 200,
    max_depth: int | None = None,
    max_features: int | float | str | None = 1.0,
    criterion: str = "squared_error",
    bootstrap: bool = True,
    bootstrap_features: bool = False,
    max_samples: int | float | None = 1.0,
    n_jobs: int = -1,
    random_seed: int = 42,
    oob_score: bool = False,
    uncertainty_mode: str = "ensemble_std",
    metadata: Mapping[str, Any] | None = None,
) -> TreeTrainerFamilySpec:
    return TreeTrainerFamilySpec(
        trainer_key=trainer_key,
        ensemble=TreeEnsembleSpec(
            ensemble_kind="bagging",
            backend="sklearn",
            n_estimators=int(n_estimators),
            aggregation="mean",
            oob_score=bool(oob_score),
            n_jobs=int(n_jobs),
            random_seed=int(random_seed),
            warm_start_enabled=True,
            trainer_state_enabled=True,
            supports_resume=True,
            supports_warm_start=True,
            supports_incremental=True,
        ),
        sampling=TreeSamplingSpec(
            bootstrap=bool(bootstrap),
            bootstrap_features=bool(bootstrap_features),
            max_samples=max_samples,
            max_features=max_features,
        ),
        splitter=TreeSplitSpec(
            criterion=criterion,
            splitter="best",
        ),
        regularization=TreeRegularizationSpec(
            max_depth=max_depth,
        ),
        task_head=TreeTaskHeadSpec(
            task="point",
            objective_family="regression",
            outputs=("mean",),
            uncertainty_mode=uncertainty_mode,
        ),
        metadata={} if metadata is None else dict(metadata),
    )


def build_adaboost_family_spec(
    *,
    trainer_key: str = "adaboost",
    n_estimators: int = 100,
    max_depth: int | None = 3,
    max_features: int | float | str | None = 1.0,
    criterion: str = "squared_error",
    learning_rate: float = 1.0,
    loss: str = "linear",
    random_seed: int = 42,
    uncertainty_mode: str = "weighted_ensemble_std",
    metadata: Mapping[str, Any] | None = None,
) -> TreeTrainerFamilySpec:
    return TreeTrainerFamilySpec(
        trainer_key=trainer_key,
        ensemble=TreeEnsembleSpec(
            ensemble_kind="adaboost",
            backend="sklearn",
            n_estimators=int(n_estimators),
            aggregation="weighted_additive",
            oob_score=False,
            n_jobs=1,
            random_seed=int(random_seed),
            warm_start_enabled=False,
            trainer_state_enabled=True,
            supports_resume=False,
            supports_warm_start=False,
            supports_incremental=False,
            learning_rate=float(learning_rate),
            loss=str(loss),
        ),
        sampling=TreeSamplingSpec(
            bootstrap=False,
            bootstrap_features=False,
            max_samples=None,
            max_features=max_features,
        ),
        splitter=TreeSplitSpec(
            criterion=criterion,
            splitter="best",
        ),
        regularization=TreeRegularizationSpec(
            max_depth=max_depth,
        ),
        task_head=TreeTaskHeadSpec(
            task="point",
            objective_family="regression",
            outputs=("mean",),
            uncertainty_mode=uncertainty_mode,
        ),
        metadata={} if metadata is None else dict(metadata),
    )


def build_unified_tree_ensemble_family_spec(
    *,
    trainer_key: str = TREE_ENSEMBLE_FORMAL_PRESET_KEY,
    ensemble_kind: str = "random_forest",
    n_estimators: int = 200,
    max_depth: int | None = None,
    max_features: int | float | str | None = 1.0,
    criterion: str = "squared_error",
    bootstrap: bool | None = None,
    bootstrap_features: bool = False,
    max_samples: int | float | None = None,
    n_jobs: int = -1,
    random_seed: int = 42,
    oob_score: bool = False,
    learning_rate: float = 1.0,
    loss: str = "linear",
    uncertainty_mode: str = "ensemble_std",
    metadata: Mapping[str, Any] | None = None,
) -> TreeTrainerFamilySpec:
    route_key = _normalize_name(ensemble_kind, "random_forest")
    route_meta = {
        "preset_kind": "formal_family",
        "surface_status": "formal",
        "route_family": TREE_ENSEMBLE_FORMAL_PRESET_KEY,
        **dict(metadata or {}),
    }
    if route_key == "extra_trees":
        return build_extra_trees_family_spec(
            trainer_key=trainer_key,
            n_estimators=n_estimators,
            max_depth=max_depth,
            max_features=max_features,
            criterion=criterion,
            bootstrap=bool(False if bootstrap is None else bootstrap),
            max_samples=max_samples,
            n_jobs=n_jobs,
            random_seed=random_seed,
            oob_score=oob_score,
            uncertainty_mode=uncertainty_mode,
            metadata=route_meta,
        )
    if route_key == "bagging":
        return build_bagging_family_spec(
            trainer_key=trainer_key,
            n_estimators=n_estimators,
            max_depth=max_depth,
            max_features=max_features,
            criterion=criterion,
            bootstrap=bool(True if bootstrap is None else bootstrap),
            bootstrap_features=bootstrap_features,
            max_samples=(1.0 if max_samples is None else max_samples),
            n_jobs=n_jobs,
            random_seed=random_seed,
            oob_score=oob_score,
            uncertainty_mode=uncertainty_mode,
            metadata=route_meta,
        )
    if route_key == "adaboost":
        return build_adaboost_family_spec(
            trainer_key=trainer_key,
            n_estimators=n_estimators,
            max_depth=(3 if max_depth is None else max_depth),
            max_features=max_features,
            criterion=criterion,
            learning_rate=learning_rate,
            loss=loss,
            random_seed=random_seed,
            uncertainty_mode="weighted_ensemble_std" if uncertainty_mode == "ensemble_std" else uncertainty_mode,
            metadata=route_meta,
        )
    return build_random_forest_family_spec(
        trainer_key=trainer_key,
        n_estimators=n_estimators,
        max_depth=max_depth,
        max_features=max_features,
        criterion=criterion,
        bootstrap=bool(True if bootstrap is None else bootstrap),
        bootstrap_features=bootstrap_features,
        max_samples=max_samples,
        n_jobs=n_jobs,
        random_seed=random_seed,
        oob_score=oob_score,
        uncertainty_mode=uncertainty_mode,
        metadata=route_meta,
    )


def tree_ensemble_route_registry() -> tuple[FamilyRouteSpec, ...]:
    return (
        FamilyRouteSpec(
            family_key=TREE_ENSEMBLE_FORMAL_PRESET_KEY,
            route_key="random_forest",
            match_fields={
                "ensemble.ensemble_kind": ("random_forest",),
                "task_head.task": ("point",),
            },
            status="stable",
            summary="Random forest route for the tree_ensemble family.",
        ),
        FamilyRouteSpec(
            family_key=TREE_ENSEMBLE_FORMAL_PRESET_KEY,
            route_key="extra_trees",
            match_fields={
                "ensemble.ensemble_kind": ("extra_trees",),
                "task_head.task": ("point",),
            },
            status="stable",
            summary="Extra-trees route for the tree_ensemble family.",
        ),
        FamilyRouteSpec(
            family_key=TREE_ENSEMBLE_FORMAL_PRESET_KEY,
            route_key="bagging",
            match_fields={
                "ensemble.ensemble_kind": ("bagging", "bagged_trees"),
                "task_head.task": ("point",),
            },
            status="stable",
            summary="Bagging route for the tree_ensemble family.",
        ),
        FamilyRouteSpec(
            family_key=TREE_ENSEMBLE_FORMAL_PRESET_KEY,
            route_key="adaboost",
            match_fields={
                "ensemble.ensemble_kind": ("adaboost",),
                "task_head.task": ("point",),
            },
            status="stable",
            summary="AdaBoost route for the tree_ensemble family.",
        ),
    )


def resolve_tree_ensemble_route_spec(
    family_spec: TreeTrainerFamilySpec | Mapping[str, Any] | None,
) -> FamilyRouteSpec:
    spec = coerce_tree_family_spec(family_spec, trainer_key=TREE_ENSEMBLE_FORMAL_PRESET_KEY)
    return resolve_family_route_spec(
        tree_ensemble_route_registry(),
        spec,
        family_key=TREE_ENSEMBLE_FORMAL_PRESET_KEY,
    )


def resolve_tree_ensemble_router_target(
    family_spec: TreeTrainerFamilySpec | Mapping[str, Any] | None,
) -> str:
    return resolve_tree_ensemble_route_spec(family_spec).route_key


def tree_ensemble_surface_contract() -> dict[str, Any]:
    return {
        "formal_preset": TREE_ENSEMBLE_FORMAL_PRESET_KEY,
        "route_keys": TREE_ENSEMBLE_ROUTE_KEYS,
        "route_registry": serialize_family_route_registry(tree_ensemble_route_registry()),
        "surface_status": {
            TREE_ENSEMBLE_FORMAL_PRESET_KEY: "formal",
            "random_forest": "route_target",
            "extra_trees": "route_target",
            "bagging": "route_target",
            "adaboost": "route_target",
        },
    }


def coerce_tree_family_spec(
    value: TreeTrainerFamilySpec | Mapping[str, Any] | None,
    *,
    trainer_key: str = "random_forest",
) -> TreeTrainerFamilySpec:
    if value is None:
        if _normalize_name(trainer_key, "random_forest") == TREE_ENSEMBLE_FORMAL_PRESET_KEY:
            return build_unified_tree_ensemble_family_spec(trainer_key=trainer_key)
        return build_random_forest_family_spec(trainer_key=trainer_key)
    if isinstance(value, TreeTrainerFamilySpec):
        return value

    raw = dict(value)
    ensemble = coerce_tree_ensemble_spec(raw.get("ensemble"))
    sampling = coerce_tree_sampling_spec(raw.get("sampling"))
    splitter = coerce_tree_split_spec(raw.get("splitter"))
    regularization = coerce_tree_regularization_spec(raw.get("regularization"))
    task_head = coerce_tree_task_head_spec(raw.get("task_head"))
    return TreeTrainerFamilySpec(
        trainer_key=str(raw.get("trainer_key", trainer_key)),
        ensemble=ensemble,
        sampling=sampling,
        splitter=splitter,
        regularization=regularization,
        task_head=task_head,
        metadata=dict(raw.get("metadata", {})),
    )


__all__ = [
    "TREE_ENSEMBLE_FORMAL_PRESET_KEY",
    "TREE_ENSEMBLE_ROUTE_KEYS",
    "TreeEnsembleSpec",
    "TreeSamplingSpec",
    "TreeSplitSpec",
    "TreeRegularizationSpec",
    "TreeTaskHeadSpec",
    "TreeTrainerFamilySpec",
    "build_adaboost_family_spec",
    "build_bagging_family_spec",
    "build_extra_trees_family_spec",
    "build_random_forest_family_spec",
    "build_unified_tree_ensemble_family_spec",
    "coerce_tree_ensemble_spec",
    "coerce_tree_sampling_spec",
    "coerce_tree_split_spec",
    "coerce_tree_regularization_spec",
    "coerce_tree_task_head_spec",
    "coerce_tree_family_spec",
    "resolve_tree_ensemble_route_spec",
    "resolve_tree_ensemble_router_target",
    "tree_ensemble_route_registry",
    "tree_ensemble_surface_contract",
]
