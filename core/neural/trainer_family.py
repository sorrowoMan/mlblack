from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Mapping

from core.common.family_router import FamilyRouteSpec, resolve_family_route_spec, serialize_family_route_registry
from core.mechanisms import MechanismProtocolBase, build_neural_family_mechanism_bindings, serialize_family_bindings

NEURAL_FORMAL_PRESET_KEY: str = "neural"
NEURAL_ROUTE_KEYS: tuple[str, ...] = ("mlp_torch", "sklearn_mlp")


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
class NeuralBackendSpec:
    parameter_backend: str = "pytorch"
    runtime_backend: str = "torch"
    trainer_kind: str = "mlp"
    continuation_mode: str = "checkpoint_resume"
    trainer_state_enabled: bool = True
    supports_resume: bool = True
    supports_warm_start: bool = False
    supports_incremental: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "parameter_backend", _normalize_name(self.parameter_backend, "pytorch"))
        object.__setattr__(self, "runtime_backend", _normalize_name(self.runtime_backend, "torch"))
        object.__setattr__(self, "trainer_kind", _normalize_name(self.trainer_kind, "mlp"))
        object.__setattr__(self, "continuation_mode", _normalize_name(self.continuation_mode, "checkpoint_resume"))
        object.__setattr__(self, "trainer_state_enabled", bool(self.trainer_state_enabled))
        object.__setattr__(self, "supports_resume", bool(self.supports_resume))
        object.__setattr__(self, "supports_warm_start", bool(self.supports_warm_start))
        object.__setattr__(self, "supports_incremental", bool(self.supports_incremental))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def as_dict(self) -> dict[str, Any]:
        return {
            "parameter_backend": self.parameter_backend,
            "runtime_backend": self.runtime_backend,
            "trainer_kind": self.trainer_kind,
            "continuation_mode": self.continuation_mode,
            "trainer_state_enabled": bool(self.trainer_state_enabled),
            "supports_resume": bool(self.supports_resume),
            "supports_warm_start": bool(self.supports_warm_start),
            "supports_incremental": bool(self.supports_incremental),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class NeuralBackboneSpec:
    hidden_layers: tuple[int, ...] = (128, 64)
    activation: str = "relu"
    dropout: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        layers = tuple(max(1, int(v)) for v in tuple(self.hidden_layers) if int(v) > 0) or (128, 64)
        object.__setattr__(self, "hidden_layers", layers)
        object.__setattr__(self, "activation", _normalize_name(self.activation, "relu"))
        object.__setattr__(self, "dropout", max(0.0, float(self.dropout)))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def as_dict(self) -> dict[str, Any]:
        return {
            "hidden_layers": tuple(int(v) for v in self.hidden_layers),
            "activation": self.activation,
            "dropout": float(self.dropout),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class NeuralOptimizationSpec:
    objective: str = "mse"
    optimizer: str | None = "adamw"
    optimizer_params: Mapping[str, Any] = field(default_factory=dict)
    solver: str | None = None
    lr: float | None = 1e-3
    weight_decay: float | None = 1e-4
    alpha: float | None = None
    learning_rate_init: float | None = None
    max_steps: int = 120
    tol: float | None = None
    n_iter_no_change: int | None = None
    early_stopping: bool = True
    early_stop_patience: int | None = 20
    early_stop_min_delta: float | None = 1e-6
    random_seed: int = 42
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "objective", _normalize_name(self.objective, "mse"))
        object.__setattr__(self, "optimizer", None if self.optimizer is None else _normalize_name(self.optimizer, "adamw"))
        object.__setattr__(self, "optimizer_params", dict(self.optimizer_params))
        object.__setattr__(self, "solver", None if self.solver is None else _normalize_name(self.solver, "adam"))
        object.__setattr__(self, "lr", None if self.lr is None else max(0.0, float(self.lr)))
        object.__setattr__(self, "weight_decay", None if self.weight_decay is None else max(0.0, float(self.weight_decay)))
        object.__setattr__(self, "alpha", None if self.alpha is None else max(0.0, float(self.alpha)))
        object.__setattr__(
            self,
            "learning_rate_init",
            None if self.learning_rate_init is None else max(0.0, float(self.learning_rate_init)),
        )
        object.__setattr__(self, "max_steps", max(1, int(self.max_steps)))
        object.__setattr__(self, "tol", None if self.tol is None else max(0.0, float(self.tol)))
        object.__setattr__(
            self,
            "n_iter_no_change",
            None if self.n_iter_no_change is None else max(1, int(self.n_iter_no_change)),
        )
        object.__setattr__(self, "early_stopping", bool(self.early_stopping))
        object.__setattr__(
            self,
            "early_stop_patience",
            None if self.early_stop_patience is None else max(1, int(self.early_stop_patience)),
        )
        object.__setattr__(
            self,
            "early_stop_min_delta",
            None if self.early_stop_min_delta is None else max(0.0, float(self.early_stop_min_delta)),
        )
        object.__setattr__(self, "random_seed", int(self.random_seed))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def as_dict(self) -> dict[str, Any]:
        return {
            "objective": self.objective,
            "optimizer": self.optimizer,
            "optimizer_params": dict(self.optimizer_params),
            "solver": self.solver,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "alpha": self.alpha,
            "learning_rate_init": self.learning_rate_init,
            "max_steps": int(self.max_steps),
            "tol": self.tol,
            "n_iter_no_change": self.n_iter_no_change,
            "early_stopping": bool(self.early_stopping),
            "early_stop_patience": self.early_stop_patience,
            "early_stop_min_delta": self.early_stop_min_delta,
            "random_seed": int(self.random_seed),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class NeuralBatchingSpec:
    batch_size: int | str = 64
    shuffle: bool = True
    drop_last: bool = False
    num_workers: int = 0
    pin_memory: bool = False
    val_ratio: float | None = 0.15
    validation_fraction: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        batch_size = self.batch_size
        if isinstance(batch_size, str):
            batch_size = str(batch_size)
        else:
            batch_size = max(1, int(batch_size))
        object.__setattr__(self, "batch_size", batch_size)
        object.__setattr__(self, "shuffle", bool(self.shuffle))
        object.__setattr__(self, "drop_last", bool(self.drop_last))
        object.__setattr__(self, "num_workers", max(0, int(self.num_workers)))
        object.__setattr__(self, "pin_memory", bool(self.pin_memory))
        object.__setattr__(self, "val_ratio", None if self.val_ratio is None else max(0.0, float(self.val_ratio)))
        object.__setattr__(
            self,
            "validation_fraction",
            None if self.validation_fraction is None else max(0.0, float(self.validation_fraction)),
        )
        object.__setattr__(self, "metadata", dict(self.metadata))

    def as_dict(self) -> dict[str, Any]:
        return {
            "batch_size": self.batch_size,
            "shuffle": bool(self.shuffle),
            "drop_last": bool(self.drop_last),
            "num_workers": int(self.num_workers),
            "pin_memory": bool(self.pin_memory),
            "val_ratio": self.val_ratio,
            "validation_fraction": self.validation_fraction,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class NeuralTaskHeadSpec:
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
class NeuralTrainerFamilySpec:
    trainer_key: str = "mlp_torch"
    backend: NeuralBackendSpec = field(default_factory=NeuralBackendSpec)
    backbone: NeuralBackboneSpec = field(default_factory=NeuralBackboneSpec)
    optimization: NeuralOptimizationSpec = field(default_factory=NeuralOptimizationSpec)
    batching: NeuralBatchingSpec = field(default_factory=NeuralBatchingSpec)
    task_head: NeuralTaskHeadSpec = field(default_factory=NeuralTaskHeadSpec)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "trainer_key", _normalize_name(self.trainer_key, "mlp_torch"))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def as_dict(self) -> dict[str, Any]:
        return {
            "trainer_key": self.trainer_key,
            "backend": self.backend.as_dict(),
            "backbone": self.backbone.as_dict(),
            "optimization": self.optimization.as_dict(),
            "batching": self.batching.as_dict(),
            "task_head": self.task_head.as_dict(),
            "metadata": dict(self.metadata),
        }

    def mechanism_bindings(self) -> tuple[MechanismProtocolBase, ...]:
        return build_neural_family_mechanism_bindings()

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

        optimization = dict(payload.get("optimization", {}) or {})
        for key in (
            "optimizer_params",
            "lr",
            "weight_decay",
            "alpha",
            "learning_rate_init",
            "max_steps",
            "tol",
            "n_iter_no_change",
            "early_stop_patience",
            "early_stop_min_delta",
            "random_seed",
        ):
            optimization.pop(key, None)
        payload["optimization"] = optimization

        batching = dict(payload.get("batching", {}) or {})
        for key in ("batch_size", "shuffle", "drop_last", "num_workers", "pin_memory", "val_ratio", "validation_fraction"):
            batching.pop(key, None)
        payload["batching"] = batching
        return _stable_hash(payload)


def coerce_neural_backend_spec(
    value: NeuralBackendSpec | Mapping[str, Any] | None,
    *,
    default: NeuralBackendSpec | Mapping[str, Any] | None = None,
) -> NeuralBackendSpec:
    if value is None:
        value = NeuralBackendSpec() if default is None else default
    if isinstance(value, NeuralBackendSpec):
        return value
    return NeuralBackendSpec(**dict(value))


def coerce_neural_backbone_spec(
    value: NeuralBackboneSpec | Mapping[str, Any] | None,
    *,
    default: NeuralBackboneSpec | Mapping[str, Any] | None = None,
) -> NeuralBackboneSpec:
    if value is None:
        value = NeuralBackboneSpec() if default is None else default
    if isinstance(value, NeuralBackboneSpec):
        return value
    return NeuralBackboneSpec(**dict(value))


def coerce_neural_optimization_spec(
    value: NeuralOptimizationSpec | Mapping[str, Any] | None,
    *,
    default: NeuralOptimizationSpec | Mapping[str, Any] | None = None,
) -> NeuralOptimizationSpec:
    if value is None:
        value = NeuralOptimizationSpec() if default is None else default
    if isinstance(value, NeuralOptimizationSpec):
        return value
    return NeuralOptimizationSpec(**dict(value))


def coerce_neural_batching_spec(
    value: NeuralBatchingSpec | Mapping[str, Any] | None,
    *,
    default: NeuralBatchingSpec | Mapping[str, Any] | None = None,
) -> NeuralBatchingSpec:
    if value is None:
        value = NeuralBatchingSpec() if default is None else default
    if isinstance(value, NeuralBatchingSpec):
        return value
    return NeuralBatchingSpec(**dict(value))


def coerce_neural_task_head_spec(
    value: NeuralTaskHeadSpec | Mapping[str, Any] | None,
    *,
    default: NeuralTaskHeadSpec | Mapping[str, Any] | None = None,
) -> NeuralTaskHeadSpec:
    if value is None:
        value = NeuralTaskHeadSpec() if default is None else default
    if isinstance(value, NeuralTaskHeadSpec):
        return value
    return NeuralTaskHeadSpec(**dict(value))


def build_torch_mlp_family_spec(
    *,
    trainer_key: str = "mlp_torch",
    hidden_layers: tuple[int, ...] = (128, 64),
    activation: str = "relu",
    dropout: float = 0.0,
    optimizer: str = "adamw",
    objective: str = "mse",
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    epochs: int = 120,
    batch_size: int | str = 64,
    shuffle: bool = True,
    drop_last: bool = False,
    num_workers: int = 0,
    pin_memory: bool = False,
    val_ratio: float = 0.15,
    early_stopping: bool = True,
    early_stop_patience: int = 20,
    early_stop_min_delta: float = 1e-6,
    random_seed: int = 42,
    uncertainty_mode: str = "residual_std",
    metadata: Mapping[str, Any] | None = None,
) -> NeuralTrainerFamilySpec:
    return NeuralTrainerFamilySpec(
        trainer_key=trainer_key,
        backend=NeuralBackendSpec(
            parameter_backend="pytorch",
            runtime_backend="torch",
            trainer_kind="mlp",
            continuation_mode="checkpoint_resume",
            trainer_state_enabled=True,
            supports_resume=True,
            supports_warm_start=False,
            supports_incremental=False,
        ),
        backbone=NeuralBackboneSpec(
            hidden_layers=hidden_layers,
            activation=activation,
            dropout=dropout,
        ),
        optimization=NeuralOptimizationSpec(
            objective=objective,
            optimizer=optimizer,
            optimizer_params={},
            solver=None,
            lr=lr,
            weight_decay=weight_decay,
            alpha=None,
            learning_rate_init=None,
            max_steps=epochs,
            tol=None,
            n_iter_no_change=None,
            early_stopping=early_stopping,
            early_stop_patience=early_stop_patience,
            early_stop_min_delta=early_stop_min_delta,
            random_seed=random_seed,
        ),
        batching=NeuralBatchingSpec(
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers,
            pin_memory=pin_memory,
            val_ratio=val_ratio,
            validation_fraction=None,
        ),
        task_head=NeuralTaskHeadSpec(
            task="point",
            objective_family="regression",
            outputs=("mean",),
            uncertainty_mode=uncertainty_mode,
        ),
        metadata={"preset_kind": "torch_backend", **dict(metadata or {})},
    )


def build_sklearn_mlp_family_spec(
    *,
    trainer_key: str = "sklearn_mlp",
    hidden_layers: tuple[int, ...] = (128, 64),
    activation: str = "relu",
    solver: str = "adam",
    alpha: float = 1e-4,
    learning_rate_init: float = 1e-3,
    max_iter: int = 300,
    tol: float = 1e-4,
    n_iter_no_change: int = 20,
    validation_fraction: float = 0.15,
    early_stopping: bool = True,
    batch_size: int | str = "auto",
    random_seed: int = 42,
    uncertainty_mode: str = "residual_std",
    metadata: Mapping[str, Any] | None = None,
) -> NeuralTrainerFamilySpec:
    return NeuralTrainerFamilySpec(
        trainer_key=trainer_key,
        backend=NeuralBackendSpec(
            parameter_backend="sklearn",
            runtime_backend="scikit-learn",
            trainer_kind="mlp",
            continuation_mode="sklearn_estimator_reuse",
            trainer_state_enabled=True,
            supports_resume=False,
            supports_warm_start=True,
            supports_incremental=False,
        ),
        backbone=NeuralBackboneSpec(
            hidden_layers=hidden_layers,
            activation=activation,
            dropout=0.0,
        ),
        optimization=NeuralOptimizationSpec(
            objective="mse",
            optimizer=None,
            optimizer_params={},
            solver=solver,
            lr=None,
            weight_decay=None,
            alpha=alpha,
            learning_rate_init=learning_rate_init,
            max_steps=max_iter,
            tol=tol,
            n_iter_no_change=n_iter_no_change,
            early_stopping=early_stopping,
            early_stop_patience=None,
            early_stop_min_delta=None,
            random_seed=random_seed,
        ),
        batching=NeuralBatchingSpec(
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=0,
            pin_memory=False,
            val_ratio=None,
            validation_fraction=validation_fraction,
        ),
        task_head=NeuralTaskHeadSpec(
            task="point",
            objective_family="regression",
            outputs=("mean",),
            uncertainty_mode=uncertainty_mode,
        ),
        metadata={"preset_kind": "sklearn_backend", **dict(metadata or {})},
    )


def build_unified_neural_family_spec(
    *,
    trainer_key: str = NEURAL_FORMAL_PRESET_KEY,
    parameter_backend: str = "pytorch",
    metadata: Mapping[str, Any] | None = None,
) -> NeuralTrainerFamilySpec:
    backend_key = _normalize_name(parameter_backend, "pytorch")
    common_meta = {
        "preset_kind": "formal_family",
        "surface_status": "formal",
        "route_family": "neural",
        **dict(metadata or {}),
    }
    if backend_key in {"sklearn", "scikit-learn", "scikit_learn"}:
        return build_sklearn_mlp_family_spec(
            trainer_key=trainer_key,
            metadata=common_meta,
        )
    return build_torch_mlp_family_spec(
        trainer_key=trainer_key,
        metadata=common_meta,
    )


def neural_route_registry() -> tuple[FamilyRouteSpec, ...]:
    return (
        FamilyRouteSpec(
            family_key=NEURAL_FORMAL_PRESET_KEY,
            route_key="mlp_torch",
            match_fields={
                "backend.parameter_backend": ("pytorch",),
                "backend.runtime_backend": ("torch",),
                "backend.trainer_kind": ("mlp",),
                "task_head.task": ("point",),
            },
            status="stable",
            summary="Torch-backed MLP route for the neural family.",
        ),
        FamilyRouteSpec(
            family_key=NEURAL_FORMAL_PRESET_KEY,
            route_key="sklearn_mlp",
            match_fields={
                "backend.parameter_backend": ("sklearn",),
                "backend.runtime_backend": ("scikit-learn",),
                "backend.trainer_kind": ("mlp",),
                "task_head.task": ("point",),
            },
            status="stable",
            summary="Scikit-learn MLP route for the neural family.",
        ),
    )


def resolve_neural_route_spec(
    family_spec: NeuralTrainerFamilySpec | Mapping[str, Any] | None,
) -> FamilyRouteSpec:
    spec = coerce_neural_family_spec(family_spec, trainer_key=NEURAL_FORMAL_PRESET_KEY)
    return resolve_family_route_spec(
        neural_route_registry(),
        spec,
        family_key=NEURAL_FORMAL_PRESET_KEY,
    )


def resolve_neural_router_target(
    family_spec: NeuralTrainerFamilySpec | Mapping[str, Any] | None,
) -> str:
    return resolve_neural_route_spec(family_spec).route_key


def neural_surface_contract() -> dict[str, Any]:
    return {
        "formal_preset": NEURAL_FORMAL_PRESET_KEY,
        "route_keys": NEURAL_ROUTE_KEYS,
        "route_registry": serialize_family_route_registry(neural_route_registry()),
        "surface_status": {
            NEURAL_FORMAL_PRESET_KEY: "formal",
            "mlp_torch": "route_target",
            "sklearn_mlp": "route_target",
        },
    }


def coerce_neural_family_spec(
    value: NeuralTrainerFamilySpec | Mapping[str, Any] | None,
    *,
    trainer_key: str = "mlp_torch",
) -> NeuralTrainerFamilySpec:
    if value is None:
        if _normalize_name(trainer_key, "mlp_torch") == NEURAL_FORMAL_PRESET_KEY:
            return build_unified_neural_family_spec(trainer_key=trainer_key)
        if str(trainer_key).strip().lower() == "sklearn_mlp":
            return build_sklearn_mlp_family_spec(trainer_key=trainer_key)
        return build_torch_mlp_family_spec(trainer_key=trainer_key)
    if isinstance(value, NeuralTrainerFamilySpec):
        return value

    raw = dict(value)
    default_spec = (
        build_sklearn_mlp_family_spec(trainer_key=trainer_key)
        if str(trainer_key).strip().lower() == "sklearn_mlp"
        else build_unified_neural_family_spec(trainer_key=trainer_key)
        if str(trainer_key).strip().lower() == NEURAL_FORMAL_PRESET_KEY
        else build_torch_mlp_family_spec(trainer_key=trainer_key)
    )
    return NeuralTrainerFamilySpec(
        trainer_key=str(raw.get("trainer_key", trainer_key)),
        backend=coerce_neural_backend_spec(raw.get("backend"), default=default_spec.backend),
        backbone=coerce_neural_backbone_spec(raw.get("backbone"), default=default_spec.backbone),
        optimization=coerce_neural_optimization_spec(raw.get("optimization"), default=default_spec.optimization),
        batching=coerce_neural_batching_spec(raw.get("batching"), default=default_spec.batching),
        task_head=coerce_neural_task_head_spec(raw.get("task_head"), default=default_spec.task_head),
        metadata=dict(raw.get("metadata", {})),
    )


__all__ = [
    "NEURAL_FORMAL_PRESET_KEY",
    "NEURAL_ROUTE_KEYS",
    "NeuralBackendSpec",
    "NeuralBackboneSpec",
    "NeuralOptimizationSpec",
    "NeuralBatchingSpec",
    "NeuralTaskHeadSpec",
    "NeuralTrainerFamilySpec",
    "build_unified_neural_family_spec",
    "build_torch_mlp_family_spec",
    "build_sklearn_mlp_family_spec",
    "coerce_neural_backend_spec",
    "coerce_neural_backbone_spec",
    "coerce_neural_optimization_spec",
    "coerce_neural_batching_spec",
    "coerce_neural_task_head_spec",
    "coerce_neural_family_spec",
    "neural_route_registry",
    "neural_surface_contract",
    "resolve_neural_route_spec",
    "resolve_neural_router_target",
]
