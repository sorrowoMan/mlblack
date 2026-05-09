from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Mapping

from core.common.family_router import FamilyRouteSpec, resolve_family_route_spec, serialize_family_route_registry
from core.mechanisms import MechanismProtocolBase, build_linear_family_mechanism_bindings, serialize_family_bindings

LINEAR_FORMAL_PRESET_KEY: str = "linear"
LINEAR_ROUTE_KEYS: tuple[str, ...] = ("ridge",)


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
class LinearBackendSpec:
    parameter_backend: str = "closed_form"
    runtime_backend: str = "numpy"
    solver_kind: str = "ridge"
    continuation_mode: str = "closed_form_refit"
    trainer_state_enabled: bool = True
    supports_resume: bool = True
    supports_warm_start: bool = True
    supports_incremental: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "parameter_backend", _normalize_name(self.parameter_backend, "closed_form"))
        object.__setattr__(self, "runtime_backend", _normalize_name(self.runtime_backend, "numpy"))
        object.__setattr__(self, "solver_kind", _normalize_name(self.solver_kind, "ridge"))
        object.__setattr__(self, "continuation_mode", _normalize_name(self.continuation_mode, "closed_form_refit"))
        object.__setattr__(self, "trainer_state_enabled", bool(self.trainer_state_enabled))
        object.__setattr__(self, "supports_resume", bool(self.supports_resume))
        object.__setattr__(self, "supports_warm_start", bool(self.supports_warm_start))
        object.__setattr__(self, "supports_incremental", bool(self.supports_incremental))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def as_dict(self) -> dict[str, Any]:
        return {
            "parameter_backend": self.parameter_backend,
            "runtime_backend": self.runtime_backend,
            "solver_kind": self.solver_kind,
            "continuation_mode": self.continuation_mode,
            "trainer_state_enabled": bool(self.trainer_state_enabled),
            "supports_resume": bool(self.supports_resume),
            "supports_warm_start": bool(self.supports_warm_start),
            "supports_incremental": bool(self.supports_incremental),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class LinearFunctionClassSpec:
    basis: str = "affine"
    fit_intercept: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "basis", _normalize_name(self.basis, "affine"))
        object.__setattr__(self, "fit_intercept", bool(self.fit_intercept))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def as_dict(self) -> dict[str, Any]:
        return {
            "basis": self.basis,
            "fit_intercept": bool(self.fit_intercept),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class LinearRegularizationSpec:
    penalty: str = "l2"
    l2: float = 1.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "penalty", _normalize_name(self.penalty, "l2"))
        object.__setattr__(self, "l2", max(0.0, float(self.l2)))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def as_dict(self) -> dict[str, Any]:
        return {
            "penalty": self.penalty,
            "l2": float(self.l2),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class LinearTaskHeadSpec:
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
class LinearTrainerFamilySpec:
    trainer_key: str = "ridge"
    backend: LinearBackendSpec = field(default_factory=LinearBackendSpec)
    function_class: LinearFunctionClassSpec = field(default_factory=LinearFunctionClassSpec)
    regularization: LinearRegularizationSpec = field(default_factory=LinearRegularizationSpec)
    task_head: LinearTaskHeadSpec = field(default_factory=LinearTaskHeadSpec)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "trainer_key", _normalize_name(self.trainer_key, "ridge"))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def as_dict(self) -> dict[str, Any]:
        return {
            "trainer_key": self.trainer_key,
            "backend": self.backend.as_dict(),
            "function_class": self.function_class.as_dict(),
            "regularization": self.regularization.as_dict(),
            "task_head": self.task_head.as_dict(),
            "metadata": dict(self.metadata),
        }

    def mechanism_bindings(self) -> tuple[MechanismProtocolBase, ...]:
        return build_linear_family_mechanism_bindings()

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

        regularization = dict(payload.get("regularization", {}) or {})
        regularization.pop("l2", None)
        payload["regularization"] = regularization
        return _stable_hash(payload)


def coerce_linear_backend_spec(
    value: LinearBackendSpec | Mapping[str, Any] | None,
    *,
    default: LinearBackendSpec | Mapping[str, Any] | None = None,
) -> LinearBackendSpec:
    if value is None:
        value = LinearBackendSpec() if default is None else default
    if isinstance(value, LinearBackendSpec):
        return value
    return LinearBackendSpec(**dict(value))


def coerce_linear_function_class_spec(
    value: LinearFunctionClassSpec | Mapping[str, Any] | None,
    *,
    default: LinearFunctionClassSpec | Mapping[str, Any] | None = None,
) -> LinearFunctionClassSpec:
    if value is None:
        value = LinearFunctionClassSpec() if default is None else default
    if isinstance(value, LinearFunctionClassSpec):
        return value
    return LinearFunctionClassSpec(**dict(value))


def coerce_linear_regularization_spec(
    value: LinearRegularizationSpec | Mapping[str, Any] | None,
    *,
    default: LinearRegularizationSpec | Mapping[str, Any] | None = None,
) -> LinearRegularizationSpec:
    if value is None:
        value = LinearRegularizationSpec() if default is None else default
    if isinstance(value, LinearRegularizationSpec):
        return value
    return LinearRegularizationSpec(**dict(value))


def coerce_linear_task_head_spec(
    value: LinearTaskHeadSpec | Mapping[str, Any] | None,
    *,
    default: LinearTaskHeadSpec | Mapping[str, Any] | None = None,
) -> LinearTaskHeadSpec:
    if value is None:
        value = LinearTaskHeadSpec() if default is None else default
    if isinstance(value, LinearTaskHeadSpec):
        return value
    return LinearTaskHeadSpec(**dict(value))


def build_ridge_family_spec(
    *,
    trainer_key: str = "ridge",
    l2: float = 1.0,
    uncertainty_mode: str = "residual_std",
    metadata: Mapping[str, Any] | None = None,
) -> LinearTrainerFamilySpec:
    return LinearTrainerFamilySpec(
        trainer_key=trainer_key,
        backend=LinearBackendSpec(
            parameter_backend="closed_form",
            runtime_backend="numpy",
            solver_kind="ridge",
            continuation_mode="closed_form_refit",
            trainer_state_enabled=True,
            supports_resume=True,
            supports_warm_start=True,
            supports_incremental=True,
        ),
        function_class=LinearFunctionClassSpec(
            basis="affine",
            fit_intercept=True,
        ),
        regularization=LinearRegularizationSpec(
            penalty="l2",
            l2=l2,
        ),
        task_head=LinearTaskHeadSpec(
            task="point",
            objective_family="regression",
            outputs=("mean",),
            uncertainty_mode=uncertainty_mode,
        ),
        metadata={} if metadata is None else dict(metadata),
    )


def build_unified_linear_family_spec(
    *,
    trainer_key: str = LINEAR_FORMAL_PRESET_KEY,
    l2: float = 1.0,
    uncertainty_mode: str = "residual_std",
    metadata: Mapping[str, Any] | None = None,
) -> LinearTrainerFamilySpec:
    return build_ridge_family_spec(
        trainer_key=trainer_key,
        l2=l2,
        uncertainty_mode=uncertainty_mode,
        metadata={
            "preset_kind": "formal_family",
            "surface_status": "formal",
            "route_family": "linear",
            **dict(metadata or {}),
        },
    )


def linear_route_registry() -> tuple[FamilyRouteSpec, ...]:
    return (
        FamilyRouteSpec(
            family_key=LINEAR_FORMAL_PRESET_KEY,
            route_key="ridge",
            match_fields={
                "backend.solver_kind": ("ridge",),
                "task_head.task": ("point",),
            },
            status="stable",
            summary="Closed-form ridge regression route for the linear family.",
        ),
    )


def resolve_linear_route_spec(
    family_spec: LinearTrainerFamilySpec | Mapping[str, Any] | None,
) -> FamilyRouteSpec:
    spec = coerce_linear_family_spec(family_spec, trainer_key=LINEAR_FORMAL_PRESET_KEY)
    return resolve_family_route_spec(
        linear_route_registry(),
        spec,
        family_key=LINEAR_FORMAL_PRESET_KEY,
    )


def resolve_linear_router_target(
    family_spec: LinearTrainerFamilySpec | Mapping[str, Any] | None,
) -> str:
    return resolve_linear_route_spec(family_spec).route_key


def linear_surface_contract() -> dict[str, Any]:
    return {
        "formal_preset": LINEAR_FORMAL_PRESET_KEY,
        "route_keys": LINEAR_ROUTE_KEYS,
        "route_registry": serialize_family_route_registry(linear_route_registry()),
        "surface_status": {
            LINEAR_FORMAL_PRESET_KEY: "formal",
            "ridge": "route_target",
        },
    }


def coerce_linear_family_spec(
    value: LinearTrainerFamilySpec | Mapping[str, Any] | None,
    *,
    trainer_key: str = "ridge",
) -> LinearTrainerFamilySpec:
    if value is None:
        if _normalize_name(trainer_key, "ridge") == LINEAR_FORMAL_PRESET_KEY:
            return build_unified_linear_family_spec(trainer_key=trainer_key)
        return build_ridge_family_spec(trainer_key=trainer_key)
    if isinstance(value, LinearTrainerFamilySpec):
        return value
    raw = dict(value)
    backend = coerce_linear_backend_spec(raw.get("backend"))
    function_class = coerce_linear_function_class_spec(raw.get("function_class"))
    regularization = coerce_linear_regularization_spec(raw.get("regularization"))
    task_head = coerce_linear_task_head_spec(raw.get("task_head"))
    return LinearTrainerFamilySpec(
        trainer_key=str(raw.get("trainer_key", trainer_key)),
        backend=backend,
        function_class=function_class,
        regularization=regularization,
        task_head=task_head,
        metadata=dict(raw.get("metadata", {}) or {}),
    )


__all__ = [
    "LINEAR_FORMAL_PRESET_KEY",
    "LINEAR_ROUTE_KEYS",
    "LinearBackendSpec",
    "LinearFunctionClassSpec",
    "LinearRegularizationSpec",
    "LinearTaskHeadSpec",
    "LinearTrainerFamilySpec",
    "build_unified_linear_family_spec",
    "build_ridge_family_spec",
    "coerce_linear_backend_spec",
    "coerce_linear_family_spec",
    "coerce_linear_function_class_spec",
    "coerce_linear_regularization_spec",
    "coerce_linear_task_head_spec",
    "linear_route_registry",
    "linear_surface_contract",
    "resolve_linear_route_spec",
    "resolve_linear_router_target",
]
