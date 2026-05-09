from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Any, Mapping

from core.common.family_router import (
    FamilyRouteSpec,
    match_family_routes,
    resolve_family_route_spec,
)
from core.symbolic.artifact_schema import symbolic_artifact_schema_descriptor
from core.mechanisms import (
    MechanismProtocolBase,
    build_symbolic_family_mechanism_bindings,
    serialize_family_bindings,
)
from core.symbolic.search_mechanism_contract import (
    SymbolicSearchMechanismContract,
    build_symbolic_search_mechanism_contracts,
    serialize_symbolic_search_mechanism_contracts,
)
from core.symbolic.structure_contract import (
    BudgetedSymbolicAssemblerContract,
    build_budgeted_symbolic_assembler_contract,
    build_symbolic_basis_discovery_contract,
    build_symbolic_regime_discovery_contract,
    SymbolicBasisDiscoveryContract,
    SymbolicRegimeDiscoveryContract,
)

SYMBOLIC_FORMAL_PRESET_KEY: str = "symbolic"
SYMBOLIC_LEGACY_PRESET_KEYS: tuple[str, ...] = (
    "symbolic_stagewise",
    "symbolic_torch",
    "symbolic_torch_interval",
)


@dataclass(frozen=True)
class SymbolicRouteSpec:
    route_key: str
    parameter_backend: str
    task: str
    structure_modes: tuple[str, ...] = field(default_factory=tuple)
    status: str = "stable"
    summary: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "route_key", _normalize_name(self.route_key, "symbolic_stagewise"))
        object.__setattr__(self, "parameter_backend", _normalize_name(self.parameter_backend, "ridge"))
        object.__setattr__(self, "task", _normalize_name(self.task, "point"))
        object.__setattr__(
            self,
            "structure_modes",
            tuple(
                _normalize_name(value, "")
                for value in tuple(self.structure_modes)
                if _normalize_name(value, "")
            ),
        )
        object.__setattr__(self, "status", _normalize_name(self.status, "stable"))
        object.__setattr__(self, "summary", str(self.summary or "").strip())

    def as_dict(self) -> dict[str, Any]:
        return {
            "route_key": self.route_key,
            "parameter_backend": self.parameter_backend,
            "task": self.task,
            "structure_modes": tuple(self.structure_modes),
            "status": self.status,
            "summary": self.summary,
        }

    def as_family_route_spec(self) -> FamilyRouteSpec:
        match_fields: dict[str, Any] = {
            "parameter_backend.backend": self.parameter_backend,
            "task_head.task": self.task,
        }
        if self.structure_modes:
            match_fields["structure_engine.structure_mode"] = tuple(self.structure_modes)
        return FamilyRouteSpec(
            family_key=SYMBOLIC_FORMAL_PRESET_KEY,
            route_key=self.route_key,
            match_fields=match_fields,
            status=self.status,
            summary=self.summary,
        )


def symbolic_route_registry() -> tuple[SymbolicRouteSpec, ...]:
    return (
        SymbolicRouteSpec(
            route_key="symbolic_stagewise",
            parameter_backend="ridge",
            task="point",
            structure_modes=("stagewise_search",),
            status="stable",
            summary="Ridge-backed symbolic point route with explicit stagewise structure search.",
        ),
        SymbolicRouteSpec(
            route_key="symbolic_orthogonal",
            parameter_backend="ridge",
            task="point",
            structure_modes=("orthogonal_basis_search",),
            status="stable",
            summary="Ridge-backed symbolic point route that discovers orthogonal/piecewise basis sets before the final small-budget symbolic assembler.",
        ),
        SymbolicRouteSpec(
            route_key="symbolic_torch",
            parameter_backend="torch",
            task="point",
            status="stable",
            summary="Torch-backed symbolic point route for gradient-trained symbolic regressors.",
        ),
        SymbolicRouteSpec(
            route_key="symbolic_torch_interval",
            parameter_backend="torch",
            task="interval",
            status="stable",
            summary="Torch-backed symbolic interval route for lower/upper predictive heads.",
        ),
    )


def serialize_symbolic_route_registry(
    routes: tuple[SymbolicRouteSpec, ...] | None = None,
) -> list[dict[str, Any]]:
    route_specs = tuple(routes) if routes is not None else symbolic_route_registry()
    return [route.as_dict() for route in route_specs]


def _format_symbolic_route(route: SymbolicRouteSpec) -> str:
    structure_modes = ",".join(route.structure_modes) if route.structure_modes else "*"
    return (
        f"{route.route_key}(parameter_backend={route.parameter_backend}, "
        f"task={route.task}, structure_modes={structure_modes})"
    )


def _route_context_label(
    *,
    backend: str,
    task: str,
    structure_mode: str,
) -> str:
    return (
        f"parameter_backend='{backend}', task='{task}', "
        f"structure_engine.structure_mode='{structure_mode}'"
    )


def symbolic_route_matches(
    route: SymbolicRouteSpec,
    *,
    parameter_backend: str,
    task: str,
    structure_mode: str,
) -> bool:
    backend_norm = _normalize_name(parameter_backend, "ridge")
    task_norm = _normalize_name(task, "point")
    structure_mode_norm = _normalize_name(structure_mode, "stagewise_search")
    if route.parameter_backend != backend_norm or route.task != task_norm:
        return False
    if route.structure_modes and structure_mode_norm not in route.structure_modes:
        return False
    return True


def match_symbolic_routes(
    family_spec: SymbolicTrainerFamilySpec | Mapping[str, Any] | None,
    *,
    default_backend: str = "ridge",
    default_task: str = "point",
) -> tuple[SymbolicRouteSpec, ...]:
    spec = coerce_symbolic_family_spec(
        family_spec,
        trainer_key=SYMBOLIC_FORMAL_PRESET_KEY,
        default_backend=default_backend,
        default_task=default_task,
    )
    backend = str(spec.parameter_backend.backend).strip().lower()
    task = str(spec.task_head.task).strip().lower()
    structure_mode = str(spec.structure_engine.structure_mode).strip().lower()
    route_specs = tuple(symbolic_route_registry())
    generic_routes = tuple(route.as_family_route_spec() for route in route_specs)
    matched_keys = {
        resolved.route_key
        for resolved in match_family_routes(generic_routes, spec)
    }
    return tuple(route for route in route_specs if route.route_key in matched_keys)


def resolve_symbolic_route_spec(
    family_spec: SymbolicTrainerFamilySpec | Mapping[str, Any] | None,
    *,
    default_backend: str = "ridge",
    default_task: str = "point",
) -> SymbolicRouteSpec:
    spec = coerce_symbolic_family_spec(
        family_spec,
        trainer_key=SYMBOLIC_FORMAL_PRESET_KEY,
        default_backend=default_backend,
        default_task=default_task,
    )
    backend = str(spec.parameter_backend.backend).strip().lower()
    task = str(spec.task_head.task).strip().lower()
    structure_mode = str(spec.structure_engine.structure_mode).strip().lower()
    route_registry = tuple(symbolic_route_registry())
    matches = match_symbolic_routes(
        spec,
        default_backend=default_backend,
        default_task=default_task,
    )
    if len(matches) > 1:
        context_label = _route_context_label(
            backend=backend,
            task=task,
            structure_mode=structure_mode,
        )
        matched_label = "; ".join(_format_symbolic_route(route) for route in matches)
        registry_label = "; ".join(_format_symbolic_route(route) for route in route_registry)
        raise ValueError(
            "symbolic route conflict: "
            f"{context_label} matched multiple registered routes [{matched_label}]. "
            f"Registered routes: [{registry_label}]"
        )
    generic_routes = tuple(route.as_family_route_spec() for route in route_registry)
    try:
        resolved = resolve_family_route_spec(
            generic_routes,
            spec,
            family_key=SYMBOLIC_FORMAL_PRESET_KEY,
        )
    except ValueError as exc:
        sibling_routes = tuple(
            route
            for route in route_registry
            if route.parameter_backend == backend and route.task == task
        )
        if sibling_routes:
            sibling_label = "; ".join(_format_symbolic_route(route) for route in sibling_routes)
            context_label = _route_context_label(
                backend=backend,
                task=task,
                structure_mode=structure_mode,
            )
            raise ValueError(
                "unsupported symbolic route: "
                f"{context_label} matched backend/task siblings but failed structure-mode constraints. "
                f"Eligible backend/task routes: [{sibling_label}]"
            ) from exc
        raise
    for route in route_registry:
        if route.route_key == resolved.route_key:
            return route
    raise ValueError(f"symbolic route registry resolved unknown route '{resolved.route_key}'")


def _normalize_name(value: str | None, default: str) -> str:
    text = str(value or "").strip().lower()
    return text or str(default)


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in sorted(dict(value).items(), key=lambda item: str(item[0]))}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_jsonable(v) for v in value]
    return str(value)


def _stable_hash(payload: Any) -> str | None:
    if payload is None:
        return None
    encoded = json.dumps(_jsonable(payload), sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    if encoded in {"null", "{}", "[]", "\"\""}:
        return None
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _infer_symbolic_backend_capabilities(
    *,
    trainer_key: str | None = None,
    backend: str | None = None,
    task: str | None = None,
) -> dict[str, bool]:
    key = _normalize_name(trainer_key, "symbolic")
    backend_norm = _normalize_name(backend, "ridge")
    task_norm = _normalize_name(task, "point")

    if key in {"symbolic_stagewise", "symbolic_orthogonal", "symbolic_torch", "symbolic_torch_interval"}:
        return {
            "trainer_state_enabled": True,
            "supports_resume": True,
            "supports_warm_start": True,
            "supports_incremental": True,
        }

    if backend_norm == "torch" and task_norm in {"point", "interval"}:
        return {
            "trainer_state_enabled": True,
            "supports_resume": True,
            "supports_warm_start": True,
            "supports_incremental": True,
        }

    if backend_norm == "ridge" and task_norm == "point":
        return {
            "trainer_state_enabled": True,
            "supports_resume": True,
            "supports_warm_start": True,
            "supports_incremental": True,
        }

    return {
        "trainer_state_enabled": False,
        "supports_resume": False,
        "supports_warm_start": False,
        "supports_incremental": False,
    }


def is_legacy_symbolic_preset(trainer_key: str | None) -> bool:
    return _normalize_name(trainer_key, SYMBOLIC_FORMAL_PRESET_KEY) in SYMBOLIC_LEGACY_PRESET_KEYS


def canonical_symbolic_preset_key(trainer_key: str | None) -> str:
    key = _normalize_name(trainer_key, SYMBOLIC_FORMAL_PRESET_KEY)
    if key == SYMBOLIC_FORMAL_PRESET_KEY or key in SYMBOLIC_LEGACY_PRESET_KEYS:
        return SYMBOLIC_FORMAL_PRESET_KEY
    return key


def symbolic_surface_contract() -> dict[str, Any]:
    return {
        "formal_preset": SYMBOLIC_FORMAL_PRESET_KEY,
        "legacy_facades": tuple(SYMBOLIC_LEGACY_PRESET_KEYS),
        "route_registry": serialize_symbolic_route_registry(),
        "route_keys": tuple(route.route_key for route in symbolic_route_registry()),
        "surface_status": {
            SYMBOLIC_FORMAL_PRESET_KEY: "formal",
            **{key: "deprecated" for key in SYMBOLIC_LEGACY_PRESET_KEYS},
        },
        "migration_target": {
            key: SYMBOLIC_FORMAL_PRESET_KEY
            for key in SYMBOLIC_LEGACY_PRESET_KEYS
        },
    }


def resolve_symbolic_router_target(
    family_spec: SymbolicTrainerFamilySpec | Mapping[str, Any] | None,
    *,
    default_backend: str = "ridge",
    default_task: str = "point",
) -> str:
    return resolve_symbolic_route_spec(
        family_spec,
        default_backend=default_backend,
        default_task=default_task,
    ).route_key


@dataclass(frozen=True)
class SymbolicStructureEngineSpec:
    structure_mode: str = "stagewise_search"
    candidate_space: str = "feature_space"
    grammar_source: str = "primitive_registry+generation_grammar"
    search_driver: str = "nsgablack"
    dynamic_pool_enabled: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "structure_mode", _normalize_name(self.structure_mode, "stagewise_search"))
        object.__setattr__(self, "candidate_space", _normalize_name(self.candidate_space, "feature_space"))
        object.__setattr__(
            self,
            "grammar_source",
            str(self.grammar_source or "primitive_registry+generation_grammar").strip()
            or "primitive_registry+generation_grammar",
        )
        object.__setattr__(self, "search_driver", _normalize_name(self.search_driver, "nsgablack"))
        object.__setattr__(self, "dynamic_pool_enabled", bool(self.dynamic_pool_enabled))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def as_dict(self) -> dict[str, Any]:
        return {
            "structure_mode": self.structure_mode,
            "candidate_space": self.candidate_space,
            "grammar_source": self.grammar_source,
            "search_driver": self.search_driver,
            "dynamic_pool_enabled": bool(self.dynamic_pool_enabled),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class SymbolicParameterBackendSpec:
    backend: str = "ridge"
    optimizer_family: str = "closed_form"
    trainer_state_enabled: bool = False
    supports_resume: bool = False
    supports_warm_start: bool = False
    supports_incremental: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "backend", _normalize_name(self.backend, "ridge"))
        object.__setattr__(self, "optimizer_family", str(self.optimizer_family or "closed_form").strip() or "closed_form")
        object.__setattr__(self, "trainer_state_enabled", bool(self.trainer_state_enabled))
        object.__setattr__(self, "supports_resume", bool(self.supports_resume))
        object.__setattr__(self, "supports_warm_start", bool(self.supports_warm_start))
        object.__setattr__(self, "supports_incremental", bool(self.supports_incremental))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def as_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "optimizer_family": self.optimizer_family,
            "trainer_state_enabled": bool(self.trainer_state_enabled),
            "supports_resume": bool(self.supports_resume),
            "supports_warm_start": bool(self.supports_warm_start),
            "supports_incremental": bool(self.supports_incremental),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class SymbolicTaskHeadSpec:
    task: str = "point"
    outputs: tuple[str, ...] = ("mean",)
    objective_family: str = "regression"
    calibration_mode: str = "none"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "task", _normalize_name(self.task, "point"))
        outs = tuple(str(v).strip().lower() for v in tuple(self.outputs) if str(v).strip()) or ("mean",)
        object.__setattr__(self, "outputs", outs)
        object.__setattr__(self, "objective_family", _normalize_name(self.objective_family, "regression"))
        object.__setattr__(self, "calibration_mode", _normalize_name(self.calibration_mode, "none"))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def as_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "outputs": tuple(self.outputs),
            "objective_family": self.objective_family,
            "calibration_mode": self.calibration_mode,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class SymbolicTrainerFamilySpec:
    trainer_key: str = "symbolic"
    structure_engine: SymbolicStructureEngineSpec = field(default_factory=SymbolicStructureEngineSpec)
    parameter_backend: SymbolicParameterBackendSpec = field(default_factory=SymbolicParameterBackendSpec)
    task_head: SymbolicTaskHeadSpec = field(default_factory=SymbolicTaskHeadSpec)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "trainer_key", str(self.trainer_key or "symbolic").strip() or "symbolic")
        object.__setattr__(self, "metadata", dict(self.metadata))

    def as_dict(self) -> dict[str, Any]:
        return {
            "trainer_key": self.trainer_key,
            "structure_engine": self.structure_engine.as_dict(),
            "parameter_backend": self.parameter_backend.as_dict(),
            "task_head": self.task_head.as_dict(),
            "metadata": dict(self.metadata),
        }

    def mechanism_bindings(self) -> tuple[MechanismProtocolBase, ...]:
        return build_symbolic_family_mechanism_bindings()

    def mechanism_binding_payload(self) -> list[dict[str, object]]:
        return serialize_family_bindings(self.mechanism_bindings())

    def search_mechanism_contracts(self) -> tuple[SymbolicSearchMechanismContract, ...]:
        return build_symbolic_search_mechanism_contracts()

    def search_mechanism_contract_payload(self) -> list[dict[str, Any]]:
        return serialize_symbolic_search_mechanism_contracts(self.search_mechanism_contracts())

    def family_signature_contracts(self) -> tuple[SymbolicSearchMechanismContract, ...]:
        return tuple(
            contract
            for contract in self.search_mechanism_contracts()
            if bool(getattr(contract, "affects_family_signature", False))
        )

    def family_signature_contract_payload(self) -> list[dict[str, Any]]:
        return serialize_symbolic_search_mechanism_contracts(self.family_signature_contracts())

    def supports_piecewise_structure(self) -> bool:
        if str(self.task_head.task).strip().lower() == "interval":
            return True
        structure_meta = dict(getattr(self.structure_engine, "metadata", {}) or {})
        task_meta = dict(getattr(self.task_head, "metadata", {}) or {})
        family_meta = dict(self.metadata or {})
        return bool(
            family_meta.get("supports_piecewise_basis")
            or structure_meta.get("supports_piecewise_basis")
            or task_meta.get("supports_piecewise_basis")
        )

    def regime_discovery_contract(self) -> SymbolicRegimeDiscoveryContract:
        supports_piecewise = self.supports_piecewise_structure()
        return build_symbolic_regime_discovery_contract(
            task=str(self.task_head.task),
            supports_piecewise=supports_piecewise,
        )

    def basis_discovery_contract(self) -> SymbolicBasisDiscoveryContract:
        supports_piecewise = self.supports_piecewise_structure()
        return build_symbolic_basis_discovery_contract(supports_piecewise=supports_piecewise)

    def budgeted_symbolic_assembler_contract(self) -> BudgetedSymbolicAssemblerContract:
        supports_piecewise = self.supports_piecewise_structure()
        return build_budgeted_symbolic_assembler_contract(supports_piecewise=supports_piecewise)

    def structure_contract_payload(self) -> dict[str, Any]:
        return {
            "regime_discovery": self.regime_discovery_contract().as_dict(),
            "basis_discovery": self.basis_discovery_contract().as_dict(),
            "budgeted_symbolic_assembler": self.budgeted_symbolic_assembler_contract().as_dict(),
        }

    def family_signature_payload(self) -> dict[str, Any]:
        payload = self.as_dict()
        parameter_backend = dict(payload.get("parameter_backend", {}) or {})
        for key in ("trainer_state_enabled", "supports_resume", "supports_warm_start", "supports_incremental"):
            parameter_backend.pop(key, None)
        payload["parameter_backend"] = parameter_backend
        payload["search_mechanism_contracts"] = self.search_mechanism_contract_payload()
        payload["search_family_signature_contracts"] = self.family_signature_contract_payload()
        payload["structure_contracts"] = self.structure_contract_payload()
        return payload

    def family_signature(self) -> str | None:
        payload = self.family_signature_payload()
        hash_payload = dict(payload)
        hash_payload.pop("search_mechanism_contracts", None)
        return _stable_hash(hash_payload)

    def description_dict(self) -> dict[str, Any]:
        payload = self.as_dict()
        payload["mechanism_bindings"] = self.mechanism_binding_payload()
        payload["search_mechanism_contracts"] = self.search_mechanism_contract_payload()
        payload["search_family_signature_contracts"] = self.family_signature_contract_payload()
        payload["structure_contracts"] = self.structure_contract_payload()
        payload["regime_discovery_contract"] = self.regime_discovery_contract().as_dict()
        payload["basis_discovery_contract"] = self.basis_discovery_contract().as_dict()
        payload["budgeted_symbolic_assembler_contract"] = self.budgeted_symbolic_assembler_contract().as_dict()
        payload["family_signature"] = self.family_signature()
        payload["artifact_schema"] = symbolic_artifact_schema_descriptor(
            task=str(self.task_head.task),
            outputs=tuple(self.task_head.outputs),
            objective_family=str(self.task_head.objective_family),
            calibration_mode=str(self.task_head.calibration_mode),
            supports_piecewise=str(self.task_head.task).strip().lower() == "interval",
        )
        return payload


def coerce_symbolic_structure_engine_spec(
    value: SymbolicStructureEngineSpec | Mapping[str, Any] | None,
    *,
    default: SymbolicStructureEngineSpec | Mapping[str, Any] | None = None,
) -> SymbolicStructureEngineSpec:
    if value is None:
        if default is None:
            return SymbolicStructureEngineSpec()
        value = default
    if isinstance(value, SymbolicStructureEngineSpec):
        return value
    return SymbolicStructureEngineSpec(**dict(value))


def coerce_symbolic_parameter_backend_spec(
    value: SymbolicParameterBackendSpec | Mapping[str, Any] | None,
    *,
    default: SymbolicParameterBackendSpec | Mapping[str, Any] | None = None,
) -> SymbolicParameterBackendSpec:
    if value is None:
        if default is None:
            return SymbolicParameterBackendSpec()
        value = default
    if isinstance(value, SymbolicParameterBackendSpec):
        return value
    return SymbolicParameterBackendSpec(**dict(value))


def coerce_symbolic_task_head_spec(
    value: SymbolicTaskHeadSpec | Mapping[str, Any] | None,
    *,
    default: SymbolicTaskHeadSpec | Mapping[str, Any] | None = None,
) -> SymbolicTaskHeadSpec:
    if value is None:
        if default is None:
            return SymbolicTaskHeadSpec()
        value = default
    if isinstance(value, SymbolicTaskHeadSpec):
        return value
    return SymbolicTaskHeadSpec(**dict(value))


def coerce_symbolic_family_spec(
    value: SymbolicTrainerFamilySpec | Mapping[str, Any] | None,
    *,
    trainer_key: str = "symbolic",
    default_backend: str = "ridge",
    default_task: str = "point",
    default_calibration_mode: str = "none",
) -> SymbolicTrainerFamilySpec:
    if value is None:
        return build_unified_symbolic_family_spec(
            trainer_key=trainer_key,
            parameter_backend=default_backend,
            task=default_task,
            calibration_mode=default_calibration_mode,
        )
    if isinstance(value, SymbolicTrainerFamilySpec):
        return value

    raw = dict(value)
    structure_engine = coerce_symbolic_structure_engine_spec(raw.get("structure_engine"))
    task_head_raw = raw.get("task_head")
    if isinstance(task_head_raw, Mapping):
        task_head = coerce_symbolic_task_head_spec(task_head_raw)
    else:
        task_head = SymbolicTaskHeadSpec(
            task=str(raw.get("task", default_task)),
            calibration_mode=str(raw.get("calibration_mode", default_calibration_mode)),
        )
    trainer_key_eff = str(raw.get("trainer_key", trainer_key))
    parameter_backend_raw = raw.get("parameter_backend")
    if isinstance(parameter_backend_raw, Mapping):
        backend_payload = dict(parameter_backend_raw)
        inferred = _infer_symbolic_backend_capabilities(
            trainer_key=trainer_key_eff,
            backend=backend_payload.get("backend", default_backend),
            task=task_head.task,
        )
        merged_backend = dict(inferred)
        merged_backend.update(backend_payload)
        parameter_backend = coerce_symbolic_parameter_backend_spec(merged_backend)
    else:
        backend_eff = (
            str(parameter_backend_raw)
            if parameter_backend_raw is not None
            else str(default_backend)
        )
        inferred = _infer_symbolic_backend_capabilities(
            trainer_key=trainer_key_eff,
            backend=backend_eff,
            task=task_head.task,
        )
        parameter_backend = SymbolicParameterBackendSpec(
            backend=backend_eff,
            trainer_state_enabled=bool(inferred["trainer_state_enabled"]),
            supports_resume=bool(inferred["supports_resume"]),
            supports_warm_start=bool(inferred["supports_warm_start"]),
            supports_incremental=bool(inferred["supports_incremental"]),
        )
    return SymbolicTrainerFamilySpec(
        trainer_key=trainer_key_eff,
        structure_engine=structure_engine,
        parameter_backend=parameter_backend,
        task_head=task_head,
        metadata=dict(raw.get("metadata", {})),
    )


def legacy_symbolic_family_spec(
    trainer_key: str,
    *,
    trainer_params: Mapping[str, Any] | None = None,
) -> SymbolicTrainerFamilySpec:
    key = _normalize_name(trainer_key, "symbolic")
    params = dict(trainer_params or {})
    has_explicit_genome = params.get("genome") is not None
    stagewise_warmup_enabled = bool(params.get("stagewise_warmup_enabled", False))
    conformal_enabled = bool(params.get("conformal_calibration", False))

    if key == "symbolic_stagewise":
        return SymbolicTrainerFamilySpec(
            trainer_key=key,
            structure_engine=SymbolicStructureEngineSpec(
                structure_mode="stagewise_search",
                search_driver="nsgablack",
                dynamic_pool_enabled=True,
            ),
            parameter_backend=SymbolicParameterBackendSpec(
                backend="ridge",
                optimizer_family="closed_form+optional_inner_opt",
            ),
            task_head=SymbolicTaskHeadSpec(
                task="point",
                outputs=("mean",),
                objective_family="regression",
                calibration_mode="none",
            ),
            metadata={
                "preset_kind": "legacy_facade",
                "surface_status": "deprecated",
                "canonical_preset": SYMBOLIC_FORMAL_PRESET_KEY,
                "migration_target": SYMBOLIC_FORMAL_PRESET_KEY,
            },
        )

    if key == "symbolic_torch":
        structure_mode = "explicit_genome" if has_explicit_genome else "seed_library"
        return SymbolicTrainerFamilySpec(
            trainer_key=key,
            structure_engine=SymbolicStructureEngineSpec(
                structure_mode=structure_mode,
                search_driver="local_seed_builder",
                dynamic_pool_enabled=False,
            ),
            parameter_backend=SymbolicParameterBackendSpec(
                backend="torch",
                optimizer_family="gradient",
            ),
            task_head=SymbolicTaskHeadSpec(
                task="point",
                outputs=("mean",),
                objective_family="regression",
                calibration_mode="none",
            ),
            metadata={
                "preset_kind": "legacy_facade",
                "surface_status": "deprecated",
                "canonical_preset": SYMBOLIC_FORMAL_PRESET_KEY,
                "migration_target": SYMBOLIC_FORMAL_PRESET_KEY,
            },
        )

    if key == "symbolic_torch_interval":
        if has_explicit_genome:
            structure_mode = "explicit_genome"
            search_driver = "none"
        elif stagewise_warmup_enabled:
            structure_mode = "stagewise_warmup_then_seed_library"
            search_driver = "nsgablack_warmup"
        else:
            structure_mode = "seed_library"
            search_driver = "local_seed_builder"
        return SymbolicTrainerFamilySpec(
            trainer_key=key,
            structure_engine=SymbolicStructureEngineSpec(
                structure_mode=structure_mode,
                search_driver=search_driver,
                dynamic_pool_enabled=bool(stagewise_warmup_enabled),
            ),
            parameter_backend=SymbolicParameterBackendSpec(
                backend="torch",
                optimizer_family="gradient",
            ),
            task_head=SymbolicTaskHeadSpec(
                task="interval",
                outputs=("lower", "upper"),
                objective_family="quantile_interval",
                calibration_mode="conformal" if conformal_enabled else "none",
            ),
            metadata={
                "preset_kind": "legacy_facade",
                "surface_status": "deprecated",
                "canonical_preset": SYMBOLIC_FORMAL_PRESET_KEY,
                "migration_target": SYMBOLIC_FORMAL_PRESET_KEY,
            },
        )

    raise ValueError(f"unsupported symbolic trainer key: {trainer_key}")


def build_unified_symbolic_family_spec(
    *,
    trainer_key: str = "symbolic",
    parameter_backend: str = "torch",
    task: str = "point",
    calibration_mode: str = "none",
    outputs: tuple[str, ...] | None = None,
    trainer_state_enabled: bool = False,
    supports_resume: bool = False,
    supports_warm_start: bool = False,
    supports_incremental: bool = False,
    supports_piecewise_basis: bool = False,
    metadata: Mapping[str, Any] | None = None,
) -> SymbolicTrainerFamilySpec:
    task_norm = _normalize_name(task, "point")
    if outputs is None:
        outputs_eff = ("lower", "upper") if task_norm == "interval" else ("mean",)
    else:
        outputs_eff = tuple(str(v).strip().lower() for v in tuple(outputs) if str(v).strip()) or (
            ("lower", "upper") if task_norm == "interval" else ("mean",)
        )
    objective_family = "quantile_interval" if task_norm == "interval" else "regression"
    structure_metadata = {"supports_piecewise_basis": bool(supports_piecewise_basis)} if supports_piecewise_basis else {}
    metadata_payload = dict(
        metadata
        or {
            "preset_kind": "unified_target",
            "surface_status": "formal",
            "canonical_preset": SYMBOLIC_FORMAL_PRESET_KEY,
            "legacy_facades": tuple(SYMBOLIC_LEGACY_PRESET_KEYS),
        }
    )
    if supports_piecewise_basis:
        metadata_payload.setdefault("supports_piecewise_basis", True)
    return SymbolicTrainerFamilySpec(
        trainer_key=str(trainer_key or "symbolic"),
        structure_engine=SymbolicStructureEngineSpec(
            structure_mode="stagewise_search",
            search_driver="nsgablack",
            dynamic_pool_enabled=True,
            metadata=structure_metadata,
        ),
        parameter_backend=SymbolicParameterBackendSpec(
            backend=parameter_backend,
            optimizer_family="gradient" if _normalize_name(parameter_backend, "torch") == "torch" else "closed_form",
            trainer_state_enabled=bool(trainer_state_enabled),
            supports_resume=bool(supports_resume),
            supports_warm_start=bool(supports_warm_start),
            supports_incremental=bool(supports_incremental),
        ),
        task_head=SymbolicTaskHeadSpec(
            task=task_norm,
            outputs=outputs_eff,
            objective_family=objective_family,
            calibration_mode=calibration_mode,
        ),
        metadata=metadata_payload,
    )


__all__ = [
    "canonical_symbolic_preset_key",
    "coerce_symbolic_family_spec",
    "coerce_symbolic_parameter_backend_spec",
    "coerce_symbolic_structure_engine_spec",
    "coerce_symbolic_task_head_spec",
    "is_legacy_symbolic_preset",
    "match_symbolic_routes",
    "resolve_symbolic_route_spec",
    "serialize_symbolic_route_registry",
    "SYMBOLIC_FORMAL_PRESET_KEY",
    "SYMBOLIC_LEGACY_PRESET_KEYS",
    "SymbolicParameterBackendSpec",
    "SymbolicRouteSpec",
    "SymbolicStructureEngineSpec",
    "SymbolicTaskHeadSpec",
    "SymbolicTrainerFamilySpec",
    "build_unified_symbolic_family_spec",
    "legacy_symbolic_family_spec",
    "resolve_symbolic_router_target",
    "symbolic_route_matches",
    "symbolic_route_registry",
    "symbolic_surface_contract",
]
