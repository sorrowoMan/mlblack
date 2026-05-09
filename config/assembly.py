from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Sequence

from core.execution import (
    coerce_execution_resource_request,
    describe_registered_execution_backends,
    describe_registered_execution_device_kinds,
    list_registered_execution_backend_keys,
    sum_execution_resource_requests,
)
from core.mechanisms import serialize_mechanism_protocols

from .defaults import create_default_config
from .registry import MLBlackConfig


# Keys that belong to semantic numericization layer, not trainer_params in semantic flow.
SEMANTIC_NUMERICIZER_KEYS: frozenset[str] = frozenset(
    {
        "numericizer",
        "modality_encoders",
        "target_codecs",
        "target_codec",
        "categorical_unknown",
    }
)

EXECUTION_GPU_STRATEGY_CHOICES: tuple[str, ...] = ("none", "fixed", "round_robin", "auto")


@dataclass(frozen=True)
class BiasSpec:
    key: str = "noop"
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class NumericizerSpec:
    key: str = "default"
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CapabilitySpec:
    key: str = "noop"
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExecutionSpec:
    """L0 execution substrate declaration for flow/scaffold config."""

    backend: str = "serial"
    max_workers: int | None = None
    fail_fast: bool = True
    gpu_strategy: str = "none"
    gpu_devices: Sequence[str | int] = field(default_factory=tuple)
    default_device: str | int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "backend", str(self.backend or "serial").strip().lower() or "serial")
        object.__setattr__(self, "gpu_strategy", str(self.gpu_strategy or "none").strip().lower() or "none")
        object.__setattr__(self, "gpu_devices", tuple(self.gpu_devices))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "backend": str(self.backend),
            "max_workers": None if self.max_workers is None else int(self.max_workers),
            "fail_fast": bool(self.fail_fast),
            "gpu_strategy": str(self.gpu_strategy),
            "gpu_devices": list(self.gpu_devices),
            "default_device": self.default_device,
        }


@dataclass(frozen=True)
class TrainerAssemblySpec:
    trainer_key: str = "ridge"
    trainer_params: Dict[str, Any] = field(default_factory=dict)
    pipeline_key: str = "identity"
    pipeline_params: Dict[str, Any] = field(default_factory=dict)
    biases: Sequence[BiasSpec] = field(default_factory=tuple)


@dataclass(frozen=True)
class FlowAssemblySpec:
    """Top-level semantic assembly spec.

    - numericizer: semantic->numeric layer
    - trainer: model training assembly
    - capabilities: flow lifecycle capability stack
    """

    trainer: TrainerAssemblySpec = field(default_factory=TrainerAssemblySpec)
    numericizer: NumericizerSpec | None = field(default_factory=NumericizerSpec)
    capabilities: Sequence[CapabilitySpec] = field(default_factory=tuple)


def coerce_execution_spec(
    value: ExecutionSpec | Mapping[str, Any] | None = None,
    *,
    fallback_backend: str = "serial",
    fallback_max_workers: int | None = None,
    fallback_fail_fast: bool = True,
    fallback_gpu_strategy: str = "none",
    fallback_gpu_devices: Sequence[str | int] = (),
    fallback_default_device: str | int | None = None,
) -> ExecutionSpec:
    if isinstance(value, ExecutionSpec):
        payload = value.to_dict()
    elif value is None:
        payload = {}
    elif isinstance(value, Mapping):
        payload = dict(value)
    else:
        raise TypeError("execution spec must be ExecutionSpec, mapping, or None")

    if "backend" not in payload and payload.get("parallel_mode") is not None:
        payload["backend"] = payload.pop("parallel_mode")
    if "gpu_strategy" not in payload and payload.get("device_strategy") is not None:
        payload["gpu_strategy"] = payload.pop("device_strategy")
    if "gpu_devices" not in payload and payload.get("devices") is not None:
        payload["gpu_devices"] = payload.pop("devices")

    return ExecutionSpec(
        backend=str(payload.get("backend", fallback_backend)),
        max_workers=(
            fallback_max_workers
            if payload.get("max_workers") is None
            else int(payload.get("max_workers"))
        ),
        fail_fast=bool(payload.get("fail_fast", fallback_fail_fast)),
        gpu_strategy=str(payload.get("gpu_strategy", fallback_gpu_strategy)),
        gpu_devices=tuple(payload.get("gpu_devices", fallback_gpu_devices)),
        default_device=payload.get("default_device", fallback_default_device),
    )


def describe_execution_spec_schema() -> Dict[str, Any]:
    backend_catalog = tuple(dict(row) for row in describe_registered_execution_backends())
    device_catalog = tuple(dict(row) for row in describe_registered_execution_device_kinds())
    defaults = ExecutionSpec().to_dict()
    device_examples: list[str] = []
    for row in device_catalog:
        for token in tuple(row.get("discovered_tokens", ())):
            key = str(token)
            if key and key not in device_examples:
                device_examples.append(key)
        for token in tuple(row.get("example_tokens", ())):
            key = str(token)
            if key and key not in device_examples:
                device_examples.append(key)

    return {
        "title": "ExecutionSpec",
        "plane": "L0",
        "description": "L0 execution substrate declaration for train/portfolio flows.",
        "defaults": defaults,
        "fields": {
            "backend": {
                "type": "string",
                "required": False,
                "default": str(defaults["backend"]),
                "enum": list(list_registered_execution_backend_keys()),
                "ui_widget": "select",
                "description": "Execution backend used by the control plane runtime.",
                "catalog": list(backend_catalog),
            },
            "max_workers": {
                "type": "integer|null",
                "required": False,
                "default": defaults["max_workers"],
                "minimum": 1,
                "description": "Worker count for parallel backends. Null lets runtime choose.",
            },
            "fail_fast": {
                "type": "boolean",
                "required": False,
                "default": bool(defaults["fail_fast"]),
                "description": "Stop the batch on first execution failure.",
            },
            "gpu_strategy": {
                "type": "string",
                "required": False,
                "default": str(defaults["gpu_strategy"]),
                "enum": list(EXECUTION_GPU_STRATEGY_CHOICES),
                "ui_widget": "select",
                "description": "How portfolio runs assign accelerator devices.",
            },
            "gpu_devices": {
                "type": "array",
                "required": False,
                "default": list(defaults["gpu_devices"]),
                "items": {"type": "string|integer"},
                "examples": list(device_examples),
                "description": "Explicit accelerator pool, for example [0, 1] or ['cuda:0', 'mps'].",
            },
            "default_device": {
                "type": "string|integer|null",
                "required": False,
                "default": defaults["default_device"],
                "examples": list(device_examples),
                "description": "Preferred device token injected into device-aware trainers when applicable.",
            },
        },
        "legacy_alias_fields": {
            "parallel_mode": "backend",
            "device_strategy": "gpu_strategy",
            "devices": "gpu_devices",
        },
        "backend_catalog": list(backend_catalog),
        "device_catalog": list(device_catalog),
    }


def _normalize_contract_key(value: Any) -> str:
    return str(value or "").strip().lower()


def _is_empty_contract_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not bool(value.strip())
    if isinstance(value, Mapping):
        return len(value) == 0
    if isinstance(value, (tuple, list, set, frozenset)):
        return len(value) == 0
    return False


def _validate_requested_trainer_registry_contract(spec: FlowAssemblySpec, cfg: MLBlackConfig) -> None:
    trainer_key = _normalize_contract_key(spec.trainer.trainer_key)
    metadata = dict(cfg.trainers.metadata(trainer_key))
    if not metadata:
        return

    backend = _normalize_contract_key(metadata.get("backend"))
    surface_status = _normalize_contract_key(metadata.get("surface_status"))
    if backend != "family_router" and surface_status != "formal":
        return

    route_family = _normalize_contract_key(metadata.get("route_family"))
    if not route_family:
        raise ValueError(f"formal trainer '{trainer_key}' is missing registry route_family metadata")

    route_registry = tuple(metadata.get("route_registry", ()) or ())
    if not route_registry:
        raise ValueError(f"formal trainer '{trainer_key}' is missing registry route_registry metadata")

    malformed: list[str] = []
    for index, row in enumerate(route_registry):
        if not isinstance(row, Mapping):
            malformed.append(f"[{index}] non-mapping")
            continue
        route_key = _normalize_contract_key(row.get("route_key"))
        match_fields = row.get("match_fields")
        if not route_key:
            malformed.append(f"[{index}] missing route_key")
        if not isinstance(match_fields, Mapping) or not dict(match_fields):
            malformed.append(f"[{index}] missing match_fields")
    if malformed:
        detail = ", ".join(malformed[:6])
        raise ValueError(f"formal trainer '{trainer_key}' has malformed route_registry entries: {detail}")


def _validate_runtime_trainer_router_contract(
    spec: FlowAssemblySpec,
    cfg: MLBlackConfig,
    trainer: Any,
) -> None:
    trainer_key = _normalize_contract_key(spec.trainer.trainer_key)
    metadata = dict(cfg.trainers.metadata(trainer_key))
    route_family = _normalize_contract_key(metadata.get("route_family"))
    if not route_family:
        return

    actual_family = _normalize_contract_key(getattr(trainer, "family_router_family", None))
    if actual_family != route_family:
        raise ValueError(
            f"trainer '{trainer_key}' built without matching family router contract: "
            f"expected family_router_family='{route_family}', got '{actual_family or '<missing>'}'"
        )

    route_target = _normalize_contract_key(getattr(trainer, "family_router_target", None))
    if not route_target:
        raise ValueError(f"trainer '{trainer_key}' is missing runtime family_router_target")

    route_spec = getattr(trainer, "family_route_spec", None)
    if not isinstance(route_spec, Mapping):
        raise ValueError(f"trainer '{trainer_key}' is missing runtime family_route_spec mapping")

    route_registry = tuple(getattr(trainer, "family_route_registry", ()) or ())
    if not route_registry:
        raise ValueError(f"trainer '{trainer_key}' is missing runtime family_route_registry")

    registry_targets = {
        _normalize_contract_key(dict(row).get("route_key"))
        for row in route_registry
        if isinstance(row, Mapping)
    }
    registry_targets.discard("")
    if route_target not in registry_targets:
        known = ", ".join(sorted(registry_targets)) or "<empty>"
        raise ValueError(
            f"trainer '{trainer_key}' resolved route target '{route_target}' not present in runtime route registry [{known}]"
        )


def _validate_selected_capability_contracts(
    spec: FlowAssemblySpec,
    capabilities: Sequence[Any],
) -> None:
    required_keys = ("requires", "provides", "mutates", "cache")
    for index, capability in enumerate(tuple(capabilities)):
        spec_key = str(spec.capabilities[index].key) if index < len(spec.capabilities) else getattr(capability, "name", index)
        contract_fn = getattr(capability, "get_context_contract", None)
        if not callable(contract_fn):
            raise ValueError(f"capability '{spec_key}' does not expose get_context_contract()")
        contract = contract_fn()
        if not isinstance(contract, Mapping):
            raise ValueError(f"capability '{spec_key}' returned non-mapping context contract")
        missing = [key for key in required_keys if key not in contract]
        if missing:
            names = ", ".join(missing)
            raise ValueError(f"capability '{spec_key}' context contract is missing keys: {names}")


def _validate_catalog_mount_contracts_for_preset(spec: FlowAssemblySpec) -> None:
    from catalog import show_entry

    preset_key = f"preset:{_normalize_contract_key(spec.trainer.trainer_key)}"
    preset = show_entry(preset_key, profile="framework-core")
    if preset is None:
        raise ValueError(f"catalog is missing selected preset entry '{preset_key}' for assembly validation")

    required_fields_by_kind = {
        "component": ("mount_plane", "mount_point", "orchestration_phases", "contract_consumes", "contract_provides", "contract_mutates"),
        "provider": ("mount_plane", "mount_point", "orchestration_phases", "contract_consumes", "contract_provides", "contract_mutates"),
        "plugin": ("mount_plane", "mount_point", "orchestration_phases", "contract_consumes", "contract_provides", "contract_mutates", "contract_cache"),
    }
    relation_kind_map = {
        "components": "component",
        "providers": "provider",
        "plugins": "plugin",
    }
    cache: dict[str, Any] = {}

    for relation_name, expected_kind in relation_kind_map.items():
        for related_key in tuple(dict(preset.relations).get(relation_name, ())):
            key = str(related_key)
            entry = cache.get(key)
            if entry is None:
                entry = show_entry(key, profile="framework-core")
                cache[key] = entry
            if entry is None:
                raise ValueError(f"{preset_key} references missing catalog entry '{key}' in relation '{relation_name}'")
            if str(entry.kind) != expected_kind:
                raise ValueError(
                    f"{preset_key} relation '{relation_name}' points to '{key}' with unexpected kind '{entry.kind}'"
                )
            fields = dict(entry.fields)
            missing = [field for field in required_fields_by_kind[expected_kind] if field not in fields]
            if missing:
                names = ", ".join(missing)
                raise ValueError(f"{key} is missing required mount contract fields: {names}")
            empty = [
                field
                for field in ("mount_plane", "mount_point", "orchestration_phases")
                if _is_empty_contract_value(fields.get(field))
            ]
            if empty:
                names = ", ".join(empty)
                raise ValueError(f"{key} has empty mount contract fields: {names}")


def validate_flow_assembly(spec: FlowAssemblySpec, config: MLBlackConfig | None = None) -> None:
    """Validate semantic boundaries for flow assembly.

    Rule: in semantic flow, numericization concerns must stay in `numericizer` layer.
    """

    trainer_params = dict(spec.trainer.trainer_params)
    overlap = sorted(k for k in trainer_params.keys() if k in SEMANTIC_NUMERICIZER_KEYS)
    if overlap:
        raise ValueError(
            "trainer_params contains semantic numericizer keys in semantic flow: "
            f"{overlap}. Move them to assembly.numericizer.params."
        )

    cfg = config or create_default_config()
    _validate_requested_trainer_registry_contract(spec, cfg)


def build_numericizer(spec: NumericizerSpec | None = None, config: MLBlackConfig | None = None):
    cfg = config or create_default_config()
    if spec is None:
        spec = NumericizerSpec()
    return cfg.numericizers.create(spec.key, **dict(spec.params))


def build_trainer(spec: TrainerAssemblySpec, config: MLBlackConfig | None = None):
    cfg = config or create_default_config()

    pipeline = cfg.pipelines.create(spec.pipeline_key, **dict(spec.pipeline_params))

    if spec.biases:
        bias_objs = [cfg.biases.create(b.key, **dict(b.params)) for b in spec.biases]
    else:
        bias_objs = [cfg.biases.create("noop")]

    trainer = cfg.trainers.create(
        spec.trainer_key,
        pipeline=pipeline,
        biases=bias_objs,
        config=dict(spec.trainer_params),
    )
    try:
        setattr(trainer, "requested_trainer_key", str(spec.trainer_key))
        setattr(trainer, "requested_trainer_params", dict(spec.trainer_params))
        setattr(trainer, "requested_pipeline_key", str(spec.pipeline_key))
    except Exception:
        pass
    return trainer


def build_capabilities(specs: Sequence[CapabilitySpec], config: MLBlackConfig | None = None) -> tuple[Any, ...]:
    cfg = config or create_default_config()
    if not specs:
        return tuple()

    out: list[Any] = []
    for spec in specs:
        out.append(cfg.capabilities.create(spec.key, **dict(spec.params)))
    return tuple(out)


def build_flow_components(spec: FlowAssemblySpec, config: MLBlackConfig | None = None) -> Dict[str, Any]:
    cfg = config or create_default_config()
    validate_flow_assembly(spec, config=cfg)

    trainer = build_trainer(spec.trainer, config=cfg)
    numericizer = None if spec.numericizer is None else build_numericizer(spec.numericizer, config=cfg)
    capabilities = build_capabilities(tuple(spec.capabilities), config=cfg)
    _validate_runtime_trainer_router_contract(spec, cfg, trainer)
    _validate_selected_capability_contracts(spec, capabilities)
    _validate_catalog_mount_contracts_for_preset(spec)
    return {
        "trainer": trainer,
        "numericizer": numericizer,
        "capabilities": capabilities,
    }


def list_registered(config: MLBlackConfig | None = None) -> Dict[str, Sequence[str]]:
    cfg = config or create_default_config()
    return {
        "pipelines": cfg.pipelines.keys(),
        "biases": cfg.biases.keys(),
        "numericizers": cfg.numericizers.keys(),
        "trainers": cfg.trainers.keys(),
        "capabilities": cfg.capabilities.keys(),
    }


def _trainer_supports_state_io(trainer: Any) -> bool:
    trainer_cls = type(trainer)
    return callable(getattr(trainer_cls, "save_trainer_state", None)) and callable(
        getattr(trainer_cls, "load_trainer_state", None)
    )


def _trainer_execution_resource_requests(trainer: Any) -> Dict[str, Any] | None:
    getter_many = getattr(trainer, "execution_resource_requests", None)
    components: list[Dict[str, Any]] = []
    if callable(getter_many):
        try:
            raw_many = tuple(getter_many())
            components = [coerce_execution_resource_request(item).as_dict() for item in raw_many]
        except Exception as exc:
            return {
                "error": f"{type(exc).__name__}: {exc}",
            }
    if components:
        total = sum_execution_resource_requests(
            tuple(coerce_execution_resource_request(item) for item in raw_many),
            label=str(getattr(trainer, "name", type(trainer).__name__)),
        )
        return {
            "request": total.as_dict(),
            "components": list(components),
        }

    getter = getattr(trainer, "execution_resource_request", None)
    if not callable(getter):
        return None
    try:
        request = coerce_execution_resource_request(getter())
    except Exception as exc:
        return {
            "error": f"{type(exc).__name__}: {exc}",
        }
    return {
        "request": request.as_dict(),
        "components": [request.as_dict()],
    }


def _extract_mechanism_bindings_from_mapping(value: Any) -> Any:
    if isinstance(value, Mapping):
        return value.get("mechanism_bindings")
    return None


def _extract_search_mechanism_contracts_from_mapping(value: Any) -> Any:
    if isinstance(value, Mapping):
        return value.get("search_mechanism_contracts")
    return None


def _resolve_trainer_mechanism_bindings(
    *,
    registry_meta: Dict[str, Any],
    capability_payload: Dict[str, Any],
    trainer: Any,
) -> list[dict[str, Any]]:
    raw = dict(capability_payload)
    trainer_tree_meta = getattr(trainer, "tree_family_metadata", None)
    trainer_symbolic_meta = getattr(trainer, "symbolic_family_metadata", None)

    candidates = (
        raw.get("mechanism_bindings"),
        _extract_mechanism_bindings_from_mapping(raw.get("tree_family")),
        _extract_mechanism_bindings_from_mapping(raw.get("symbolic_family")),
        _extract_mechanism_bindings_from_mapping(trainer_tree_meta),
        _extract_mechanism_bindings_from_mapping(trainer_symbolic_meta),
        registry_meta.get("mechanism_bindings"),
    )
    for candidate in candidates:
        if candidate is None:
            continue
        return serialize_mechanism_protocols(candidate)
    return []


def _resolve_trainer_search_mechanism_contracts(
    *,
    registry_meta: Dict[str, Any],
    capability_payload: Dict[str, Any],
    trainer: Any,
) -> list[dict[str, Any]]:
    from core.symbolic.search_mechanism_contract import serialize_symbolic_search_mechanism_contracts

    raw = dict(capability_payload)
    trainer_symbolic_meta = getattr(trainer, "symbolic_family_metadata", None)

    candidates = (
        raw.get("search_mechanism_contracts"),
        _extract_search_mechanism_contracts_from_mapping(raw.get("symbolic_family")),
        _extract_search_mechanism_contracts_from_mapping(trainer_symbolic_meta),
        registry_meta.get("search_mechanism_contracts"),
    )
    for candidate in candidates:
        if candidate is None:
            continue
        return serialize_symbolic_search_mechanism_contracts(candidate)
    return []


def _normalize_trainer_contract(
    *,
    registry_meta: Dict[str, Any],
    capability_payload: Dict[str, Any],
    trainer: Any,
) -> Dict[str, Any]:
    from training.capabilities import coerce_trainer_capabilities

    caps = coerce_trainer_capabilities(capability_payload)
    raw = dict(caps.metadata)
    runtime = dict(raw.get("runtime", {}) or {})
    supports = dict(raw.get("supports", {}) or {})
    artifacts = dict(raw.get("artifacts", {}) or {})
    has_state_io = _trainer_supports_state_io(trainer)
    execution_request = _trainer_execution_resource_requests(trainer)
    mechanism_bindings = _resolve_trainer_mechanism_bindings(
        registry_meta=registry_meta,
        capability_payload=capability_payload,
        trainer=trainer,
    )
    search_mechanism_contracts = _resolve_trainer_search_mechanism_contracts(
        registry_meta=registry_meta,
        capability_payload=capability_payload,
        trainer=trainer,
    )

    return {
        "trainer_key": str(getattr(trainer, "name", registry_meta.get("name", "")) or ""),
        "family": str(raw.get("model_family", registry_meta.get("family", "")) or ""),
        "backend": str(raw.get("backend", registry_meta.get("backend", "")) or ""),
        "nonlinear": bool(raw.get("nonlinear", registry_meta.get("nonlinear", False))),
        "training_modes": {
            "fresh": bool(caps.supports_fresh),
            "resume": bool(caps.supports_resume),
            "warm_start": bool(caps.supports_warm_start),
            "incremental": bool(caps.supports_incremental),
            "recalibrate": bool(caps.supports_recalibration),
        },
        "trainer_state": {
            "enabled": bool(runtime.get("trainer_state", False) or has_state_io),
            "save_load": bool(runtime.get("save_load_trainer_state", False) or has_state_io),
        },
        "execution_resources": {
            **({} if execution_request is None else dict(execution_request)),
        },
        "supports": supports,
        "artifacts": artifacts,
        "runtime": runtime,
        "mechanism_bindings": list(mechanism_bindings),
        "search_mechanism_contracts": list(search_mechanism_contracts),
    }


def _describe_trainer_entry(
    cfg: MLBlackConfig,
    key: str,
    *,
    include_dynamic: bool,
) -> Dict[str, Any]:
    registry_meta = cfg.trainers.metadata(key)
    item: Dict[str, Any] = {
        "key": str(key),
        "metadata": dict(registry_meta),
    }

    if not include_dynamic:
        return item

    try:
        trainer = cfg.trainers.create(
            key,
            pipeline=cfg.pipelines.create("identity"),
            biases=[cfg.biases.create("noop")],
            config={},
        )
        cap_fn = getattr(trainer, "capabilities", None)
        if callable(cap_fn):
            capability_payload = dict(cap_fn())
            contract = _normalize_trainer_contract(
                registry_meta=registry_meta,
                capability_payload=capability_payload,
                trainer=trainer,
            )
            item["capabilities"] = capability_payload
            item["contract"] = contract
            item["metadata"] = {
                **dict(registry_meta),
                "trainer_contract": dict(contract),
            }
    except Exception as exc:
        item["capabilities_error"] = f"{type(exc).__name__}: {exc}"

    return item


def describe_registered(config: MLBlackConfig | None = None) -> Dict[str, Sequence[Dict[str, Any]]]:
    cfg = config or create_default_config()
    return {
        "pipelines": cfg.pipelines.describe(),
        "biases": cfg.biases.describe(),
        "numericizers": cfg.numericizers.describe(),
        "trainers": tuple(_describe_trainer_entry(cfg, key, include_dynamic=True) for key in cfg.trainers.keys()),
        "capabilities": cfg.capabilities.describe(),
    }


def describe_trainers(
    config: MLBlackConfig | None = None,
    *,
    include_dynamic: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """Return trainer descriptions from registry metadata + normalized runtime contract."""
    cfg = config or create_default_config()
    out: Dict[str, Dict[str, Any]] = {}

    for key in cfg.trainers.keys():
        item = _describe_trainer_entry(cfg, key, include_dynamic=include_dynamic)
        out[key] = {
            "registry": dict(item.get("metadata", {})),
        }
        if "capabilities" in item:
            out[key]["capabilities"] = dict(item["capabilities"])
        if "contract" in item:
            out[key]["contract"] = dict(item["contract"])
        if "capabilities_error" in item:
            out[key]["capabilities_error"] = str(item["capabilities_error"])

    return out
