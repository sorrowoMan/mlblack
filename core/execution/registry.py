from __future__ import annotations

import concurrent.futures
from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Callable, Mapping, Sequence


ExecutorFactory = Callable[[int], Any]
DeviceTokenNormalizer = Callable[[str | int], str | None]
TorchDeviceResolver = Callable[[Any, str], Any | None]
DeviceDiscoverer = Callable[[Any | None], Sequence[str]]


def _normalize_text_sequence(values: Sequence[str], *, exclude: Sequence[str] = ()) -> tuple[str, ...]:
    seen = {str(item).strip().lower() for item in tuple(exclude)}
    out: list[str] = []
    for raw in tuple(values):
        key = str(raw).strip().lower()
        if not key or key in seen:
            continue
        out.append(key)
        seen.add(key)
    return tuple(out)


@dataclass(frozen=True)
class ExecutionBackendSpec:
    key: str
    supports_parallel: bool = False
    executor_factory: ExecutorFactory | None = None
    aliases: Sequence[str] = field(default_factory=tuple)
    description: str = ""
    supported_device_kinds: Sequence[str] = field(default_factory=lambda: ("cpu",))
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        key = str(self.key).strip().lower()
        object.__setattr__(self, "key", key)
        object.__setattr__(self, "aliases", _normalize_text_sequence(self.aliases, exclude=(key,)))
        object.__setattr__(
            self,
            "supported_device_kinds",
            _normalize_text_sequence(self.supported_device_kinds),
        )
        object.__setattr__(self, "description", str(self.description))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": str(self.key),
            "aliases": list(self.aliases),
            "supports_parallel": bool(self.supports_parallel),
            "description": str(self.description),
            "supported_device_kinds": list(self.supported_device_kinds),
            "metadata": dict(self.metadata),
        }


class ExecutionBackendRegistry:
    _GLOBAL: "ExecutionBackendRegistry | None" = None

    def __init__(self) -> None:
        self._lock = RLock()
        self._specs: dict[str, ExecutionBackendSpec] = {}
        self._aliases: dict[str, str] = {}

    @classmethod
    def global_registry(cls) -> "ExecutionBackendRegistry":
        if cls._GLOBAL is None:
            registry = cls()
            registry.register(
                ExecutionBackendSpec(
                    key="serial",
                    supports_parallel=False,
                    executor_factory=None,
                    aliases=("sync", "inline"),
                    description="Inline single-process execution on the caller thread.",
                    supported_device_kinds=("cpu", "cuda", "mps"),
                    metadata={"kind": "inline", "concurrency_model": "synchronous"},
                )
            )
            registry.register(
                ExecutionBackendSpec(
                    key="thread",
                    supports_parallel=True,
                    executor_factory=lambda max_workers: concurrent.futures.ThreadPoolExecutor(
                        max_workers=max_workers
                    ),
                    aliases=("threads", "thread_pool"),
                    description="Thread-pool execution for I/O-bound or GIL-friendly tasks.",
                    supported_device_kinds=("cpu", "cuda", "mps"),
                    metadata={"kind": "thread_pool", "concurrency_model": "shared_memory"},
                )
            )
            registry.register(
                ExecutionBackendSpec(
                    key="process",
                    supports_parallel=True,
                    executor_factory=lambda max_workers: concurrent.futures.ProcessPoolExecutor(
                        max_workers=max_workers
                    ),
                    aliases=("multiprocess", "process_pool"),
                    description="Process-pool execution for CPU-bound picklable tasks.",
                    supported_device_kinds=("cpu",),
                    metadata={"kind": "process_pool", "concurrency_model": "multi_process"},
                )
            )
            cls._GLOBAL = registry
        return cls._GLOBAL

    def register(self, spec: ExecutionBackendSpec) -> None:
        normalized = ExecutionBackendSpec(
            key=spec.key,
            supports_parallel=bool(spec.supports_parallel),
            executor_factory=spec.executor_factory,
            aliases=tuple(spec.aliases),
            description=str(spec.description),
            supported_device_kinds=tuple(spec.supported_device_kinds),
            metadata=dict(spec.metadata),
        )
        if not normalized.key:
            raise ValueError("Execution backend key must be non-empty")
        if bool(normalized.supports_parallel) and normalized.executor_factory is None:
            raise ValueError(f"Parallel backend '{normalized.key}' must provide executor_factory")
        with self._lock:
            self._drop_aliases_for_key(normalized.key)
            for alias in normalized.aliases:
                existing = self._aliases.get(alias)
                if existing is not None and existing != normalized.key:
                    raise ValueError(
                        f"Execution backend alias '{alias}' already registered for backend '{existing}'"
                    )
            self._specs[normalized.key] = normalized
            for alias in normalized.aliases:
                self._aliases[alias] = normalized.key

    def resolve(self, key: str | None) -> ExecutionBackendSpec:
        normalized = str(key or "serial").strip().lower() or "serial"
        with self._lock:
            canonical = self._aliases.get(normalized, normalized)
            spec = self._specs.get(canonical)
            allowed = tuple(sorted(self._specs))
        if spec is None:
            raise ValueError(f"Unsupported execution backend '{key}'. Allowed: {list(allowed)}")
        return spec

    def list_specs(self) -> tuple[ExecutionBackendSpec, ...]:
        with self._lock:
            items = tuple(self._specs.values())
        return tuple(sorted(items, key=lambda item: str(item.key)))

    def describe_specs(self) -> tuple[dict[str, Any], ...]:
        return tuple(spec.as_dict() for spec in self.list_specs())

    def list_keys(self) -> tuple[str, ...]:
        return tuple(str(spec.key) for spec in self.list_specs())

    def _drop_aliases_for_key(self, key: str) -> None:
        to_delete = [alias for alias, target in self._aliases.items() if target == key]
        for alias in to_delete:
            del self._aliases[alias]


@dataclass(frozen=True)
class ExecutionDeviceKindSpec:
    kind: str
    aliases: Sequence[str] = field(default_factory=tuple)
    description: str = ""
    example_tokens: Sequence[str] = field(default_factory=tuple)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        kind = str(self.kind).strip().lower()
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "aliases", _normalize_text_sequence(self.aliases, exclude=(kind,)))
        object.__setattr__(self, "example_tokens", tuple(str(x) for x in tuple(self.example_tokens)))
        object.__setattr__(self, "description", str(self.description))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def as_dict(
        self,
        *,
        available: bool | None = None,
        discovered_tokens: Sequence[str] = (),
    ) -> dict[str, Any]:
        return {
            "kind": str(self.kind),
            "aliases": list(self.aliases),
            "description": str(self.description),
            "example_tokens": list(self.example_tokens),
            "available": available,
            "discovered_tokens": [str(x) for x in tuple(discovered_tokens)],
            "metadata": dict(self.metadata),
        }


class ExecutionDeviceRegistry:
    _GLOBAL: "ExecutionDeviceRegistry | None" = None

    def __init__(self) -> None:
        self._lock = RLock()
        self._normalizers: list[tuple[str, DeviceTokenNormalizer]] = []
        self._torch_resolvers: list[tuple[str, TorchDeviceResolver]] = []
        self._discoverers: dict[str, DeviceDiscoverer] = {}
        self._kind_specs: dict[str, ExecutionDeviceKindSpec] = {}
        self._kind_aliases: dict[str, str] = {}

    @classmethod
    def global_registry(cls) -> "ExecutionDeviceRegistry":
        if cls._GLOBAL is None:
            registry = cls()
            registry.register_kind_spec(
                ExecutionDeviceKindSpec(
                    kind="cpu",
                    aliases=("host",),
                    description="Host CPU execution token.",
                    example_tokens=("cpu",),
                    metadata={"family": "host"},
                )
            )
            registry.register_kind_spec(
                ExecutionDeviceKindSpec(
                    kind="cuda",
                    aliases=("gpu", "nvidia"),
                    description="CUDA GPU execution token family.",
                    example_tokens=("cuda", "cuda:0", "cuda:1"),
                    metadata={"family": "gpu"},
                )
            )
            registry.register_kind_spec(
                ExecutionDeviceKindSpec(
                    kind="mps",
                    aliases=("metal",),
                    description="Apple Metal Performance Shaders execution token.",
                    example_tokens=("mps",),
                    metadata={"family": "gpu"},
                )
            )
            registry.register_token_normalizer("default", _default_device_token_normalizer)
            registry.register_torch_resolver("default", _default_torch_device_resolver)
            registry.register_discoverer("cpu", _default_cpu_discoverer)
            registry.register_discoverer("cuda", _default_cuda_discoverer)
            registry.register_discoverer("mps", _default_mps_discoverer)
            cls._GLOBAL = registry
        return cls._GLOBAL

    def register_kind_spec(self, spec: ExecutionDeviceKindSpec) -> None:
        normalized = ExecutionDeviceKindSpec(
            kind=spec.kind,
            aliases=tuple(spec.aliases),
            description=str(spec.description),
            example_tokens=tuple(spec.example_tokens),
            metadata=dict(spec.metadata),
        )
        if not normalized.kind:
            raise ValueError("device kind must be non-empty")
        with self._lock:
            self._drop_kind_aliases_for_kind(normalized.kind)
            for alias in normalized.aliases:
                existing = self._kind_aliases.get(alias)
                if existing is not None and existing != normalized.kind:
                    raise ValueError(f"device kind alias '{alias}' already registered for '{existing}'")
            self._kind_specs[normalized.kind] = normalized
            for alias in normalized.aliases:
                self._kind_aliases[alias] = normalized.kind

    def register_token_normalizer(self, name: str, fn: DeviceTokenNormalizer) -> None:
        with self._lock:
            self._normalizers.append((str(name), fn))

    def register_torch_resolver(self, name: str, fn: TorchDeviceResolver) -> None:
        with self._lock:
            self._torch_resolvers.append((str(name), fn))

    def register_discoverer(self, kind: str, fn: DeviceDiscoverer) -> None:
        key = self.resolve_kind_key(kind)
        with self._lock:
            self._discoverers[key] = fn

    def resolve_kind(self, kind: str | None) -> ExecutionDeviceKindSpec:
        key = self.resolve_kind_key(kind)
        with self._lock:
            spec = self._kind_specs.get(key)
            allowed = tuple(sorted(self._kind_specs))
        if spec is None:
            raise ValueError(f"Unsupported execution device kind '{kind}'. Allowed: {list(allowed)}")
        return spec

    def resolve_kind_key(self, kind: str | None) -> str:
        normalized = str(kind or "cpu").strip().lower() or "cpu"
        with self._lock:
            return self._kind_aliases.get(normalized, normalized)

    def list_kind_specs(self) -> tuple[ExecutionDeviceKindSpec, ...]:
        with self._lock:
            items = tuple(self._kind_specs.values())
        return tuple(sorted(items, key=lambda item: str(item.kind)))

    def list_kind_keys(self) -> tuple[str, ...]:
        return tuple(str(spec.kind) for spec in self.list_kind_specs())

    def describe_kinds(self, *, torch_module: Any | None = None) -> tuple[dict[str, Any], ...]:
        rows: list[dict[str, Any]] = []
        for spec in self.list_kind_specs():
            discovered = self.discover(spec.kind, torch_module=torch_module)
            available = bool(discovered)
            rows.append(spec.as_dict(available=available, discovered_tokens=discovered))
        return tuple(rows)

    def normalize_token(self, raw: str | int) -> str:
        with self._lock:
            normalizers = tuple(self._normalizers)
        for _name, fn in normalizers:
            token = fn(raw)
            if token is not None:
                return str(token)
        raise ValueError(f"Unsupported device token '{raw}'")

    def resolve_torch_device(self, torch_module: Any, requested: str | int) -> Any:
        token = self.normalize_token(requested)
        with self._lock:
            resolvers = tuple(self._torch_resolvers)
        for _name, fn in resolvers:
            device = fn(torch_module, token)
            if device is not None:
                return device
        raise ValueError(f"Unsupported torch device request '{requested}'")

    def discover(self, kind: str = "cuda", *, torch_module: Any | None = None) -> tuple[str, ...]:
        key = str(kind or "cuda").strip().lower() or "cuda"
        if key in {"all", "*"}:
            out: list[str] = []
            for spec in self.list_kind_specs():
                for token in self.discover(spec.kind, torch_module=torch_module):
                    if token not in out:
                        out.append(token)
            return tuple(out)

        resolved_kind = self.resolve_kind_key(key)
        with self._lock:
            fn = self._discoverers.get(resolved_kind)
        if fn is None:
            return tuple()
        out: list[str] = []
        for item in tuple(fn(torch_module)):
            token = self.normalize_token(item)
            if token not in out:
                out.append(token)
        return tuple(out)

    def _drop_kind_aliases_for_kind(self, kind: str) -> None:
        to_delete = [alias for alias, target in self._kind_aliases.items() if target == kind]
        for alias in to_delete:
            del self._kind_aliases[alias]


def _default_device_token_normalizer(raw: str | int) -> str | None:
    if isinstance(raw, int):
        if int(raw) < 0:
            raise ValueError(f"GPU device index must be >= 0, got: {raw}")
        return f"cuda:{int(raw)}"

    text = str(raw or "").strip().lower()
    if not text:
        raise ValueError("device token must be non-empty")

    if text in {"auto", "cpu", "cuda", "mps"}:
        return text
    if text == "gpu":
        return "cuda"
    if text in {"cpu:0", "mps:0"}:
        return text.split(":", 1)[0]
    if text.isdigit():
        return f"cuda:{int(text)}"
    if text.startswith("gpu:") and text.split(":", 1)[1].isdigit():
        return f"cuda:{int(text.split(':', 1)[1])}"
    if text.startswith("cuda:") and text.split(":", 1)[1].isdigit():
        return f"cuda:{int(text.split(':', 1)[1])}"
    if text.startswith("mps:") and text.split(":", 1)[1].isdigit():
        return "mps"
    return None


def _torch_has_cuda(torch_module: Any) -> bool:
    try:
        return bool(torch_module.cuda.is_available())
    except Exception:
        return False


def _torch_cuda_device_count(torch_module: Any) -> int:
    try:
        return int(torch_module.cuda.device_count())
    except Exception:
        return 0


def _torch_has_mps(torch_module: Any) -> bool:
    try:
        backends = getattr(torch_module, "backends", None)
        mps = None if backends is None else getattr(backends, "mps", None)
        if mps is None:
            return False
        return bool(mps.is_available())
    except Exception:
        return False


def _default_torch_device_resolver(torch_module: Any, token: str) -> Any | None:
    key = str(token).strip().lower()
    if key == "auto":
        if _torch_has_cuda(torch_module):
            return torch_module.device("cuda")
        if _torch_has_mps(torch_module):
            return torch_module.device("mps")
        return torch_module.device("cpu")
    if key == "cpu":
        return torch_module.device("cpu")
    if key == "mps":
        if not _torch_has_mps(torch_module):
            raise RuntimeError("MPS requested but not available")
        return torch_module.device("mps")
    if key == "cuda":
        if not _torch_has_cuda(torch_module):
            raise RuntimeError("CUDA requested but not available")
        return torch_module.device("cuda")
    if key.startswith("cuda:") and key.split(":", 1)[1].isdigit():
        if not _torch_has_cuda(torch_module):
            raise RuntimeError(f"CUDA device '{token}' requested but CUDA is not available")
        index = int(key.split(":", 1)[1])
        device_count = _torch_cuda_device_count(torch_module)
        if index < 0 or index >= device_count:
            raise RuntimeError(
                f"CUDA device index out of range: requested={token}, available=0..{max(0, device_count - 1)}"
            )
        return torch_module.device(f"cuda:{index}")
    return None


def _default_cpu_discoverer(torch_module: Any | None) -> Sequence[str]:
    _ = torch_module
    return ("cpu",)


def _default_cuda_discoverer(torch_module: Any | None) -> Sequence[str]:
    module = torch_module
    if module is None:
        try:
            import torch as module  # type: ignore[assignment]
        except Exception:
            return tuple()
    if not _torch_has_cuda(module):
        return tuple()
    count = _torch_cuda_device_count(module)
    return tuple(f"cuda:{idx}" for idx in range(max(0, count)))


def _default_mps_discoverer(torch_module: Any | None) -> Sequence[str]:
    module = torch_module
    if module is None:
        try:
            import torch as module  # type: ignore[assignment]
        except Exception:
            return tuple()
    if not _torch_has_mps(module):
        return tuple()
    return ("mps",)


def global_execution_backend_registry() -> ExecutionBackendRegistry:
    return ExecutionBackendRegistry.global_registry()


def global_execution_device_registry() -> ExecutionDeviceRegistry:
    return ExecutionDeviceRegistry.global_registry()


def list_registered_execution_backends() -> tuple[ExecutionBackendSpec, ...]:
    return global_execution_backend_registry().list_specs()


def list_registered_execution_backend_keys() -> tuple[str, ...]:
    return global_execution_backend_registry().list_keys()


def describe_registered_execution_backends() -> tuple[dict[str, Any], ...]:
    return global_execution_backend_registry().describe_specs()


def list_registered_execution_device_kinds() -> tuple[str, ...]:
    return global_execution_device_registry().list_kind_keys()


def describe_registered_execution_device_kinds(
    *,
    torch_module: Any | None = None,
) -> tuple[dict[str, Any], ...]:
    return global_execution_device_registry().describe_kinds(torch_module=torch_module)


def normalize_execution_device_token(raw: str | int) -> str:
    return global_execution_device_registry().normalize_token(raw)


def discover_execution_devices(kind: str = "cuda", *, torch_module: Any | None = None) -> tuple[str, ...]:
    return global_execution_device_registry().discover(kind, torch_module=torch_module)


def resolve_torch_execution_device(torch_module: Any, requested: str | int) -> Any:
    return global_execution_device_registry().resolve_torch_device(torch_module, requested)


__all__ = [
    "ExecutionBackendRegistry",
    "ExecutionBackendSpec",
    "ExecutionDeviceKindSpec",
    "ExecutionDeviceRegistry",
    "describe_registered_execution_backends",
    "describe_registered_execution_device_kinds",
    "discover_execution_devices",
    "global_execution_backend_registry",
    "global_execution_device_registry",
    "list_registered_execution_backend_keys",
    "list_registered_execution_backends",
    "list_registered_execution_device_kinds",
    "normalize_execution_device_token",
    "resolve_torch_execution_device",
]
