from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class BackendCapabilityContract:
    """Catalog-facing contract for one backend capability component.

    The catalog indexes capability components, not every small function. The
    method map records the callable surface exposed by that component so
    assembly/doctor can explain exactly which backend method is missing.
    """

    backend: str
    capability: str
    provides: tuple[str, ...]
    requires: tuple[str, ...] = ()
    optional: tuple[str, ...] = ()
    methods: Mapping[str, str] = field(default_factory=dict)
    tensor_kinds: tuple[str, ...] = ()
    model_kinds: tuple[str, ...] = ()
    routes: tuple[str, ...] = ()
    heads: tuple[str, ...] = ()
    supports_autograd: bool = False
    supports_stateful_module: bool = False
    supports_functional_params: bool = False
    supports_gpu: bool = False
    supports_resume: bool = False
    notes: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "capability": self.capability,
            "provides": tuple(self.provides),
            "requires": tuple(self.requires),
            "optional": tuple(self.optional),
            "methods": dict(self.methods),
            "tensor_kinds": tuple(self.tensor_kinds),
            "model_kinds": tuple(self.model_kinds),
            "routes": tuple(self.routes),
            "heads": tuple(self.heads),
            "supports_autograd": bool(self.supports_autograd),
            "supports_stateful_module": bool(self.supports_stateful_module),
            "supports_functional_params": bool(self.supports_functional_params),
            "supports_gpu": bool(self.supports_gpu),
            "supports_resume": bool(self.supports_resume),
            "notes": str(self.notes),
        }


@dataclass(frozen=True)
class BackendContract:
    """Aggregated backend contract built from capability components."""

    name: str
    capabilities: tuple[BackendCapabilityContract, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def provides(self) -> tuple[str, ...]:
        seen: set[str] = set()
        values: list[str] = []
        for capability in self.capabilities:
            for item in capability.provides:
                key = str(item)
                if key not in seen:
                    seen.add(key)
                    values.append(key)
        return tuple(values)

    @property
    def methods(self) -> Mapping[str, str]:
        methods: dict[str, str] = {}
        for capability in self.capabilities:
            methods.update({str(key): str(value) for key, value in capability.methods.items()})
        return methods

    def supports(self, requirement: str) -> bool:
        key = str(requirement)
        return key in set(self.provides) or key in set(self.methods)

    def missing(self, requirements: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(str(item) for item in requirements if not self.supports(str(item)))

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "provides": tuple(self.provides),
            "methods": dict(self.methods),
            "capabilities": tuple(item.as_dict() for item in self.capabilities),
            "metadata": dict(self.metadata),
        }


def ensure_backend_supports(backend: Any, requirements: tuple[str, ...], *, consumer: str = "") -> None:
    """Fail fast with a readable backend capability error."""

    contract = backend.contract() if hasattr(backend, "contract") else None
    if contract is None:
        raise TypeError(f"backend {backend!r} does not expose contract()")
    missing = contract.missing(tuple(str(item) for item in requirements))
    if missing:
        label = f" for {consumer}" if consumer else ""
        raise ValueError(
            f"backend {contract.name!r} is missing required capabilities{label}: "
            + ", ".join(str(item) for item in missing)
        )


__all__ = ["BackendCapabilityContract", "BackendContract", "ensure_backend_supports"]
