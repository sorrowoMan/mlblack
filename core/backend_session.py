from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from mlblack.backends import get_backend, resolve_backend


@dataclass(frozen=True)
class ComputeBackendSpec:
    """Trainer/L0-level compute backend declaration.

    This is not a resource allocator. It is the inner trainer's fixed compute
    execution context after nsgablack/resource policy has authorized resources.
    """

    name: str = "auto"
    device: str = "cpu"
    device_policy: str = "fallback_cpu"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_value(
        cls,
        value: str | Mapping[str, Any] | "ComputeBackendSpec" | "ComputeBackendSession" | None,
        *,
        resource_context: Mapping[str, Any] | None = None,
    ) -> "ComputeBackendSpec":
        if isinstance(value, ComputeBackendSession):
            return value.spec
        if isinstance(value, ComputeBackendSpec):
            return value
        resource = dict(resource_context or {})
        if isinstance(value, str):
            payload: dict[str, Any] = {"name": value}
        else:
            payload = dict(value or {})
        raw_name = payload.get(
            "name",
            payload.get("requested_name", payload.get("backend", payload.get("compute_backend"))),
        )
        if raw_name is None:
            raw_name = _resource_backend_name(resource)
        raw_device = payload.get("device", resource.get("device", "cpu"))
        return cls(
            name=str(raw_name or "auto"),
            device=str(raw_device or "cpu"),
            device_policy=str(payload.get("device_policy", payload.get("policy", "fallback_cpu"))),
            metadata=dict(payload.get("metadata", {}) or {}),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "device": self.device,
            "device_policy": self.device_policy,
            "metadata": dict(self.metadata),
        }


class ComputeBackendSession:
    """A fixed compute backend session consumed by representation/problem/adapter."""

    def __init__(self, spec: ComputeBackendSpec | str | Mapping[str, Any] | None = None) -> None:
        self.spec = ComputeBackendSpec.from_value(spec)
        self._backend: Any | None = None

    @property
    def requested_name(self) -> str:
        return str(self.spec.name or "auto")

    @property
    def device(self) -> str:
        return str(self.spec.device or "cpu")

    @property
    def device_policy(self) -> str:
        return str(self.spec.device_policy or "fallback_cpu")

    @property
    def resolved_name(self) -> str:
        if self._backend is not None:
            return str(self._backend.contract().name)
        return self.requested_name

    def ensure(self, requirements: tuple[str, ...] = (), *, consumer: str = "") -> Any:
        required = tuple(str(item) for item in requirements)
        if self._backend is None:
            self._backend = self._resolve(required, consumer=consumer)
        missing = self._backend.contract().missing(required)
        if missing:
            label = f" for {consumer}" if consumer else ""
            raise ValueError(
                f"compute backend {self._backend.contract().name!r} is missing required capabilities{label}: "
                + ", ".join(str(item) for item in missing)
            )
        return self._backend

    def contract_dict(self) -> dict[str, Any] | None:
        if self._backend is None:
            return None
        return self._backend.contract().as_dict()

    def as_dict(self) -> dict[str, Any]:
        return {
            "requested_name": self.requested_name,
            "resolved_name": self.resolved_name,
            "device": self.device,
            "device_policy": self.device_policy,
            "resolved": self._backend is not None,
            "contract": self.contract_dict(),
            "metadata": dict(self.spec.metadata),
        }

    def context_items(self) -> dict[str, Any]:
        return {
            "backend.session": self,
            "backend.name": self.resolved_name,
            "backend.requested_name": self.requested_name,
            "backend.device": self.device,
            "backend.device_policy": self.device_policy,
            "backend.contract": self.contract_dict(),
        }

    def close(self) -> None:
        """Release backend-local resources when a session is replaced."""
        backend = self._backend
        if backend is None:
            return
        for name in ("close", "shutdown", "teardown"):
            fn = getattr(backend, name, None)
            if callable(fn):
                fn()
                break
        self._backend = None

    def _resolve(self, requirements: tuple[str, ...], *, consumer: str = "") -> Any:
        name = self.requested_name.strip().lower()
        if name in {"", "auto"}:
            return resolve_backend(requirements)
        backend = get_backend(name)
        missing = backend.contract().missing(requirements)
        if missing:
            label = f" for {consumer}" if consumer else ""
            raise ValueError(
                f"compute backend {backend.contract().name!r} is missing required capabilities{label}: "
                + ", ".join(str(item) for item in missing)
            )
        return backend


def get_compute_backend_from_context(
    context: Mapping[str, Any] | None,
    requirements: tuple[str, ...] = (),
    *,
    consumer: str = "",
) -> Any:
    ctx = dict(context or {})
    session = ctx.get("backend.session")
    if isinstance(session, ComputeBackendSession):
        return session.ensure(requirements, consumer=consumer)
    label = f" for {consumer}" if consumer else ""
    raise ValueError(
        f"compute backend session is required{label}. "
        "Use LearningSolver.compute_backend_session / LearningSolver.build_context(); "
        "ad-hoc backend.name fallback is disabled."
    )


def _resource_backend_name(resource: Mapping[str, Any]) -> str:
    raw = str(resource.get("compute_backend", "auto") or "auto").strip().lower()
    if raw in {"", "auto", "cpu", "cuda", "gpu", "tpu"}:
        return "auto"
    return raw


__all__ = [
    "ComputeBackendSession",
    "ComputeBackendSpec",
    "get_compute_backend_from_context",
]
