from __future__ import annotations

from dataclasses import dataclass, field
from time import time
from typing import Any, Mapping
from uuid import uuid4


@dataclass(frozen=True)
class ResourceContext:
    """Passive resource grant injected by an outer orchestrator.

    mlblack does not allocate resources. nsgablack owns authorization, leases,
    backend selection and scheduling. This object only normalizes the
    JSON-compatible context that inner ML components may read and audit.
    """

    scope: str = "training"
    execution_backend: str = "outer"
    compute_backend: str = "auto"
    device: str = "cpu"
    threads: int = 1
    nested: bool = False
    namespace: str = ""
    grant: Mapping[str, Any] = field(default_factory=dict)
    lease: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "scope", str(self.scope or "training"))
        object.__setattr__(self, "execution_backend", str(self.execution_backend or "outer"))
        object.__setattr__(self, "compute_backend", str(self.compute_backend or "auto"))
        object.__setattr__(self, "device", str(self.device or "cpu"))
        object.__setattr__(self, "threads", max(1, int(self.threads or 1)))
        object.__setattr__(self, "nested", bool(self.nested))
        object.__setattr__(self, "namespace", str(self.namespace or ""))
        object.__setattr__(self, "grant", dict(self.grant or {}))
        object.__setattr__(self, "lease", dict(self.lease or {}))
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | "ResourceContext" | None) -> "ResourceContext":
        if isinstance(value, ResourceContext):
            return value
        payload = dict(value or {})
        grant = dict(payload.get("grant", {}) or {})
        lease = dict(payload.get("lease", payload.get("resource_lease", {})) or {})
        tokens = tuple(payload.get("device_tokens", grant.get("device_tokens", lease.get("device_tokens", ()))) or ())
        raw_device = payload.get("device", grant.get("device", ""))
        device = str(raw_device or "").strip()
        if device.lower() in {"", "auto", "none"}:
            device = str(tokens[0]) if tokens else "cpu"
        threads = payload.get("threads", grant.get("threads", lease.get("threads", 1)))
        return cls(
            scope=str(payload.get("scope", grant.get("phase", lease.get("scope", "training")))),
            execution_backend=str(payload.get("execution_backend", payload.get("backend", grant.get("backend", "outer")))),
            compute_backend=str(payload.get("compute_backend", grant.get("compute_backend", "auto"))),
            device=device,
            threads=int(threads or 1),
            nested=bool(payload.get("nested", bool(lease))),
            namespace=str(payload.get("namespace", grant.get("label", lease.get("lease_id", "")))),
            grant=grant,
            lease=lease,
            metadata=dict(payload.get("metadata", {}) or {}),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "scope": self.scope,
            "execution_backend": self.execution_backend,
            "compute_backend": self.compute_backend,
            "device": self.device,
            "threads": int(self.threads),
            "nested": bool(self.nested),
            "namespace": self.namespace,
            "grant": dict(self.grant),
            "lease": dict(self.lease),
            "metadata": dict(self.metadata),
        }

    def context_items(self, *, prefix: str = "resource") -> dict[str, Any]:
        base = str(prefix or "resource")
        return {
            base: self.as_dict(),
            f"{base}.context": self.as_dict(),
            f"{base}.scope": self.scope,
            f"{base}.execution_backend": self.execution_backend,
            f"{base}.compute_backend": self.compute_backend,
            f"{base}.device": self.device,
            f"{base}.threads": int(self.threads),
            f"{base}.nested": bool(self.nested),
            f"{base}.namespace": self.namespace,
            f"{base}.lease": dict(self.lease),
        }


@dataclass(frozen=True)
class ResourceEvent:
    """Audit-only resource event; not a scheduling primitive."""

    topic: str
    payload: Mapping[str, Any] = field(default_factory=dict)
    event_id: str = field(default_factory=lambda: f"evt_{uuid4().hex[:16]}")
    created_at: float = field(default_factory=time)

    def as_dict(self) -> dict[str, Any]:
        return {
            "event_id": str(self.event_id),
            "topic": str(self.topic),
            "payload": dict(self.payload),
            "created_at": float(self.created_at),
        }


class ResourceAudit:
    """Small in-memory audit log for externally injected ResourceContext."""

    def __init__(self) -> None:
        self._events: list[ResourceEvent] = []

    def record(self, topic: str, payload: Mapping[str, Any] | None = None) -> ResourceEvent:
        event = ResourceEvent(topic=str(topic), payload=dict(payload or {}))
        self._events.append(event)
        return event

    def events(self, *, topic: str | None = None, limit: int = 100) -> tuple[ResourceEvent, ...]:
        rows = self._events if topic is None else [event for event in self._events if event.topic == topic]
        return tuple(rows[-max(0, int(limit)):])

    def as_dict(self) -> dict[str, Any]:
        return {"events": [event.as_dict() for event in self._events]}


def coerce_resource_context(value: Mapping[str, Any] | ResourceContext | None) -> ResourceContext:
    return ResourceContext.from_mapping(value)
