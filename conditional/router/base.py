from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence, runtime_checkable


RouteKey = tuple[int, ...]


@runtime_checkable
class RouteCanonicalizer(Protocol):
    def normalize_key(self, raw_key: Sequence[int], *, width: int | None = None) -> RouteKey: ...

    def canonicalize(self, raw_key: Sequence[int], *, policy: Any) -> RouteKey: ...


@dataclass(frozen=True)
class DefaultRouteCanonicalizer:
    """Generic canonicalizer for sample-level routing policies."""

    def normalize_key(self, raw_key: Sequence[int], *, width: int | None = None) -> RouteKey:
        seq = tuple(int(v > 0) for v in raw_key)
        if width is None:
            return seq
        if len(seq) >= int(width):
            return tuple(seq[: int(width)])
        padded = list(seq)
        while len(padded) < int(width):
            padded.append(0)
        return tuple(padded[: int(width)])

    def canonicalize(self, raw_key: Sequence[int], *, policy: Any) -> RouteKey:
        gate_names = tuple(str(v) for v in getattr(policy, "gate_names", ()))
        route_order = tuple(tuple(int(x) for x in row) for row in getattr(policy, "route_order", ()))
        key = self.normalize_key(raw_key, width=len(gate_names) if gate_names else None)
        if key in route_order:
            return key
        default_route = getattr(policy, "default_route", None)
        if default_route is not None:
            return self.normalize_key(default_route, width=len(gate_names) if gate_names else len(default_route))
        if route_order:
            return tuple(route_order[-1])
        if gate_names:
            return tuple(0 for _ in gate_names)
        return key


__all__ = ["DefaultRouteCanonicalizer", "RouteCanonicalizer", "RouteKey"]
