from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from conditional.router.base import DefaultRouteCanonicalizer, RouteCanonicalizer, RouteKey


def _coerce_route_order(value: Any) -> tuple[RouteKey, ...]:
    rows = tuple(value or ())
    return tuple(tuple(int(x) for x in row) for row in rows)


def _coerce_gate_names(value: Any) -> tuple[str, ...]:
    return tuple(str(v) for v in tuple(value or ()))


@dataclass(frozen=True)
class RouterPolicyAdapter:
    """
    Generic sample-router policy.

    This adapter intentionally accepts existing regime-style policies such as
    `TrafficHolidayRegimePolicy` by normalizing `regime_order -> route_order`.
    """

    gate_names: tuple[str, ...]
    route_order: tuple[RouteKey, ...]
    canonicalizer: RouteCanonicalizer = field(default_factory=DefaultRouteCanonicalizer)
    default_route: RouteKey | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        gate_names = _coerce_gate_names(self.gate_names)
        route_order = _coerce_route_order(self.route_order)
        default_route = None if self.default_route is None else tuple(int(x) for x in self.default_route)
        if default_route is None:
            if route_order:
                default_route = tuple(route_order[-1])
            else:
                default_route = tuple(0 for _ in gate_names)
        object.__setattr__(self, "gate_names", gate_names)
        object.__setattr__(self, "route_order", route_order)
        object.__setattr__(self, "default_route", tuple(default_route))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def normalize_key(self, raw_key: RouteKey | tuple[int, ...]) -> RouteKey:
        return self.canonicalizer.normalize_key(raw_key, width=len(self.gate_names))

    def canonicalize(self, raw_key: RouteKey | tuple[int, ...]) -> RouteKey:
        return self.canonicalizer.canonicalize(raw_key, policy=self)

    @property
    def regime_order(self) -> tuple[RouteKey, ...]:
        value = self.metadata.get("regime_order")
        return _coerce_route_order(value) if value is not None else self.route_order

    @property
    def holiday_keys(self) -> tuple[RouteKey, ...]:
        value = self.metadata.get("holiday_keys")
        return _coerce_route_order(value) if value is not None else ()


def adapt_router_policy(policy: Any | None) -> RouterPolicyAdapter:
    if isinstance(policy, RouterPolicyAdapter):
        return policy
    if policy is None:
        return RouterPolicyAdapter(gate_names=(), route_order=())

    gate_names = _coerce_gate_names(getattr(policy, "gate_names", ()))
    route_order = _coerce_route_order(
        getattr(policy, "route_order", None) if hasattr(policy, "route_order") else getattr(policy, "regime_order", ())
    )
    default_route = getattr(policy, "default_route", None)
    canonicalizer = getattr(policy, "canonicalizer", DefaultRouteCanonicalizer())
    metadata: dict[str, Any] = {}
    if hasattr(policy, "holiday_keys"):
        metadata["holiday_keys"] = _coerce_route_order(getattr(policy, "holiday_keys"))
    if hasattr(policy, "regime_order"):
        metadata["regime_order"] = _coerce_route_order(getattr(policy, "regime_order"))
    if hasattr(policy, "route_metadata"):
        try:
            metadata.update(dict(getattr(policy, "route_metadata")))
        except Exception:
            pass
    return RouterPolicyAdapter(
        gate_names=gate_names,
        route_order=route_order,
        canonicalizer=canonicalizer,
        default_route=default_route,
        metadata=metadata,
    )


__all__ = ["RouterPolicyAdapter", "adapt_router_policy"]
