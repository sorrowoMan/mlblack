from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from conditional.router.base import RouteKey
from conditional.router.policy import RouterPolicyAdapter, adapt_router_policy


def _empty_policy() -> RouterPolicyAdapter:
    return RouterPolicyAdapter(gate_names=(), route_order=())


@dataclass(frozen=True)
class RouteResolution:
    enabled: bool
    gate_idx: tuple[int, ...] | None
    policy: RouterPolicyAdapter = field(default_factory=_empty_policy)


def resolve_router(
    feature_names: Sequence[str],
    enabled: bool,
    *,
    policy: RouterPolicyAdapter | object | None,
) -> RouteResolution:
    adapted = adapt_router_policy(policy)
    names = tuple(str(v) for v in feature_names)
    if not bool(enabled) or not adapted.gate_names:
        return RouteResolution(enabled=False, gate_idx=None, policy=adapted)
    gate_idx_list = [names.index(name) for name in adapted.gate_names if name in names]
    if len(gate_idx_list) != len(adapted.gate_names):
        return RouteResolution(enabled=False, gate_idx=None, policy=adapted)
    return RouteResolution(
        enabled=True,
        gate_idx=tuple(int(v) for v in gate_idx_list),
        policy=adapted,
    )


def normalize_route_key(
    raw_key: Sequence[int],
    *,
    policy: RouterPolicyAdapter | object | None,
) -> RouteKey:
    adapted = adapt_router_policy(policy)
    return adapted.normalize_key(tuple(int(v) for v in raw_key))


def map_to_route(
    raw_key: Sequence[int],
    *,
    policy: RouterPolicyAdapter | object | None,
) -> RouteKey:
    adapted = adapt_router_policy(policy)
    return adapted.canonicalize(tuple(int(v) for v in raw_key))


def route_keys_from_matrix(
    X: np.ndarray,
    gate_idx: Sequence[int],
    *,
    policy: RouterPolicyAdapter | object | None,
) -> tuple[RouteKey, ...]:
    adapted = adapt_router_policy(policy)
    x = np.asarray(X, dtype=float)
    indices = tuple(int(v) for v in gate_idx)
    out: list[RouteKey] = []
    for row in range(int(x.shape[0])):
        raw = tuple(int(x[row, idx] > 0.5) for idx in indices)
        out.append(adapted.canonicalize(raw))
    return tuple(out)


__all__ = [
    "RouteResolution",
    "map_to_route",
    "normalize_route_key",
    "resolve_router",
    "route_keys_from_matrix",
]
