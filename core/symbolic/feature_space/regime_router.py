from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence, cast, runtime_checkable

import numpy as np
from conditional.router import (
    DefaultRouteCanonicalizer,
    RouterPolicyAdapter,
    map_to_route,
    normalize_route_key,
    resolve_router,
    route_keys_from_matrix,
)
from conditional.router.policy import adapt_router_policy

Fixed4RegimeKey = tuple[int, int, int, int]


# Core keeps only a generic strict4 gate contract.
# Domain/task-specific feature names must be injected from upper config / assembly.
STRICT4_GATE_NAMES: tuple[str, str, str, str] = (
    "strict4_gate_0",
    "strict4_gate_1",
    "strict4_gate_2",
    "strict4_gate_3",
)

STRICT4_REGIME_ORDER: tuple[tuple[int, int, int, int], ...] = (
    (1, 1, 0, 0),
    (1, 0, 1, 0),
    (0, 0, 0, 1),
    (0, 0, 0, 0),
)

STRICT4_HOLIDAY_KEYS: tuple[tuple[int, int, int, int], ...] = (
    (1, 1, 0, 0),
    (1, 0, 1, 0),
)


@runtime_checkable
class RegimeCanonicalizer(Protocol):
    def normalize_key(self, raw_key: Sequence[int]) -> Fixed4RegimeKey: ...

    def canonicalize(
        self,
        raw_key: Sequence[int],
        *,
        policy: RegimePolicy,
    ) -> Fixed4RegimeKey: ...


@runtime_checkable
class RegimePolicy(Protocol):
    gate_names: tuple[str, str, str, str]
    regime_order: tuple[Fixed4RegimeKey, ...]
    holiday_keys: tuple[Fixed4RegimeKey, ...]
    canonicalizer: RegimeCanonicalizer


@dataclass(frozen=True)
class DefaultFixed4RegimeCanonicalizer(DefaultRouteCanonicalizer):
    def normalize_key(self, raw_key: Sequence[int]) -> Fixed4RegimeKey:
        return cast(Fixed4RegimeKey, super().normalize_key(raw_key, width=4))

    def canonicalize(
        self,
        raw_key: Sequence[int],
        *,
        policy: RegimePolicy,
    ) -> Fixed4RegimeKey:
        key = self.normalize_key(raw_key)
        if key in policy.regime_order:
            return key
        zero = self.normalize_key((0, 0, 0, 0))
        if zero in policy.regime_order:
            return zero
        if policy.regime_order:
            return tuple(policy.regime_order[-1])
        return zero


@dataclass(frozen=True)
class Strict4RouterSpec:
    gate_names: tuple[str, str, str, str] = STRICT4_GATE_NAMES
    regime_order: tuple[Fixed4RegimeKey, ...] = STRICT4_REGIME_ORDER
    holiday_keys: tuple[Fixed4RegimeKey, ...] = STRICT4_HOLIDAY_KEYS
    canonicalizer: RegimeCanonicalizer = field(default_factory=DefaultFixed4RegimeCanonicalizer)


@dataclass(frozen=True)
class RegimeResolution:
    enabled: bool
    gate_idx: tuple[int, int, int, int] | None
    regime_policy: RegimePolicy = field(default_factory=Strict4RouterSpec)

    @property
    def router_spec(self) -> RegimePolicy:
        return self.regime_policy


Strict4Resolution = RegimeResolution


@dataclass(frozen=True)
class BranchTrainSelection:
    train_used: np.ndarray
    train_source: str
    min_train_effective: int
    use_branch: bool


def _resolve_regime_policy(router_spec: RegimePolicy | None) -> RegimePolicy:
    return router_spec if router_spec is not None else Strict4RouterSpec()


def _adapt_regime_policy(router_spec: RegimePolicy | None) -> RouterPolicyAdapter:
    return adapt_router_policy(_resolve_regime_policy(router_spec))


def resolve_regime(
    feature_names: Sequence[str],
    enabled: bool,
    *,
    regime_policy: RegimePolicy | None = None,
) -> RegimeResolution:
    spec = _resolve_regime_policy(regime_policy)
    resolution = resolve_router(
        feature_names,
        enabled,
        policy=spec,
    )
    gate_idx = None if resolution.gate_idx is None else tuple(int(v) for v in resolution.gate_idx[:4])
    if gate_idx is not None and len(gate_idx) != 4:
        gate_idx = None
    return RegimeResolution(
        enabled=bool(resolution.enabled and gate_idx is not None),
        gate_idx=cast(tuple[int, int, int, int] | None, gate_idx),
        regime_policy=spec,
    )


def resolve_strict4(
    feature_names: Sequence[str],
    enabled: bool,
    *,
    router_spec: RegimePolicy | None = None,
) -> Strict4Resolution:
    return resolve_regime(
        feature_names,
        enabled,
        regime_policy=router_spec,
    )


def normalize_regime_key(
    raw_key: tuple[int, ...],
    *,
    regime_policy: RegimePolicy | None = None,
) -> Fixed4RegimeKey:
    return cast(Fixed4RegimeKey, normalize_route_key(raw_key, policy=_adapt_regime_policy(regime_policy)))


def normalize_fixed4_key(
    raw_key: tuple[int, ...],
    *,
    router_spec: RegimePolicy | None = None,
) -> Fixed4RegimeKey:
    return normalize_regime_key(raw_key, regime_policy=router_spec)


def map_to_regime(
    raw_key: tuple[int, ...],
    *,
    regime_policy: RegimePolicy | None = None,
) -> Fixed4RegimeKey:
    return cast(Fixed4RegimeKey, map_to_route(raw_key, policy=_adapt_regime_policy(regime_policy)))


def map_to_strict4_regime(
    raw_key: tuple[int, ...],
    *,
    router_spec: RegimePolicy | None = None,
) -> Fixed4RegimeKey:
    return map_to_regime(raw_key, regime_policy=router_spec)


def regime_keys_from_X(
    X: np.ndarray,
    gate_idx: tuple[int, int, int, int],
    *,
    regime_policy: RegimePolicy | None = None,
) -> tuple[Fixed4RegimeKey, ...]:
    keys = route_keys_from_matrix(
        np.asarray(X, dtype=float),
        gate_idx,
        policy=_adapt_regime_policy(regime_policy),
    )
    return tuple(cast(Fixed4RegimeKey, tuple(int(x) for x in key[:4])) for key in keys)


def strict4_keys_from_X(
    X: np.ndarray,
    gate_idx: tuple[int, int, int, int],
    *,
    router_spec: RegimePolicy | None = None,
) -> tuple[Fixed4RegimeKey, ...]:
    return regime_keys_from_X(X, gate_idx, regime_policy=router_spec)


def build_regime_index(
    keys: Sequence[Fixed4RegimeKey],
    regimes: Sequence[Fixed4RegimeKey] | None = None,
    *,
    router_spec: RegimePolicy | None = None,
) -> dict[Fixed4RegimeKey, np.ndarray]:
    spec = _adapt_regime_policy(router_spec)
    resolved_regimes = tuple(regimes) if regimes is not None else tuple(cast(Sequence[Fixed4RegimeKey], spec.route_order))
    return {
        tuple(regime): np.asarray([idx for idx, key in enumerate(keys) if tuple(key) == tuple(regime)], dtype=int)
        for regime in resolved_regimes
    }


def holiday_union_indices(
    idx_by_key: dict[Fixed4RegimeKey, np.ndarray],
    holiday_keys: Sequence[Fixed4RegimeKey] | None = None,
    *,
    router_spec: RegimePolicy | None = None,
) -> np.ndarray:
    spec = _adapt_regime_policy(router_spec)
    resolved_holiday_keys = (
        tuple(holiday_keys)
        if holiday_keys is not None
        else tuple(cast(Sequence[Fixed4RegimeKey], spec.holiday_keys))
    )
    merged: set[int] = set()
    for key in resolved_holiday_keys:
        merged |= set(np.asarray(idx_by_key.get(tuple(key), np.asarray([], dtype=int)), dtype=int).tolist())
    return np.asarray(sorted(merged), dtype=int)


def resolve_branch_train_selection(
    *,
    regime: Fixed4RegimeKey,
    idx_tr_by_key: dict[Fixed4RegimeKey, np.ndarray],
    regime_min_branch_train: int,
    base_regime_min_branch_train: int,
    total_train_size: int,
    holiday_keys: Sequence[Fixed4RegimeKey] | None = None,
    router_spec: RegimePolicy | None = None,
) -> BranchTrainSelection:
    spec = _adapt_regime_policy(router_spec)
    resolved_holiday_keys = (
        tuple(holiday_keys)
        if holiday_keys is not None
        else tuple(cast(Sequence[Fixed4RegimeKey], spec.holiday_keys))
    )
    tr_local = np.asarray(idx_tr_by_key.get(tuple(regime), np.asarray([], dtype=int)), dtype=int)
    train_used = np.asarray(tr_local, dtype=int)
    train_source = "self"
    if tuple(regime) in {tuple(v) for v in resolved_holiday_keys} and int(train_used.size) < int(regime_min_branch_train):
        train_used = holiday_union_indices(
            idx_by_key=idx_tr_by_key,
            holiday_keys=resolved_holiday_keys,
            router_spec=spec,
        )
        train_source = "holiday_union"
    min_train_eff = int(regime_min_branch_train)
    if train_source == "holiday_union":
        relaxed = int(max(base_regime_min_branch_train, round(0.08 * float(total_train_size))))
        min_train_eff = int(min(regime_min_branch_train, relaxed))
    use_branch = bool(int(train_used.size) >= int(min_train_eff))
    return BranchTrainSelection(
        train_used=np.asarray(train_used, dtype=int),
        train_source=str(train_source),
        min_train_effective=int(min_train_eff),
        use_branch=bool(use_branch),
    )


__all__ = [
    "Fixed4RegimeKey",
    "RegimeCanonicalizer",
    "RegimePolicy",
    "RegimeResolution",
    "DefaultFixed4RegimeCanonicalizer",
    "STRICT4_GATE_NAMES",
    "STRICT4_REGIME_ORDER",
    "STRICT4_HOLIDAY_KEYS",
    "Strict4RouterSpec",
    "Strict4Resolution",
    "resolve_regime",
    "BranchTrainSelection",
    "resolve_strict4",
    "normalize_regime_key",
    "normalize_fixed4_key",
    "map_to_regime",
    "map_to_strict4_regime",
    "regime_keys_from_X",
    "strict4_keys_from_X",
    "build_regime_index",
    "holiday_union_indices",
    "resolve_branch_train_selection",
]
