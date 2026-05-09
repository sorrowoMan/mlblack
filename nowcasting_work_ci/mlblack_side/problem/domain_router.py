from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from conditional.router import RouterPolicyAdapter, adapt_router_policy
from bias import BranchPolicyConfig
from core.symbolic.feature_space.regime_router import (
    DefaultFixed4RegimeCanonicalizer,
    Fixed4RegimeKey,
    RegimeCanonicalizer,
    RegimePolicy,
    STRICT4_HOLIDAY_KEYS,
    STRICT4_REGIME_ORDER,
)


WORK_CI_STRICT4_GATE_NAMES: tuple[str, str, str, str] = (
    "is_holiday_day_or_window",
    "is_holiday_near",
    "is_holiday_mid",
    "is_nonwork_weekend",
)


@dataclass(frozen=True)
class TrafficHolidayRegimeCanonicalizer:
    _base: DefaultFixed4RegimeCanonicalizer = field(default_factory=DefaultFixed4RegimeCanonicalizer)

    def normalize_key(self, raw_key: Sequence[int]) -> Fixed4RegimeKey:
        return self._base.normalize_key(raw_key)

    def canonicalize(
        self,
        raw_key: Sequence[int],
        *,
        policy: RegimePolicy,
    ) -> Fixed4RegimeKey:
        key = self.normalize_key(raw_key)
        if key in policy.regime_order:
            return key
        if key[0] > 0 and key[1] > 0:
            return (1, 1, 0, 0)
        if key[0] > 0 and key[2] > 0:
            return (1, 0, 1, 0)
        if key[3] > 0:
            return (0, 0, 0, 1)
        zero = self.normalize_key((0, 0, 0, 0))
        if zero in policy.regime_order:
            return zero
        if policy.regime_order:
            return tuple(policy.regime_order[-1])
        return zero


@dataclass(frozen=True)
class TrafficHolidayRegimePolicy:
    gate_names: tuple[str, str, str, str] = WORK_CI_STRICT4_GATE_NAMES
    regime_order: tuple[Fixed4RegimeKey, ...] = STRICT4_REGIME_ORDER
    holiday_keys: tuple[Fixed4RegimeKey, ...] = STRICT4_HOLIDAY_KEYS
    canonicalizer: RegimeCanonicalizer = field(default_factory=TrafficHolidayRegimeCanonicalizer)


WORK_CI_STRICT4_POLICY = TrafficHolidayRegimePolicy()

# Backward-compatible alias for older internal code paths.
WORK_CI_STRICT4_ROUTER_SPEC = WORK_CI_STRICT4_POLICY


def build_work_ci_conditional_router_policy() -> RouterPolicyAdapter:
    """Expose the traffic holiday policy through the generic conditional/router contract."""
    return adapt_router_policy(WORK_CI_STRICT4_POLICY)


def default_work_ci_branch_policy() -> BranchPolicyConfig:
    return BranchPolicyConfig(
        enabled=False,
        min_branch_train=64,
        parallel_workers=4,
        regime_policy=WORK_CI_STRICT4_POLICY,
        gate_names=WORK_CI_STRICT4_POLICY.gate_names,
        regime_order=WORK_CI_STRICT4_POLICY.regime_order,
        holiday_keys=WORK_CI_STRICT4_POLICY.holiday_keys,
    )


def build_work_ci_branch_policy(
    *,
    enabled: bool,
    min_branch_train: int,
    parallel_workers: int,
) -> BranchPolicyConfig:
    base = default_work_ci_branch_policy()
    return BranchPolicyConfig(
        enabled=bool(enabled),
        min_branch_train=int(max(8, min_branch_train)),
        parallel_workers=int(max(1, parallel_workers)),
        regime_policy=WORK_CI_STRICT4_POLICY,
        gate_names=base.gate_names,
        regime_order=base.regime_order,
        holiday_keys=base.holiday_keys,
    )


__all__ = [
    "WORK_CI_STRICT4_GATE_NAMES",
    "WORK_CI_STRICT4_POLICY",
    "WORK_CI_STRICT4_ROUTER_SPEC",
    "TrafficHolidayRegimeCanonicalizer",
    "TrafficHolidayRegimePolicy",
    "build_work_ci_conditional_router_policy",
    "default_work_ci_branch_policy",
    "build_work_ci_branch_policy",
]
