from __future__ import annotations

from typing import Sequence

from core.symbolic.feature_space.regime_router import (
    RegimePolicy,
    RegimeResolution,
    Strict4RouterSpec,
    resolve_regime,
)

from .config import BranchPolicyConfig


BranchPolicyResolution = RegimeResolution


def build_branch_regime_policy(cfg: BranchPolicyConfig) -> RegimePolicy:
    if cfg.regime_policy is not None:
        return cfg.regime_policy
    default_spec = Strict4RouterSpec()
    return Strict4RouterSpec(
        gate_names=default_spec.gate_names if cfg.gate_names is None else tuple(str(v) for v in cfg.gate_names),
        regime_order=(
            default_spec.regime_order
            if cfg.regime_order is None
            else tuple(tuple(int(x) for x in row) for row in cfg.regime_order)
        ),
        holiday_keys=(
            default_spec.holiday_keys
            if cfg.holiday_keys is None
            else tuple(tuple(int(x) for x in row) for row in cfg.holiday_keys)
        ),
    )


def build_branch_router_spec(cfg: BranchPolicyConfig) -> RegimePolicy:
    return build_branch_regime_policy(cfg)


def resolve_branch_policy(
    feature_names: Sequence[str],
    *,
    cfg: BranchPolicyConfig,
) -> BranchPolicyResolution:
    return resolve_regime(
        feature_names,
        enabled=bool(cfg.enabled),
        regime_policy=build_branch_regime_policy(cfg),
    )


def resolve_branch_workers(
    cfg: BranchPolicyConfig,
    *,
    reinvest_enabled: bool = False,
    reinvest_mult: float = 1.0,
) -> int:
    workers = int(max(1, cfg.parallel_workers))
    if reinvest_enabled and bool(cfg.enabled):
        workers = int(max(workers, round(workers * float(max(1.0, reinvest_mult)))))
    return workers


__all__ = [
    "BranchPolicyConfig",
    "BranchPolicyResolution",
    "build_branch_regime_policy",
    "build_branch_router_spec",
    "resolve_branch_policy",
    "resolve_branch_workers",
]
