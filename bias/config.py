from __future__ import annotations

from dataclasses import dataclass

from core.symbolic.feature_space.activation_config import DynamicActivationConfig
from core.symbolic.feature_space.regime_router import RegimePolicy


@dataclass(frozen=True)
class BranchPolicyConfig:
    enabled: bool = False
    min_branch_train: int = 64
    parallel_workers: int = 4
    regime_policy: RegimePolicy | None = None
    gate_names: tuple[str, str, str, str] | None = None
    regime_order: tuple[tuple[int, int, int, int], ...] | None = None
    holiday_keys: tuple[tuple[int, int, int, int], ...] | None = None


@dataclass(frozen=True)
class ObjectivePolicyConfig:
    coverage_error_threshold: float = 0.03


@dataclass(frozen=True)
class DynamicPoolPolicyConfig:
    enabled: bool = True
    epochs: int = 5
    init_minimal: bool = True
    expand_max_new: int = 64
    focus_top_features: int = 8
    partner_topk: int = 6
    top_cache_use: int = 32
    max_pool_size: int = 640
    unary_top_k: int = 6
    pair_top_k: int = 8
    gate_top_k: int = 6
    recursive_depth: int = 2
    recursive_seed_top_k: int = 3
    recursive_pair_seed_top_k: int = 2
    recursive_max_complexity: float = 9.5
    allow_trig: bool = True
    allow_safe_exp: bool = True
    allow_safe_log: bool = True
    allow_safe_ratio: bool = True
    family_budget_csv: str = DynamicActivationConfig().family_budget_csv


__all__ = [
    "BranchPolicyConfig",
    "ObjectivePolicyConfig",
    "DynamicPoolPolicyConfig",
]
