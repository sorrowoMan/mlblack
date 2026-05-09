from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from core.symbolic.feature_space.activation_config import DynamicActivationConfig, resolve_dynamic_activation_kwargs
from core.symbolic.feature_space.candidate_pool import CandidateTerm

from .config import DynamicPoolPolicyConfig


def build_dynamic_activation_policy(cfg: DynamicPoolPolicyConfig) -> dict[str, Any]:
    return resolve_dynamic_activation_kwargs(
        DynamicActivationConfig(
            unary_top_k=int(max(1, cfg.unary_top_k)),
            pair_top_k=int(max(1, cfg.pair_top_k)),
            gate_top_k=int(max(1, cfg.gate_top_k)),
            recursive_depth=int(max(1, cfg.recursive_depth)),
            recursive_seed_top_k=int(max(1, cfg.recursive_seed_top_k)),
            recursive_pair_seed_top_k=int(max(1, cfg.recursive_pair_seed_top_k)),
            recursive_max_complexity=float(max(3.0, cfg.recursive_max_complexity)),
            allow_trig=bool(cfg.allow_trig),
            allow_safe_exp=bool(cfg.allow_safe_exp),
            allow_safe_log=bool(cfg.allow_safe_log),
            allow_safe_ratio=bool(cfg.allow_safe_ratio),
            family_budget_csv=str(cfg.family_budget_csv),
        )
    )


def build_epoch_generations(total_generations: int, *, cfg: DynamicPoolPolicyConfig) -> list[int]:
    generations = int(max(1, total_generations))
    if not bool(cfg.enabled):
        return [generations]
    epochs = int(max(1, cfg.epochs))
    if epochs <= 1:
        return [generations]
    base = int(generations // epochs)
    rem = int(generations - base * epochs)
    return [int(max(1, base + (1 if i < rem else 0))) for i in range(epochs)]


def should_expand_dynamic_pool(
    cfg: DynamicPoolPolicyConfig,
    *,
    epoch_idx: int,
    epoch_generations: Sequence[int],
    has_active_subset: bool,
) -> bool:
    return bool(cfg.enabled) and bool(has_active_subset) and int(epoch_idx) < int(len(epoch_generations) - 1)


def collect_selected_expr_keys(
    top_cache_rows: Sequence[Mapping[str, Any]],
    candidates: Sequence[CandidateTerm],
    *,
    top_cache_use: int,
) -> set[str]:
    selected_keys: set[str] = set()
    limit = int(max(1, top_cache_use))
    for row in top_cache_rows[:limit]:
        for idx in [int(v) for v in row.get("subset_idx", [])]:
            if 0 <= idx < len(candidates):
                selected_keys.add(json.dumps(candidates[idx].expr, sort_keys=True))
    return selected_keys


__all__ = [
    "DynamicPoolPolicyConfig",
    "build_dynamic_activation_policy",
    "build_epoch_generations",
    "should_expand_dynamic_pool",
    "collect_selected_expr_keys",
]
