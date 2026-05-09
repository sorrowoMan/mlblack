from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DynamicActivationConfig:
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
    family_budget_csv: str = (
        "poly:12,"
        "bounded:10,"
        "saturation:10,"
        "radial:10,"
        "trig:8,"
        "safe_log:8,"
        "safe_exp:12,"
        "safe_ratio:8,"
        "interaction_basic:16,"
        "interaction_poly:12,"
        "interaction_compose:12,"
        "interaction_ratio:12,"
        "interaction_saturation:12,"
        "interaction_radial:10,"
        "interaction_rational:12,"
        "gate_interaction:12"
    )


def parse_family_budget_csv(text: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for raw in str(text).split(","):
        item = str(raw).strip()
        if not item or ":" not in item:
            continue
        key, value = item.split(":", 1)
        family = str(key).strip()
        if not family:
            continue
        try:
            budget = int(value.strip())
        except Exception:
            continue
        out[family] = int(max(0, budget))
    return out


def resolve_dynamic_activation_kwargs(cfg: DynamicActivationConfig) -> dict[str, Any]:
    return {
        "unary_top_k": int(max(1, cfg.unary_top_k)),
        "pair_top_k": int(max(1, cfg.pair_top_k)),
        "gate_top_k": int(max(1, cfg.gate_top_k)),
        "recursive_depth": int(max(1, cfg.recursive_depth)),
        "recursive_seed_top_k": int(max(1, cfg.recursive_seed_top_k)),
        "recursive_pair_seed_top_k": int(max(1, cfg.recursive_pair_seed_top_k)),
        "recursive_max_complexity": float(max(3.0, cfg.recursive_max_complexity)),
        "allow_trig": bool(cfg.allow_trig),
        "allow_safe_exp": bool(cfg.allow_safe_exp),
        "allow_safe_log": bool(cfg.allow_safe_log),
        "allow_safe_ratio": bool(cfg.allow_safe_ratio),
        "family_budget": parse_family_budget_csv(str(cfg.family_budget_csv)),
    }


__all__ = [
    "DynamicActivationConfig",
    "parse_family_budget_csv",
    "resolve_dynamic_activation_kwargs",
]
