from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from core.symbolic.feature_space.activation_config import DynamicActivationConfig, resolve_dynamic_activation_kwargs
from core.symbolic.feature_space.candidate_pool import CandidateTerm, _build_candidate_pool
from core.symbolic.feature_space.feature_bundle import FeatureBundle


@dataclass(frozen=True)
class CandidatePoolConfig:
    dynamic_pool_enabled: bool = True
    dynamic_init_minimal: bool = True
    safe_log1p_abs: bool = True
    safe_exp_clip: bool = True
    safe_reciprocal: bool = True
    safe_exp_clip_k: float = 8.0
    safe_reciprocal_eps: float = 1e-3
    dynamic_activation: DynamicActivationConfig = field(default_factory=DynamicActivationConfig)
    conditional_config: Any | None = None


def build_candidate_pool(feature_bundle: FeatureBundle, cfg: CandidatePoolConfig) -> list[CandidateTerm]:
    return _build_candidate_pool(
        feature_bundle.X_train,
        feature_bundle.y_train,
        feature_names=feature_bundle.feature_names,
        topk_for_pairs=6,
        include_pair_interactions=bool(not cfg.dynamic_init_minimal),
        include_gradient_enrich=bool(not cfg.dynamic_init_minimal),
        include_safe_log1p_abs=bool(cfg.safe_log1p_abs),
        include_safe_exp_clip=bool(cfg.safe_exp_clip),
        include_safe_reciprocal=bool(cfg.safe_reciprocal),
        safe_exp_clip_k=float(max(1.0, cfg.safe_exp_clip_k)),
        safe_reciprocal_eps=float(max(1e-8, cfg.safe_reciprocal_eps)),
        activation_config=resolve_dynamic_activation_kwargs(cfg.dynamic_activation),
        conditional_config=cfg.conditional_config,
    )


def build_full_candidate_pool(feature_bundle: FeatureBundle, cfg: CandidatePoolConfig) -> list[CandidateTerm]:
    return _build_candidate_pool(
        feature_bundle.X_train,
        feature_bundle.y_train,
        feature_names=feature_bundle.feature_names,
        topk_for_pairs=6,
        include_pair_interactions=True,
        include_gradient_enrich=True,
        include_safe_log1p_abs=bool(cfg.safe_log1p_abs),
        include_safe_exp_clip=bool(cfg.safe_exp_clip),
        include_safe_reciprocal=bool(cfg.safe_reciprocal),
        safe_exp_clip_k=float(max(1.0, cfg.safe_exp_clip_k)),
        safe_reciprocal_eps=float(max(1e-8, cfg.safe_reciprocal_eps)),
        activation_config=resolve_dynamic_activation_kwargs(cfg.dynamic_activation),
        conditional_config=cfg.conditional_config,
    )


__all__ = [
    "CandidatePoolConfig",
    "build_candidate_pool",
    "build_full_candidate_pool",
]
