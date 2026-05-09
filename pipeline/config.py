from __future__ import annotations

from dataclasses import dataclass, field

from .feature_space import (
    CandidatePoolConfig,
    FeatureEngineeringConfig,
    RegimeFeaturePackConfig,
    TemporalFeaturePackConfig,
)


@dataclass(frozen=True)
class FeatureSpaceBuilderConfig:
    feature_engineering: FeatureEngineeringConfig = field(default_factory=FeatureEngineeringConfig)
    temporal_pack: TemporalFeaturePackConfig = field(default_factory=TemporalFeaturePackConfig)
    regime_pack: RegimeFeaturePackConfig = field(default_factory=RegimeFeaturePackConfig)
    candidate_pool: CandidatePoolConfig = field(default_factory=CandidatePoolConfig)
    build_full_candidate_pool: bool = False


__all__ = ["FeatureSpaceBuilderConfig"]
