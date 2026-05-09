from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .config import FeatureSpaceBuilderConfig
from .feature_space import (
    CandidateTerm,
    FeatureBundle,
    RegimeFeaturePackResult,
    TemporalFeaturePackResult,
    apply_regime_feature_pack,
    apply_temporal_feature_pack,
    build_candidate_pool,
    build_feature_bundle,
    build_full_candidate_pool,
)


@dataclass(frozen=True)
class FeatureSpaceBuildInput:
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_names: Sequence[str]


@dataclass(frozen=True)
class FeatureSpaceBuildResult:
    feature_bundle: FeatureBundle
    temporal_pack_result: TemporalFeaturePackResult
    regime_pack_result: RegimeFeaturePackResult
    candidates: tuple[CandidateTerm, ...]

    @property
    def X_train(self) -> np.ndarray:
        return self.feature_bundle.X_train

    @property
    def y_train(self) -> np.ndarray:
        return self.feature_bundle.y_train

    @property
    def X_test(self) -> np.ndarray:
        return self.feature_bundle.X_test

    @property
    def y_test(self) -> np.ndarray:
        return self.feature_bundle.y_test

    @property
    def feature_names(self) -> tuple[str, ...]:
        return self.feature_bundle.feature_names


def build_feature_space(
    *,
    inputs: FeatureSpaceBuildInput,
    cfg: FeatureSpaceBuilderConfig,
) -> FeatureSpaceBuildResult:
    base_bundle = build_feature_bundle(
        X_train=np.asarray(inputs.X_train, dtype=float),
        y_train=np.asarray(inputs.y_train, dtype=float),
        X_test=np.asarray(inputs.X_test, dtype=float),
        y_test=np.asarray(inputs.y_test, dtype=float),
        feature_names=tuple(str(v) for v in inputs.feature_names),
        cfg=cfg.feature_engineering,
    )

    temporal_pack = apply_temporal_feature_pack(
        X_train=base_bundle.X_train,
        X_test=base_bundle.X_test,
        feature_names=base_bundle.feature_names,
        config=cfg.temporal_pack,
    )
    regime_pack = apply_regime_feature_pack(
        X_train=temporal_pack.X_train,
        X_test=temporal_pack.X_test,
        feature_names=temporal_pack.feature_names,
        config=cfg.regime_pack,
    )

    final_bundle = FeatureBundle(
        X_train=np.asarray(regime_pack.X_train, dtype=float),
        y_train=np.asarray(base_bundle.y_train, dtype=float),
        X_test=np.asarray(regime_pack.X_test, dtype=float),
        y_test=np.asarray(base_bundle.y_test, dtype=float),
        feature_names=tuple(str(v) for v in regime_pack.feature_names),
        n_features_raw=int(base_bundle.n_features_raw),
        feature_names_raw=tuple(str(v) for v in base_bundle.feature_names_raw),
        lag_added_features=tuple(str(v) for v in base_bundle.lag_added_features),
        lag_cross_added_features=tuple(str(v) for v in base_bundle.lag_cross_added_features),
        dropped_features=tuple(str(v) for v in base_bundle.dropped_features),
    )

    if bool(cfg.build_full_candidate_pool):
        candidates = tuple(build_full_candidate_pool(final_bundle, cfg.candidate_pool))
    else:
        candidates = tuple(build_candidate_pool(final_bundle, cfg.candidate_pool))

    return FeatureSpaceBuildResult(
        feature_bundle=final_bundle,
        temporal_pack_result=temporal_pack,
        regime_pack_result=regime_pack,
        candidates=tuple(candidates),
    )


__all__ = [
    "FeatureSpaceBuildInput",
    "FeatureSpaceBuildResult",
    "build_feature_space",
]
