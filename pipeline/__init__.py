from __future__ import annotations

from typing import Any, Dict

from .base import BasePipeline
from .config import FeatureSpaceBuilderConfig
from .feature_space_builder import FeatureSpaceBuildInput, FeatureSpaceBuildResult, build_feature_space
from .identity import IdentityPipeline
from .zscore import ZScorePipeline

_PIPELINE_REGISTRY = {
    "identity": IdentityPipeline,
    "zscore": ZScorePipeline,
}


def create_pipeline(name: str, state: Dict[str, Any] | None = None) -> BasePipeline:
    key = str(name or "identity").strip().lower()
    cls = _PIPELINE_REGISTRY.get(key)
    if cls is None:
        raise KeyError(f"Unknown pipeline: {name}")
    pipe = cls()
    if state is not None:
        pipe.load_state_dict(dict(state))
    return pipe


__all__ = [
    "BasePipeline",
    "FeatureSpaceBuilderConfig",
    "FeatureSpaceBuildInput",
    "FeatureSpaceBuildResult",
    "IdentityPipeline",
    "ZScorePipeline",
    "build_feature_space",
    "create_pipeline",
]
