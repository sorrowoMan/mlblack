"""Canonical feature-pipeline entry for ETF temporal forecasting."""

from __future__ import annotations

from typing import Any, Mapping

from .etf_feature_construction import EtfFeatureBuilder, FeatureBuildSpec


def build_pipeline(
    spec: FeatureBuildSpec | None = None,
    *,
    resource_context: Mapping[str, Any] | None = None,
    component_overrides: Mapping[str, Any] | None = None,
):
    """Build the feature-construction component consumed by this Case."""

    del resource_context
    overrides = dict(component_overrides or {})
    return overrides.get("feature_builder") or EtfFeatureBuilder(spec)


__all__ = ["EtfFeatureBuilder", "FeatureBuildSpec", "build_pipeline"]
