"""Canonical data-pipeline entry for this migrated compatibility Case."""

from __future__ import annotations

from typing import Any, Mapping

from .example_pipeline import build_data_view


def build_pipeline(
    X=None,
    y=None,
    *,
    feature_names=(),
    target_name: str = "target",
    resource_context: Mapping[str, Any] | None = None,
    component_overrides: Mapping[str, Any] | None = None,
):
    """Build a NumericDataView when this legacy runner is given explicit data."""

    del resource_context
    overrides = dict(component_overrides or {})
    features = overrides.get("features", X)
    target = overrides.get("target", y)
    if features is None or target is None:
        raise ValueError("features and target are required to build this compatibility pipeline")
    return build_data_view(
        features,
        target,
        feature_names=overrides.get("feature_names", feature_names),
        target_name=str(overrides.get("target_name", target_name)),
    )


__all__ = ["build_data_view", "build_pipeline"]

