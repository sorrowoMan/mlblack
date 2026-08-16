"""Canonical pipeline entry for the blackbase substrate demo."""

from __future__ import annotations

from typing import Any, Mapping

from .representation import SimpleRepresentation


def build_pipeline(
    n_features: int,
    *,
    seed: int = 42,
    resource_context: Mapping[str, Any] | None = None,
    component_overrides: Mapping[str, Any] | None = None,
):
    """Build the representation component used by this ML Case."""

    del resource_context
    overrides = dict(component_overrides or {})
    return overrides.get("representation") or SimpleRepresentation(n_features, seed=seed)


__all__ = ["SimpleRepresentation", "build_pipeline"]
