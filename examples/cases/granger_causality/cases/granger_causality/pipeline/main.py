"""Canonical representation-pipeline entry for Granger causality."""

from __future__ import annotations

from typing import Any, Mapping

from .representation.granger_representation import GrangerRepresentation


def build_pipeline(
    n_vars: int,
    *,
    init_scale: float = 0.05,
    resource_context: Mapping[str, Any] | None = None,
    component_overrides: Mapping[str, Any] | None = None,
):
    """Build the VAR coefficient representation."""

    del resource_context
    overrides = dict(component_overrides or {})
    return overrides.get("representation") or GrangerRepresentation(
        n_vars,
        init_scale=init_scale,
        name="granger_rep",
    )


__all__ = ["GrangerRepresentation", "build_pipeline"]
