"""Canonical representation-pipeline entry for t-SNE."""

from __future__ import annotations

from typing import Any, Mapping

from .representation.tsne_representation import TSNERepresentation


def build_pipeline(
    n_samples: int,
    *,
    output_dim: int = 2,
    resource_context: Mapping[str, Any] | None = None,
    component_overrides: Mapping[str, Any] | None = None,
):
    """Build the low-dimensional embedding representation."""

    del resource_context
    overrides = dict(component_overrides or {})
    return overrides.get("representation") or TSNERepresentation(
        n_samples=n_samples,
        output_dim=output_dim,
    )


__all__ = ["TSNERepresentation", "build_pipeline"]
