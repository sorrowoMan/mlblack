"""Canonical pipeline entry for matrix factorization."""

from __future__ import annotations

from typing import Any, Mapping

from .mf_pipeline import build_rating_data_view, generate_synthetic_ratings
from .representation.mf_representation import MFRepresentation


def build_pipeline(
    n_users: int,
    n_items: int,
    *,
    k: int = 5,
    init_scale: float = 0.05,
    nmf: bool = False,
    resource_context: Mapping[str, Any] | None = None,
    component_overrides: Mapping[str, Any] | None = None,
):
    """Build the factor-matrix representation used by the Trainer."""

    del resource_context
    overrides = dict(component_overrides or {})
    return overrides.get("representation") or MFRepresentation(
        n_users,
        n_items,
        k=k,
        init_scale=init_scale,
        nmf=nmf,
        name="mf_rep",
    )


__all__ = ["MFRepresentation", "build_pipeline", "build_rating_data_view", "generate_synthetic_ratings"]
