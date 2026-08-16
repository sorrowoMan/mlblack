"""Canonical pipeline entry for symbolic orthogonal nested search."""

from .representation import build_representation_pipeline


def build_pipeline(*args, **kwargs):
    """Build the outer-search representation pipeline."""

    return build_representation_pipeline(*args, **kwargs)


__all__ = ["build_pipeline", "build_representation_pipeline"]
