"""Canonical outer-search representation pipeline."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from nsgablack.core.base import BlackBoxProblem
from nsgablack.representation import RepresentationPipeline
from nsgablack.representation.continuous import (
    ClipRepair,
    ContextGaussianMutation,
    UniformInitializer,
)


def _bounds_arrays(problem: BlackBoxProblem) -> tuple[np.ndarray, np.ndarray]:
    bounds = getattr(problem, "bounds", None)
    dimension = int(getattr(problem, "dimension", 0))
    if isinstance(bounds, dict):
        keys = list(getattr(problem, "variables", []))
        if len(keys) != dimension or any(key not in bounds for key in keys):
            keys = list(bounds.keys())
        pairs = [bounds[key] for key in keys]
    else:
        pairs = list(bounds or [])
    if len(pairs) != dimension:
        return np.full(dimension, -1.0), np.full(dimension, 1.0)
    low = np.asarray([float(pair[0]) for pair in pairs], dtype=float)
    high = np.asarray([float(pair[1]) for pair in pairs], dtype=float)
    return np.minimum(low, high), np.maximum(low, high)


def build_pipeline(
    problem: BlackBoxProblem,
    *,
    resource_context: Mapping[str, Any] | None = None,
    component_overrides: Mapping[str, Any] | None = None,
) -> RepresentationPipeline:
    """Build the candidate lifecycle used by the outer optimizer."""

    del resource_context
    options = dict(component_overrides or {})
    low, high = _bounds_arrays(problem)
    return RepresentationPipeline(
        initializer=UniformInitializer(low=low, high=high),
        mutator=ContextGaussianMutation(
            base_sigma=float(options.get("base_sigma", 0.18)),
            sigma_key=str(options.get("sigma_key", "mutation_sigma")),
            low=low,
            high=high,
        ),
        repair=ClipRepair(low=low, high=high),
    )


def run_pipeline_slot(*args, **kwargs) -> RepresentationPipeline:
    return build_pipeline(*args, **kwargs)


__all__ = ["build_pipeline", "run_pipeline_slot"]
