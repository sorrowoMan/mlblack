"""Canonical representation-pipeline entry for the L0 resource demo."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from mlblack.core.representation import ModelRepresentation
from mlblack.core.types import UnknownState


class L0ResourceDemoRepresentation(ModelRepresentation):
    """Small bounded-vector representation used by the resource audit Case."""

    name = "l0_resource_demo_representation"

    def __init__(self, dimension: int = 3, bounds: tuple = (-5, 5)):
        self.dimension = int(dimension)
        self.bounds = tuple(bounds)

    def init(self, context: Mapping[str, Any]) -> UnknownState:
        del context
        low, high = self.bounds
        return UnknownState(values=np.random.uniform(low, high, self.dimension))

    def decode(self, state: UnknownState, context: Mapping[str, Any] | None = None) -> Any:
        del context
        return state.values

    def mutate(self, state: UnknownState, context: Mapping[str, Any] | None = None) -> UnknownState:
        del context
        mutated = state.values + np.random.normal(0, 0.5, state.values.shape)
        mutated = np.clip(mutated, self.bounds[0], self.bounds[1])
        return state.with_values(mutated)


def build_pipeline(
    *,
    dimension: int = 3,
    bounds: tuple = (-5, 5),
    resource_context: Mapping[str, Any] | None = None,
    component_overrides: Mapping[str, Any] | None = None,
):
    """Build the representation while honoring standard component overrides."""

    del resource_context
    overrides = dict(component_overrides or {})
    return overrides.get("representation") or L0ResourceDemoRepresentation(
        dimension=dimension,
        bounds=bounds,
    )


__all__ = ["L0ResourceDemoRepresentation", "build_pipeline"]
