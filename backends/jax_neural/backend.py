from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from mlblack.backends.contracts import BackendContract
from mlblack.representations.codecs.neural.specs import NeuralGraphSpec

from .capabilities import (
    JaxArtifactsCapability,
    JaxAutogradCapability,
    JaxLossesCapability,
    JaxNeuralLoweringCapability,
    JaxOptimizersCapability,
    JaxTensorCapability,
)


class JaxNeuralBackend:
    """JAX compute backend assembled from catalogable capability components."""

    name = "jax"

    def __init__(self) -> None:
        self.tensor = JaxTensorCapability()
        self.lowering = JaxNeuralLoweringCapability()
        self.losses = JaxLossesCapability()
        self.autograd = JaxAutogradCapability()
        self.optimizers = JaxOptimizersCapability()
        self.artifacts = JaxArtifactsCapability()
        self.capabilities = (
            self.tensor,
            self.lowering,
            self.losses,
            self.autograd,
            self.optimizers,
            self.artifacts,
        )

    def contract(self) -> BackendContract:
        return BackendContract(
            name=self.name,
            capabilities=tuple(item.contract for item in self.capabilities),
            metadata={"family": "neural", "engine": "jax", "parameter_style": "functional"},
        )

    def supports(self, requirement: str) -> bool:
        return self.contract().supports(str(requirement))

    def route(self, spec: NeuralGraphSpec) -> str:
        return self.lowering.route(spec)

    def parameter_layout(self, spec: NeuralGraphSpec) -> tuple[tuple[tuple[int, ...], ...], tuple[str, ...]]:
        return self.lowering.parameter_layout(spec)

    def initial_values(self, spec: NeuralGraphSpec, *, random_seed: int = 42) -> np.ndarray:
        return self.lowering.initial_values(spec, random_seed=random_seed)

    def decode_neural_graph(
        self,
        values: np.ndarray,
        spec: NeuralGraphSpec,
        *,
        random_seed: int = 42,
        context: Mapping[str, Any] | None = None,
    ) -> Any:
        _ = context
        return self.lowering.decode_neural_graph(values, spec, random_seed=random_seed)


__all__ = ["JaxNeuralBackend"]
