from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from mlblack.backends.contracts import BackendContract
from mlblack.representations.codecs.neural.specs import NeuralGraphSpec

from .capabilities import (
    TorchArtifactsCapability,
    TorchAutogradCapability,
    TorchLossesCapability,
    TorchNeuralLoweringCapability,
    TorchOptimizersCapability,
    TorchTensorCapability,
)


class TorchNeuralBackend:
    """Torch compute backend assembled from catalogable capability components."""

    name = "torch"

    def __init__(self) -> None:
        self.tensor = TorchTensorCapability()
        self.lowering = TorchNeuralLoweringCapability()
        self.autograd = TorchAutogradCapability(self.tensor)
        self.optimizers = TorchOptimizersCapability(self.tensor, self.autograd)
        self.losses = TorchLossesCapability(self.tensor)
        self.artifacts = TorchArtifactsCapability(self.tensor, self.autograd)
        self.capabilities = (
            self.tensor,
            self.lowering,
            self.autograd,
            self.optimizers,
            self.losses,
            self.artifacts,
        )

    def contract(self) -> BackendContract:
        return BackendContract(
            name=self.name,
            capabilities=tuple(item.contract for item in self.capabilities),
            metadata={"family": "neural", "engine": "torch"},
        )

    def supports(self, requirement: str) -> bool:
        return self.contract().supports(str(requirement))

    def route(self, spec: NeuralGraphSpec) -> str:
        return self.lowering.route(spec)

    def parameter_layout(self, spec: NeuralGraphSpec) -> tuple[tuple[tuple[int, ...], ...], tuple[str, ...]]:
        return self.lowering.parameter_layout(spec)

    def initial_values(self, spec: NeuralGraphSpec, *, random_seed: int = 42) -> np.ndarray:
        return self.lowering.initial_values(spec, random_seed=random_seed)

    def decode_neural_graph(self, values: np.ndarray, spec: NeuralGraphSpec, *, random_seed: int = 42, context: Mapping[str, Any] | None = None) -> Any:
        _ = context
        return self.lowering.decode_neural_graph(values, spec, random_seed=random_seed)


__all__ = ["TorchNeuralBackend"]
