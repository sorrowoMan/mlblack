from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from mlblack.backends.contracts import BackendContract
from mlblack.representations.codecs.neural.specs import NeuralGraphSpec

from .capabilities import (
    NumpyArtifactsCapability,
    NumpyLossesCapability,
    NumpyNeuralLoweringCapability,
    NumpyTensorCapability,
)


class NumpyNeuralBackend:
    """CPU numpy compute backend assembled from catalogable capabilities."""

    name = "numpy"

    def __init__(self) -> None:
        self.tensor = NumpyTensorCapability()
        self.lowering = NumpyNeuralLoweringCapability()
        self.losses = NumpyLossesCapability()
        self.artifacts = NumpyArtifactsCapability()
        self.capabilities = (
            self.tensor,
            self.lowering,
            self.losses,
            self.artifacts,
        )

    def contract(self) -> BackendContract:
        return BackendContract(
            name=self.name,
            capabilities=tuple(item.contract for item in self.capabilities),
            metadata={"family": "neural", "engine": "numpy", "device": "cpu"},
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


__all__ = ["NumpyNeuralBackend"]
