from __future__ import annotations

from typing import Any

import numpy as np

from mlblack.backends.contracts import BackendCapabilityContract
from mlblack.representations.codecs.neural.specs import NeuralGraphSpec

from ..graph import decode_tiny_gnn, gnn_initial_values, gnn_parameter_layout, is_tiny_gnn_spec
from ..transformer import (
    decode_tiny_transformer,
    is_tiny_transformer_spec,
    transformer_initial_values,
    transformer_parameter_layout,
)
from ..vision import cnn_initial_values, cnn_parameter_layout, decode_tiny_cnn, is_tiny_cnn_spec


class TorchNeuralLoweringCapability:
    contract = BackendCapabilityContract(
        backend="torch",
        capability="neural_lowering",
        provides=(
            "neural.lowering",
            "neural.lowering.transformer",
            "neural.lowering.cnn",
            "neural.lowering.gnn",
            "parameters.layout",
            "parameters.init",
            "parameters.flat_import",
        ),
        methods={
            "neural.lowering": "decode_neural_graph(values, spec, random_seed) -> torch.nn.Module",
            "parameters.layout": "parameter_layout(spec) -> (shapes, names)",
            "parameters.init": "initial_values(spec, random_seed) -> np.ndarray",
        },
        model_kinds=("torch.nn.Module",),
        routes=("tiny_transformer", "tiny_cnn", "tiny_gnn"),
        supports_stateful_module=True,
        notes="Lowers backend-agnostic NeuralGraphSpec routes into tiny torch modules.",
    )

    def route(self, spec: NeuralGraphSpec) -> str:
        if is_tiny_transformer_spec(spec):
            return "tiny_transformer"
        if is_tiny_cnn_spec(spec):
            return "tiny_cnn"
        if is_tiny_gnn_spec(spec):
            return "tiny_gnn"
        return str(spec.metadata.get("route", "unknown"))

    def supports_spec(self, spec: NeuralGraphSpec) -> bool:
        return self.route(spec) in {"tiny_transformer", "tiny_cnn", "tiny_gnn"}

    def parameter_layout(self, spec: NeuralGraphSpec) -> tuple[tuple[tuple[int, ...], ...], tuple[str, ...]]:
        route = self.route(spec)
        if route == "tiny_transformer":
            return transformer_parameter_layout(spec)
        if route == "tiny_cnn":
            return cnn_parameter_layout(spec)
        if route == "tiny_gnn":
            return gnn_parameter_layout(spec)
        raise ValueError(f"torch backend cannot build neural parameter layout for route={route!r}")

    def initial_values(self, spec: NeuralGraphSpec, *, random_seed: int = 42) -> np.ndarray:
        route = self.route(spec)
        if route == "tiny_transformer":
            return transformer_initial_values(spec, random_seed=random_seed)
        if route == "tiny_cnn":
            return cnn_initial_values(spec, random_seed=random_seed)
        if route == "tiny_gnn":
            return gnn_initial_values(spec, random_seed=random_seed)
        raise ValueError(f"torch backend cannot initialize neural route={route!r}")

    def decode_neural_graph(self, values: np.ndarray, spec: NeuralGraphSpec, *, random_seed: int = 42) -> Any:
        route = self.route(spec)
        if route == "tiny_transformer":
            return decode_tiny_transformer(values, spec, random_seed=random_seed)
        if route == "tiny_cnn":
            return decode_tiny_cnn(values, spec, random_seed=random_seed)
        if route == "tiny_gnn":
            return decode_tiny_gnn(values, spec, random_seed=random_seed)
        raise ValueError(f"torch backend cannot decode neural route={route!r}")


__all__ = ["TorchNeuralLoweringCapability"]
