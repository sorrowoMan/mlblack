from __future__ import annotations

from typing import Any

import numpy as np

from mlblack.backends.contracts import BackendCapabilityContract
from mlblack.models import NumpyMLPPointModel, mlp_parameter_shapes, split_mlp_parameters
from mlblack.representations.codecs.neural.specs import NeuralBlockSpec, NeuralGraphSpec, NeuralHeadSpec


class NumpyNeuralLoweringCapability:
    contract = BackendCapabilityContract(
        backend="numpy",
        capability="neural_lowering",
        provides=(
            "neural.lowering",
            "neural.lowering.mlp",
            "parameters.layout",
            "parameters.init",
            "parameters.flat_import",
        ),
        methods={
            "neural.lowering": "decode_neural_graph(values, spec, random_seed) -> NumpyMLPPointModel",
            "parameters.layout": "parameter_layout(spec) -> (shapes, names)",
            "parameters.init": "initial_values(spec, random_seed) -> np.ndarray",
        },
        tensor_kinds=("np.ndarray",),
        model_kinds=("NumpyMLPPointModel",),
        routes=("mlp",),
        supports_stateful_module=False,
        notes="Lowers backend-agnostic MLP NeuralGraphSpec into a pure numpy point model.",
    )

    def route(self, spec: NeuralGraphSpec) -> str:
        return "mlp" if _is_mlp_spec(spec) else str(spec.metadata.get("route", "unknown"))

    def supports_spec(self, spec: NeuralGraphSpec) -> bool:
        return self.route(spec) == "mlp"

    def parameter_layout(self, spec: NeuralGraphSpec) -> tuple[tuple[tuple[int, ...], ...], tuple[str, ...]]:
        if self.route(spec) != "mlp":
            raise ValueError(f"numpy backend cannot build neural parameter layout for route={self.route(spec)!r}")
        input_dim, hidden_layers, output_dim, _activation = _mlp_parts(spec)
        shapes = mlp_parameter_shapes(input_dim, hidden_layers, output_dim)
        names: list[str] = []
        for idx in range(len(shapes) // 2):
            names.append(f"mlp.layers.{idx}.weight")
            names.append(f"mlp.layers.{idx}.bias")
        return tuple(shapes), tuple(names)

    def initial_values(self, spec: NeuralGraphSpec, *, random_seed: int = 42) -> np.ndarray:
        shapes, _names = self.parameter_layout(spec)
        total = int(sum(np.prod(shape) for shape in shapes))
        scale = float(dict(spec.parameterization).get("init_scale", 0.02))
        return np.random.default_rng(int(random_seed)).normal(loc=0.0, scale=scale, size=total)

    def decode_neural_graph(self, values: np.ndarray, spec: NeuralGraphSpec, *, random_seed: int = 42) -> NumpyMLPPointModel:
        _ = random_seed
        input_dim, hidden_layers, output_dim, activation = _mlp_parts(spec)
        weights, biases = split_mlp_parameters(
            np.asarray(values, dtype=float),
            input_dim=input_dim,
            hidden_layers=hidden_layers,
            output_dim=output_dim,
        )
        return NumpyMLPPointModel(
            weights=weights,
            biases=biases,
            activation=activation,
            metadata={"backend": "numpy", "route": "mlp", "graph_name": spec.name},
        )


def _is_mlp_spec(spec: NeuralGraphSpec) -> bool:
    blocks = spec.block_specs()
    return len(blocks) == 1 and str(blocks[0].kind).lower() in {"mlp", "mlp_block", "feed_forward"}


def _mlp_parts(spec: NeuralGraphSpec) -> tuple[int, tuple[int, ...], int, str]:
    input_cfg = dict(spec.input)
    input_dim = int(input_cfg.get("input_dim", input_cfg.get("dimension", 0)))
    if input_dim <= 0:
        raise ValueError("MLP NeuralGraphSpec requires input.input_dim")
    blocks = spec.block_specs()
    if len(blocks) != 1:
        raise ValueError("MLP NeuralGraphSpec expects exactly one block")
    block: NeuralBlockSpec = blocks[0]
    params = dict(block.params)
    hidden_layers = tuple(int(v) for v in params.get("hidden_layers", (64, 32)))
    activation = str(params.get("activation", "relu"))
    heads = spec.head_specs()
    if heads:
        head: NeuralHeadSpec = heads[0]
        output_dim = int(dict(head.params).get("output_dim", 1))
    else:
        output_dim = 1
    return input_dim, hidden_layers, output_dim, activation


__all__ = ["NumpyNeuralLoweringCapability"]
