from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from mlblack.backends.contracts import BackendCapabilityContract
from mlblack.models import mlp_parameter_shapes
from mlblack.representations.codecs.neural.specs import NeuralBlockSpec, NeuralGraphSpec, NeuralHeadSpec


@dataclass(frozen=True)
class TensorFlowMLPPointModel:
    """Functional TensorFlow MLP model decoded from a flat parameter vector."""

    values: np.ndarray
    input_dim: int
    hidden_layers: tuple[int, ...]
    output_dim: int = 1
    activation: str = "relu"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def predict(self, X: np.ndarray) -> np.ndarray:
        output = self.predict_from_values(self.values, X)
        return np.asarray(output.numpy(), dtype=float).reshape(np.asarray(X).shape[0], -1)[:, 0]

    def predict_from_values(self, values: Any, X: Any) -> Any:
        tf = _tf()
        return _forward_flat(
            tf.convert_to_tensor(values, dtype=tf.float32),
            tf.convert_to_tensor(X, dtype=tf.float32),
            input_dim=int(self.input_dim),
            hidden_layers=tuple(int(v) for v in self.hidden_layers),
            output_dim=int(self.output_dim),
            activation=str(self.activation),
        )

    def parameter_shapes(self) -> tuple[tuple[int, ...], ...]:
        return mlp_parameter_shapes(int(self.input_dim), tuple(int(v) for v in self.hidden_layers), int(self.output_dim))


class TensorFlowNeuralLoweringCapability:
    contract = BackendCapabilityContract(
        backend="tensorflow",
        capability="neural_lowering",
        provides=(
            "neural.lowering",
            "neural.lowering.mlp",
            "parameters.layout",
            "parameters.init",
            "parameters.flat_import",
        ),
        methods={
            "neural.lowering": "decode_neural_graph(values, spec, random_seed) -> TensorFlowMLPPointModel",
            "parameters.layout": "parameter_layout(spec) -> (shapes, names)",
            "parameters.init": "initial_values(spec, random_seed) -> np.ndarray",
        },
        tensor_kinds=("tf.Tensor",),
        model_kinds=("TensorFlowMLPPointModel",),
        routes=("mlp",),
        supports_functional_params=True,
        notes="Lowers MLP NeuralGraphSpec into a TensorFlow functional model using predict_from_values(...).",
    )

    def route(self, spec: NeuralGraphSpec) -> str:
        return "mlp" if _is_mlp_spec(spec) else str(spec.metadata.get("route", "unknown"))

    def supports_spec(self, spec: NeuralGraphSpec) -> bool:
        return self.route(spec) == "mlp"

    def parameter_layout(self, spec: NeuralGraphSpec) -> tuple[tuple[tuple[int, ...], ...], tuple[str, ...]]:
        if self.route(spec) != "mlp":
            raise ValueError(f"tensorflow backend cannot build neural parameter layout for route={self.route(spec)!r}")
        input_dim, hidden_layers, output_dim, _activation = _mlp_parts(spec)
        shapes = mlp_parameter_shapes(input_dim, hidden_layers, output_dim)
        names: list[str] = []
        for idx in range(len(shapes) // 2):
            names.append(f"mlp.layers.{idx}.weight")
            names.append(f"mlp.layers.{idx}.bias")
        return tuple(shapes), tuple(names)

    def initial_values(self, spec: NeuralGraphSpec, *, random_seed: int = 42) -> np.ndarray:
        _ = _tf()
        shapes, _names = self.parameter_layout(spec)
        total = int(sum(np.prod(shape) for shape in shapes))
        scale = float(dict(spec.parameterization).get("init_scale", 0.02))
        return np.random.default_rng(int(random_seed)).normal(loc=0.0, scale=scale, size=total)

    def decode_neural_graph(self, values: np.ndarray, spec: NeuralGraphSpec, *, random_seed: int = 42) -> TensorFlowMLPPointModel:
        _ = random_seed
        input_dim, hidden_layers, output_dim, activation = _mlp_parts(spec)
        return TensorFlowMLPPointModel(
            values=np.asarray(values, dtype=float).reshape(-1),
            input_dim=int(input_dim),
            hidden_layers=tuple(int(v) for v in hidden_layers),
            output_dim=int(output_dim),
            activation=str(activation),
            metadata={"backend": "tensorflow", "route": "mlp", "graph_name": spec.name},
        )


def _forward_flat(
    values: Any,
    X: Any,
    *,
    input_dim: int,
    hidden_layers: tuple[int, ...],
    output_dim: int,
    activation: str,
) -> Any:
    tf = _tf()
    weights, biases = _split_flat_tf(
        values,
        input_dim=int(input_dim),
        hidden_layers=tuple(int(v) for v in hidden_layers),
        output_dim=int(output_dim),
    )
    out = X
    last = len(weights) - 1
    for idx, (weight, bias) in enumerate(zip(weights, biases)):
        out = tf.linalg.matmul(out, tf.cast(weight, out.dtype)) + tf.cast(bias, out.dtype)
        if idx < last:
            out = _activate(out, activation)
    return out


def _split_flat_tf(
    values: Any,
    *,
    input_dim: int,
    hidden_layers: tuple[int, ...],
    output_dim: int,
) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    tf = _tf()
    arr = tf.reshape(values, (-1,))
    shapes = mlp_parameter_shapes(input_dim, hidden_layers, output_dim)
    expected = int(sum(np.prod(shape) for shape in shapes))
    if int(arr.shape[0]) != expected:
        raise ValueError(f"parameter vector has {int(arr.shape[0])} values but MLP expects {expected}")
    offset = 0
    weights: list[Any] = []
    biases: list[Any] = []
    for idx, shape in enumerate(shapes):
        size = int(np.prod(shape))
        block = tf.reshape(arr[offset : offset + size], tuple(int(v) for v in shape))
        offset += size
        if idx % 2 == 0:
            weights.append(block)
        else:
            biases.append(block)
    return tuple(weights), tuple(biases)


def _activate(x: Any, activation: str) -> Any:
    tf = _tf()
    key = str(activation or "relu").strip().lower()
    if key == "relu":
        return tf.nn.relu(x)
    if key == "tanh":
        return tf.math.tanh(x)
    if key in {"sigmoid", "logistic"}:
        return tf.math.sigmoid(x)
    if key in {"identity", "linear", "none"}:
        return x
    raise ValueError(f"unsupported activation: {activation}")


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


def _tf() -> Any:
    try:
        import tensorflow as tf
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("tensorflow backend requires optional dependency 'tensorflow'") from exc
    return tf


__all__ = ["TensorFlowMLPPointModel", "TensorFlowNeuralLoweringCapability"]
