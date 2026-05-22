from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class NumpyMLPPointModel:
    """Small decoded MLP point model backed by numpy arrays.

    This is the pure-python decoder target. Torch/sklearn routes can use
    EstimatorSpecModel instead when training is delegated to an external
    backend.
    """

    weights: tuple[np.ndarray, ...]
    biases: tuple[np.ndarray, ...]
    activation: str = "relu"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def predict(self, X: np.ndarray) -> np.ndarray:
        out = np.asarray(X, dtype=float)
        if out.ndim != 2:
            raise ValueError("X must be 2D")
        for idx, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            out = out @ np.asarray(weight, dtype=float) + np.asarray(bias, dtype=float)
            if idx < len(self.weights) - 1:
                out = _activate(out, self.activation)
        return np.asarray(out, dtype=float).reshape(out.shape[0], -1)[:, 0]

    def parameter_shapes(self) -> tuple[tuple[int, ...], ...]:
        shapes: list[tuple[int, ...]] = []
        for weight, bias in zip(self.weights, self.biases):
            shapes.append(tuple(np.asarray(weight).shape))
            shapes.append(tuple(np.asarray(bias).shape))
        return tuple(shapes)


def mlp_parameter_shapes(input_dim: int, hidden_layers: Sequence[int], output_dim: int = 1) -> tuple[tuple[int, ...], ...]:
    dims = (int(input_dim), *(int(v) for v in hidden_layers), int(output_dim))
    shapes: list[tuple[int, ...]] = []
    for left, right in zip(dims[:-1], dims[1:]):
        shapes.append((left, right))
        shapes.append((right,))
    return tuple(shapes)


def split_mlp_parameters(
    values: np.ndarray,
    *,
    input_dim: int,
    hidden_layers: Sequence[int],
    output_dim: int = 1,
) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
    arr = np.asarray(values, dtype=float).reshape(-1)
    shapes = mlp_parameter_shapes(input_dim, hidden_layers, output_dim)
    expected = int(sum(np.prod(shape) for shape in shapes))
    if arr.shape[0] != expected:
        raise ValueError(f"parameter vector has {arr.shape[0]} values but MLP expects {expected}")
    offset = 0
    weights: list[np.ndarray] = []
    biases: list[np.ndarray] = []
    for idx, shape in enumerate(shapes):
        size = int(np.prod(shape))
        block = arr[offset : offset + size].reshape(shape)
        offset += size
        if idx % 2 == 0:
            weights.append(block)
        else:
            biases.append(block)
    return tuple(weights), tuple(biases)


def _activate(x: np.ndarray, activation: str) -> np.ndarray:
    key = str(activation or "relu").strip().lower()
    if key == "relu":
        return np.maximum(x, 0.0)
    if key == "tanh":
        return np.tanh(x)
    if key in {"sigmoid", "logistic"}:
        return 1.0 / (1.0 + np.exp(-x))
    if key in {"identity", "linear", "none"}:
        return x
    raise ValueError(f"unsupported activation: {activation}")
