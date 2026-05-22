from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from mlblack.models import NumpyMLPPointModel, mlp_parameter_shapes, split_mlp_parameters


@dataclass(frozen=True)
class NeuralBackboneSpec:
    hidden_layers: tuple[int, ...] = (64, 32)
    activation: str = "relu"
    dropout: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "hidden_layers": tuple(int(v) for v in self.hidden_layers),
            "activation": str(self.activation),
            "dropout": float(self.dropout),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class NeuralOptimizationSpec:
    optimizer: str = "adamw"
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    max_steps: int = 120
    early_stopping: bool = True
    early_stop_patience: int = 20
    early_stop_min_delta: float = 1e-6
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "optimizer": str(self.optimizer),
            "learning_rate": float(self.learning_rate),
            "weight_decay": float(self.weight_decay),
            "max_steps": int(self.max_steps),
            "early_stopping": bool(self.early_stopping),
            "early_stop_patience": int(self.early_stop_patience),
            "early_stop_min_delta": float(self.early_stop_min_delta),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class NeuralBatchingSpec:
    batch_size: int | None = 64
    shuffle: bool = True
    drop_last: bool = False
    num_workers: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "batch_size": self.batch_size,
            "shuffle": bool(self.shuffle),
            "drop_last": bool(self.drop_last),
            "num_workers": int(self.num_workers),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class NeuralEngineSpec:
    engine: str = "torch"
    device: str = "cpu"
    checkpoint_enabled: bool = True
    resume_enabled: bool = True
    trainer_state_enabled: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "engine": str(self.engine),
            "device": str(self.device),
            "checkpoint_enabled": bool(self.checkpoint_enabled),
            "resume_enabled": bool(self.resume_enabled),
            "trainer_state_enabled": bool(self.trainer_state_enabled),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class NumpyMLPCodecConfig:
    input_dim: int
    output_dim: int = 1
    backbone: NeuralBackboneSpec = field(default_factory=NeuralBackboneSpec)
    optimization: NeuralOptimizationSpec = field(default_factory=NeuralOptimizationSpec)
    batching: NeuralBatchingSpec = field(default_factory=NeuralBatchingSpec)
    engine: NeuralEngineSpec = field(default_factory=NeuralEngineSpec)
    init_scale: float = 0.02
    random_seed: int = 42
    representation_name: str = "numpy_mlp_point"


class NumpyMLPCodec:
    """Numpy MLP parameter codec with explicit mechanism specs."""

    def __init__(self, config: NumpyMLPCodecConfig) -> None:
        self.config = config
        self.shapes = mlp_parameter_shapes(config.input_dim, config.backbone.hidden_layers, config.output_dim)
        self.base_dimension = int(sum(np.prod(shape) for shape in self.shapes))
        self._rng = np.random.default_rng(int(config.random_seed))

    @classmethod
    def from_data(
        cls,
        X: np.ndarray,
        *,
        hidden_layers: Sequence[int] = (64, 32),
        activation: str = "relu",
        config: NumpyMLPCodecConfig | None = None,
    ) -> "NumpyMLPCodec":
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim != 2:
            raise ValueError("X must be 2D")
        cfg = config or NumpyMLPCodecConfig(
            input_dim=int(X_arr.shape[1]),
            backbone=NeuralBackboneSpec(hidden_layers=tuple(int(v) for v in hidden_layers), activation=activation),
        )
        return cls(cfg)

    def init_values(self) -> np.ndarray:
        return self._rng.normal(loc=0.0, scale=float(self.config.init_scale), size=self.base_dimension)

    def decode(self, values: np.ndarray, context: Mapping[str, Any] | None = None) -> NumpyMLPPointModel:
        ctx = dict(context or {})
        weights, biases = split_mlp_parameters(
            np.asarray(values, dtype=float),
            input_dim=int(self.config.input_dim),
            hidden_layers=tuple(self.config.backbone.hidden_layers),
            output_dim=int(self.config.output_dim),
        )
        return NumpyMLPPointModel(
            weights=weights,
            biases=biases,
            activation=str(self.config.backbone.activation),
            metadata={"representation": self.config.representation_name, "head_block": ctx.get("head.block")},
        )

    def describe(self) -> dict[str, Any]:
        return {
            "codec": "numpy_mlp",
            "base_dimension": int(self.base_dimension),
            "input_dim": int(self.config.input_dim),
            "output_dim": int(self.config.output_dim),
            "parameter_shapes": tuple(self.shapes),
            "backbone": self.config.backbone.as_dict(),
            "optimization": self.config.optimization.as_dict(),
            "batching": self.config.batching.as_dict(),
            "engine": self.config.engine.as_dict(),
        }


