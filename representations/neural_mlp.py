from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from mlblack.core.contracts import ComponentContract
from mlblack.core.head import OutputHead
from mlblack.core.representation import ModelRepresentation
from mlblack.core.types import UnknownState
from mlblack.representations.heads import PointHead
from mlblack.models import NumpyMLPPointModel
from mlblack.representations.codecs import (
    NeuralBackboneSpec,
    NeuralBatchingSpec,
    NeuralOptimizationSpec,
    NeuralEngineSpec,
    NumpyMLPCodec,
    NumpyMLPCodecConfig,
)


@dataclass(frozen=True)
class NumpyMLPPointConfig:
    input_dim: int
    hidden_layers: tuple[int, ...] = (64, 32)
    output_dim: int = 1
    activation: str = "relu"
    dropout: float = 0.0
    optimizer: str = "adamw"
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    batch_size: int | None = 64
    device: str = "cpu"
    checkpoint_enabled: bool = True
    resume_enabled: bool = True
    init_scale: float = 0.02
    random_seed: int = 42


class NumpyMLPPointRepresentation(ModelRepresentation):
    """Unknown vector -> numpy MLP model with pluggable output head."""

    name = "numpy_mlp_point"
    context_requires = ('candidate.unknown_state',)
    context_optional = ()
    context_provides = ('candidate.output', 'model.predict')
    context_mutates = ('candidate.repaired_state',)
    context_cache = ()
    requires_metrics = ()
    metrics_fallback = "strict"
    context_notes = 'Reads context fields: candidate.unknown_state; provides candidate.output, model.predict; mutates candidate.repaired_state.'
    contract = ComponentContract(
        name=name,
        requires=("candidate.unknown_state",),
        provides=("candidate.output", "model.predict"),
        mutates=("candidate.repaired_state",),
        supports_gradient=False,
        supports_batch=True,
        supports_resume=True,
        metadata={"family": "neural", "route": "numpy_mlp", "head": "point"},
    )

    def __init__(self, config: NumpyMLPPointConfig, head: OutputHead | None = None) -> None:
        self.config = config
        self.codec = NumpyMLPCodec(
            NumpyMLPCodecConfig(
                input_dim=int(config.input_dim),
                output_dim=int(config.output_dim),
                backbone=NeuralBackboneSpec(
                    hidden_layers=tuple(config.hidden_layers),
                    activation=str(config.activation),
                    dropout=float(config.dropout),
                ),
                optimization=NeuralOptimizationSpec(
                    optimizer=str(config.optimizer),
                    learning_rate=float(config.learning_rate),
                    weight_decay=float(config.weight_decay),
                ),
                batching=NeuralBatchingSpec(batch_size=config.batch_size),
                engine=NeuralEngineSpec(
                    device=str(config.device),
                    checkpoint_enabled=bool(config.checkpoint_enabled),
                    resume_enabled=bool(config.resume_enabled),
                ),
                init_scale=float(config.init_scale),
                random_seed=int(config.random_seed),
                representation_name=self.name,
            )
        )
        self.head = head or PointHead()
        self.shapes = self.codec.shapes
        self.base_dimension = int(self.codec.base_dimension)
        self.dimension = int(self.head.parameter_size(self.base_dimension))

    @classmethod
    def from_data(
        cls,
        X: np.ndarray,
        *,
        hidden_layers: Sequence[int] = (64, 32),
        activation: str = "relu",
        config: NumpyMLPPointConfig | None = None,
        head: OutputHead | None = None,
    ) -> "NumpyMLPPointRepresentation":
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim != 2:
            raise ValueError("X must be 2D")
        cfg = config or NumpyMLPPointConfig(
            input_dim=int(X_arr.shape[1]),
            hidden_layers=tuple(int(v) for v in hidden_layers),
            activation=activation,
        )
        return cls(cfg, head=head)

    def init(self, context: Mapping[str, Any]) -> UnknownState:
        _ = context
        values = np.zeros(self.dimension, dtype=float)
        base_values = self.codec.init_values()
        values[: base_values.shape[0]] = base_values
        return UnknownState(values=np.asarray(values, dtype=float), metadata={"source": "numpy_mlp_point_init"})

    def decode(self, state: UnknownState, context: Mapping[str, Any] | None = None) -> Any:
        ctx = dict(context or {})
        arr = np.asarray(state.values, dtype=float).reshape(-1)
        if arr.shape[0] != self.dimension:
            raise ValueError(f"state dimension {arr.shape[0]} does not match representation dimension {self.dimension}")
        return self.head.decode(
            arr,
            base_dimension=self.base_dimension,
            base_decode=self._decode_base,
            context=ctx,
        )

    def _decode_base(self, values: np.ndarray, context: Mapping[str, Any] | None = None) -> NumpyMLPPointModel:
        ctx = dict(context or {})
        return self.codec.decode(values, ctx)

    def repair(self, state: UnknownState, context: Mapping[str, Any] | None = None) -> UnknownState:
        _ = context
        arr = self.head.repair_values(state.values, base_dimension=self.base_dimension)
        return state.with_values(arr)

    def get_contract(self) -> ComponentContract:
        return self.contract.merged(self.head.get_contract(), name=self.name)

    def describe(self) -> Mapping[str, Any]:
        return {
            "name": self.name,
            "dimension": int(self.dimension),
            "base_dimension": int(self.base_dimension),
            "codec": self.codec.describe(),
            "head": self.head.describe(base_dimension=self.base_dimension),
        }



