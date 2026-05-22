from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from mlblack.core.contracts import ComponentContract
from mlblack.core.representation import ModelRepresentation
from mlblack.core.types import UnknownState
from mlblack.representations.codecs import NeuralGraphCodec, NeuralGraphSpec, ParameterLayout


@dataclass(frozen=True)
class NeuralGraphRepresentationConfig:
    graph_spec: NeuralGraphSpec | Mapping[str, Any]
    init_scale: float = 0.02
    random_seed: int = 42
    representation_name: str = "neural_graph"


class NeuralGraphRepresentation(ModelRepresentation):
    """Unknown vector -> configured neural graph model.

    This is the generic neural decoder surface. The graph spec defines the
    forward structure, while the unknown state stores trainable parameter
    values for that structure.
    """

    name = "neural_graph"
    backend_requires = ("parameters.layout", "parameters.init", "neural.lowering")
    context_requires = ("candidate.unknown_state", "neural.graph_spec")
    context_optional = ()
    context_provides = ("candidate.model", "model.logits", "model.hidden_states", "neural.parameter_layout", "backend.contract")
    context_mutates = ("candidate.repaired_state",)
    context_cache = ()
    requires_metrics = ()
    metrics_fallback = "strict"
    context_notes = "Reads a flat unknown state and neural graph spec; decodes a neural model with logits/hidden-state outputs."
    contract = ComponentContract(
        name=name,
        requires=("candidate.unknown_state", "neural.graph_spec"),
        provides=("candidate.model", "model.logits", "model.hidden_states", "neural.parameter_layout", "backend.contract"),
        mutates=("candidate.repaired_state",),
        supports_gradient=True,
        supports_batch=True,
        supports_resume=True,
        metadata={"family": "neural", "route": "neural_graph"},
    )

    def __init__(self, config: NeuralGraphRepresentationConfig | NeuralGraphSpec | Mapping[str, Any]) -> None:
        if isinstance(config, NeuralGraphRepresentationConfig):
            cfg = config
        else:
            cfg = NeuralGraphRepresentationConfig(graph_spec=config)
        self.config = cfg
        self.codec = NeuralGraphCodec(
            cfg.graph_spec,
            init_scale=float(cfg.init_scale),
            random_seed=int(cfg.random_seed),
            representation_name=str(cfg.representation_name),
        )
        self.graph_spec = self.codec.spec
        self.layout: ParameterLayout | None = None
        self.dimension = 0
        self.base_dimension = 0

    @classmethod
    def tiny_transformer(
        cls,
        *,
        vocab_size: int,
        max_length: int,
        hidden_dim: int = 32,
        num_layers: int = 1,
        num_heads: int = 4,
        ffn_expansion_ratio: float = 2.0,
        attention_kind: str = "causal_self_attention",
        ffn_kind: str = "mlp",
        activation: str = "gelu",
        norm: str = "layer_norm",
        norm_position: str = "pre",
        position_encoding: str = "learned",
        lora: Mapping[str, Any] | None = None,
        qlora: Mapping[str, Any] | None = None,
        heads: tuple[Mapping[str, Any] | str, ...] | None = None,
        random_seed: int = 42,
        representation_name: str = "tiny_transformer",
    ) -> "NeuralGraphRepresentation":
        spec = NeuralGraphSpec.tiny_transformer(
            vocab_size=int(vocab_size),
            max_length=int(max_length),
            hidden_dim=int(hidden_dim),
            num_layers=int(num_layers),
            num_heads=int(num_heads),
            ffn_expansion_ratio=float(ffn_expansion_ratio),
            attention_kind=str(attention_kind),
            ffn_kind=str(ffn_kind),
            activation=str(activation),
            norm=str(norm),
            norm_position=str(norm_position),
            position_encoding=str(position_encoding),
            lora=lora,
            qlora=qlora,
            heads=heads,
            name=representation_name,
        )
        return cls(
            NeuralGraphRepresentationConfig(
                graph_spec=spec,
                random_seed=int(random_seed),
                representation_name=representation_name,
            )
        )

    @classmethod
    def tiny_cnn(
        cls,
        *,
        channels: int,
        height: int,
        width: int,
        conv_channels: tuple[int, ...] = (8, 16),
        kernel_size: int = 3,
        activation: str = "relu",
        dropout: float = 0.0,
        heads: tuple[Mapping[str, Any] | str, ...] | None = None,
        random_seed: int = 42,
        representation_name: str = "tiny_cnn",
    ) -> "NeuralGraphRepresentation":
        spec = NeuralGraphSpec.tiny_cnn(
            channels=int(channels),
            height=int(height),
            width=int(width),
            conv_channels=tuple(int(v) for v in conv_channels),
            kernel_size=int(kernel_size),
            activation=str(activation),
            dropout=float(dropout),
            heads=heads,
            name=representation_name,
        )
        return cls(
            NeuralGraphRepresentationConfig(
                graph_spec=spec,
                random_seed=int(random_seed),
                representation_name=representation_name,
            )
        )

    @classmethod
    def tiny_gnn(
        cls,
        *,
        node_feature_dim: int,
        num_nodes: int,
        hidden_dim: int = 16,
        num_layers: int = 2,
        activation: str = "relu",
        dropout: float = 0.0,
        pooling: str = "mean",
        heads: tuple[Mapping[str, Any] | str, ...] | None = None,
        random_seed: int = 42,
        representation_name: str = "tiny_gnn",
    ) -> "NeuralGraphRepresentation":
        spec = NeuralGraphSpec.tiny_gnn(
            node_feature_dim=int(node_feature_dim),
            num_nodes=int(num_nodes),
            hidden_dim=int(hidden_dim),
            num_layers=int(num_layers),
            activation=str(activation),
            dropout=float(dropout),
            pooling=str(pooling),
            heads=heads,
            name=representation_name,
        )
        return cls(
            NeuralGraphRepresentationConfig(
                graph_spec=spec,
                random_seed=int(random_seed),
                representation_name=representation_name,
            )
        )

    def setup(self, trainer: Any, context: Mapping[str, Any]) -> None:
        _ = trainer
        self._ensure_layout(context)

    def init(self, context: Mapping[str, Any]) -> UnknownState:
        self._ensure_layout(context)
        return UnknownState(
            values=self.codec.init_values(context),
            metadata={"source": "neural_graph_init", "graph_name": self.graph_spec.name},
        )

    def decode(self, state: UnknownState, context: Mapping[str, Any] | None = None) -> Any:
        layout = self._ensure_layout(context)
        arr = np.asarray(state.values, dtype=float).reshape(-1)
        if arr.shape[0] != layout.total_size:
            raise ValueError(f"state dimension {arr.shape[0]} does not match neural graph dimension {layout.total_size}")
        return self.codec.decode(arr, context=context)

    def repair(self, state: UnknownState, context: Mapping[str, Any] | None = None) -> UnknownState:
        layout = self._ensure_layout(context)
        arr = np.asarray(state.values, dtype=float).reshape(-1)
        if arr.shape[0] != layout.total_size:
            raise ValueError(f"state dimension {arr.shape[0]} does not match neural graph dimension {layout.total_size}")
        if not np.all(np.isfinite(arr)):
            arr = np.nan_to_num(arr, nan=0.0, posinf=1e3, neginf=-1e3)
        return state.with_values(arr, repaired=True)

    def describe(self) -> Mapping[str, Any]:
        layout = self.layout
        return {
            "name": self.name,
            "dimension": int(self.dimension),
            "base_dimension": int(self.base_dimension),
            "codec": self.codec.describe(),
            "graph_spec": self.graph_spec.as_dict(),
            "parameter_layout": None if layout is None else layout.as_dict(),
        }

    def get_contract(self) -> ComponentContract:
        return self.contract

    def _ensure_layout(self, context: Mapping[str, Any] | None = None) -> ParameterLayout:
        layout = self.codec.parameter_layout(context)
        self.layout = layout
        self.dimension = int(layout.total_size)
        self.base_dimension = int(layout.total_size)
        return layout


__all__ = ["NeuralGraphRepresentation", "NeuralGraphRepresentationConfig"]
