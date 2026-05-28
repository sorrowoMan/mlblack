from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class NeuralComponentSpec:
    """Small declarative unit inside a neural graph.

    This describes structure, not trained parameter values.
    """

    kind: str
    name: str = ""
    params: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_value(cls, value: str | Mapping[str, Any] | "NeuralComponentSpec") -> "NeuralComponentSpec":
        if isinstance(value, NeuralComponentSpec):
            return value
        if isinstance(value, str):
            return cls(kind=value)
        payload = dict(value)
        return cls(
            kind=str(payload.get("kind", payload.get("type", ""))),
            name=str(payload.get("name", "")),
            params=dict(payload.get("params", payload.get("config", {})) or {}),
            metadata=dict(payload.get("metadata", {}) or {}),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "name": self.name,
            "params": dict(self.params),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class NeuralBlockSpec:
    """A repeatable block in a neural forward graph."""

    kind: str
    repeat: int = 1
    mechanisms: Mapping[str, NeuralComponentSpec | Mapping[str, Any] | str] = field(default_factory=dict)
    params: Mapping[str, Any] = field(default_factory=dict)
    residual: Mapping[str, Any] = field(default_factory=dict)
    norm: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_value(cls, value: Mapping[str, Any] | "NeuralBlockSpec") -> "NeuralBlockSpec":
        if isinstance(value, NeuralBlockSpec):
            return value
        payload = dict(value)
        mechanisms = {
            str(key): NeuralComponentSpec.from_value(component)
            for key, component in dict(payload.get("mechanisms", {}) or {}).items()
        }
        return cls(
            kind=str(payload.get("kind", payload.get("type", ""))),
            repeat=int(payload.get("repeat", 1)),
            mechanisms=mechanisms,
            params=dict(payload.get("params", payload.get("config", {})) or {}),
            residual=dict(payload.get("residual", {}) or {}),
            norm=dict(payload.get("norm", {}) or {}),
            metadata=dict(payload.get("metadata", {}) or {}),
        )

    def mechanism_specs(self) -> dict[str, NeuralComponentSpec]:
        return {str(key): NeuralComponentSpec.from_value(value) for key, value in dict(self.mechanisms).items()}

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "repeat": int(self.repeat),
            "mechanisms": {key: value.as_dict() for key, value in self.mechanism_specs().items()},
            "params": dict(self.params),
            "residual": dict(self.residual),
            "norm": dict(self.norm),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class NeuralHeadSpec:
    """Output interpretation attached to a neural graph."""

    kind: str
    name: str = ""
    params: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_value(cls, value: str | Mapping[str, Any] | "NeuralHeadSpec") -> "NeuralHeadSpec":
        if isinstance(value, NeuralHeadSpec):
            return value
        if isinstance(value, str):
            return cls(kind=value, name=value)
        payload = dict(value)
        return cls(
            kind=str(payload.get("kind", payload.get("type", ""))),
            name=str(payload.get("name", payload.get("kind", ""))),
            params=dict(payload.get("params", payload.get("config", {})) or {}),
            metadata=dict(payload.get("metadata", {}) or {}),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "name": self.name,
            "params": dict(self.params),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class NeuralGraphSpec:
    """Declarative neural graph structure.

    The spec is the structure surface. Parameter values live in a codec state,
    trainer state, or artifact.
    """

    name: str
    input: Mapping[str, Any]
    blocks: Sequence[NeuralBlockSpec | Mapping[str, Any]] = tuple()
    heads: Sequence[NeuralHeadSpec | Mapping[str, Any] | str] = tuple()
    parameterization: Mapping[str, Any] = field(default_factory=dict)
    audit: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_value(cls, value: Mapping[str, Any] | "NeuralGraphSpec") -> "NeuralGraphSpec":
        if isinstance(value, NeuralGraphSpec):
            return value
        payload = dict(value)
        return cls(
            name=str(payload.get("name", "neural_graph")),
            input=dict(payload.get("input", {}) or {}),
            blocks=tuple(NeuralBlockSpec.from_value(item) for item in payload.get("blocks", ())),
            heads=tuple(NeuralHeadSpec.from_value(item) for item in payload.get("heads", ())),
            parameterization=dict(payload.get("parameterization", {}) or {}),
            audit=dict(payload.get("audit", {}) or {}),
            metadata=dict(payload.get("metadata", {}) or {}),
        )

    @classmethod
    def mlp(
        cls,
        *,
        input_dim: int,
        hidden_layers: Sequence[int] = (64, 32),
        output_dim: int = 1,
        activation: str = "relu",
        dropout: float = 0.0,
        name: str = "mlp_graph",
    ) -> "NeuralGraphSpec":
        return cls(
            name=name,
            input={"kind": "vector", "input_dim": int(input_dim)},
            blocks=(
                NeuralBlockSpec(
                    kind="mlp",
                    params={
                        "hidden_layers": tuple(int(v) for v in hidden_layers),
                        "activation": str(activation),
                        "dropout": float(dropout),
                    },
                ),
            ),
            heads=(NeuralHeadSpec(kind="point", name="point", params={"output_dim": int(output_dim)}),),
            metadata={"family": "neural", "route": "mlp"},
        )

    @classmethod
    def tiny_transformer(
        cls,
        *,
        vocab_size: int,
        max_length: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        ffn_expansion_ratio: float = 4.0,
        attention_kind: str = "causal_self_attention",
        ffn_kind: str = "mlp",
        activation: str = "gelu",
        norm: str = "layer_norm",
        norm_position: str = "pre",
        position_encoding: str = "learned",
        residual: str = "plain",
        dropout: float = 0.0,
        lora: Mapping[str, Any] | None = None,
        qlora: Mapping[str, Any] | None = None,
        heads: Sequence[NeuralHeadSpec | Mapping[str, Any] | str] | None = None,
        name: str = "tiny_transformer_graph",
    ) -> "NeuralGraphSpec":
        head_specs: Sequence[NeuralHeadSpec | Mapping[str, Any] | str]
        head_specs = heads or ({"kind": "classification", "name": "classification", "params": {"num_classes": 2}},)
        return cls(
            name=name,
            input={
                "kind": "token_ids",
                "vocab_size": int(vocab_size),
                "max_length": int(max_length),
                "hidden_dim": int(hidden_dim),
                "position_encoding": str(position_encoding),
            },
            blocks=(
                NeuralBlockSpec(
                    kind="transformer_decoder_block",
                    repeat=int(num_layers),
                    mechanisms={
                        "attention": NeuralComponentSpec(
                            kind=str(attention_kind),
                            params={
                                "num_heads": int(num_heads),
                                "head_dim": int(hidden_dim) // max(1, int(num_heads)),
                                "dropout": float(dropout),
                                "position_encoding": str(position_encoding),
                            },
                        ),
                        "ffn": NeuralComponentSpec(
                            kind=str(ffn_kind),
                            params={
                                "expansion_ratio": float(ffn_expansion_ratio),
                                "activation": str(activation),
                                "dropout": float(dropout),
                            },
                        ),
                    },
                    residual={"kind": str(residual), "scale": 1.0},
                    norm={"kind": str(norm), "position": str(norm_position)},
                    metadata={"family": "transformer"},
                ),
            ),
            heads=tuple(NeuralHeadSpec.from_value(item) for item in head_specs),
            parameterization={"lora": dict(lora or {}), "qlora": dict(qlora or {})},
            metadata={"family": "neural", "route": "tiny_transformer"},
        )

    @classmethod
    def tiny_cnn(
        cls,
        *,
        channels: int,
        height: int,
        width: int,
        conv_channels: Sequence[int] = (8, 16),
        kernel_size: int = 3,
        activation: str = "relu",
        dropout: float = 0.0,
        heads: Sequence[NeuralHeadSpec | Mapping[str, Any] | str] | None = None,
        name: str = "tiny_cnn_graph",
    ) -> "NeuralGraphSpec":
        head_specs = heads or ({"kind": "classification", "name": "classification", "params": {"num_classes": 2}},)
        return cls(
            name=name,
            input={
                "kind": "image",
                "channels": int(channels),
                "height": int(height),
                "width": int(width),
            },
            blocks=(
                NeuralBlockSpec(
                    kind="cnn_block",
                    params={
                        "conv_channels": tuple(int(v) for v in conv_channels),
                        "kernel_size": int(kernel_size),
                        "activation": str(activation),
                        "dropout": float(dropout),
                    },
                    metadata={"family": "cnn"},
                ),
            ),
            heads=tuple(NeuralHeadSpec.from_value(item) for item in head_specs),
            metadata={"family": "neural", "route": "tiny_cnn"},
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
        heads: Sequence[NeuralHeadSpec | Mapping[str, Any] | str] | None = None,
        name: str = "tiny_gnn_graph",
    ) -> "NeuralGraphSpec":
        head_specs = heads or ({"kind": "classification", "name": "classification", "params": {"num_classes": 2}},)
        return cls(
            name=name,
            input={
                "kind": "graph",
                "node_feature_dim": int(node_feature_dim),
                "num_nodes": int(num_nodes),
            },
            blocks=(
                NeuralBlockSpec(
                    kind="gnn_block",
                    repeat=int(num_layers),
                    params={
                        "hidden_dim": int(hidden_dim),
                        "activation": str(activation),
                        "dropout": float(dropout),
                        "pooling": str(pooling),
                    },
                    metadata={"family": "gnn"},
                ),
            ),
            heads=tuple(NeuralHeadSpec.from_value(item) for item in head_specs),
            metadata={"family": "neural", "route": "tiny_gnn"},
        )

    @classmethod
    def temporal_lstm(
        cls,
        *,
        input_dim: int,
        sequence_length: int,
        hidden_dim: int = 32,
        num_layers: int = 1,
        output_dim: int = 1,
        dropout: float = 0.0,
        bidirectional: bool = False,
        heads: Sequence[NeuralHeadSpec | Mapping[str, Any] | str] | None = None,
        name: str = "temporal_lstm_graph",
    ) -> "NeuralGraphSpec":
        head_specs = heads or ({"kind": "forecast", "name": "forecast", "params": {"output_dim": int(output_dim)}},)
        return cls(
            name=name,
            input={
                "kind": "sequence",
                "input_dim": int(input_dim),
                "sequence_length": int(sequence_length),
            },
            blocks=(
                NeuralBlockSpec(
                    kind="lstm_block",
                    params={
                        "hidden_dim": int(hidden_dim),
                        "num_layers": int(num_layers),
                        "dropout": float(dropout),
                        "bidirectional": bool(bidirectional),
                    },
                    metadata={"family": "temporal", "route": "temporal_lstm"},
                ),
            ),
            heads=tuple(NeuralHeadSpec.from_value(item) for item in head_specs),
            metadata={"family": "neural", "route": "temporal_lstm"},
        )

    @classmethod
    def temporal_tcn(
        cls,
        *,
        input_dim: int,
        sequence_length: int,
        channels: Sequence[int] = (32, 32),
        kernel_size: int = 3,
        dilation_base: int = 2,
        output_dim: int = 1,
        activation: str = "relu",
        dropout: float = 0.0,
        heads: Sequence[NeuralHeadSpec | Mapping[str, Any] | str] | None = None,
        name: str = "temporal_tcn_graph",
    ) -> "NeuralGraphSpec":
        head_specs = heads or ({"kind": "forecast", "name": "forecast", "params": {"output_dim": int(output_dim)}},)
        return cls(
            name=name,
            input={
                "kind": "sequence",
                "input_dim": int(input_dim),
                "sequence_length": int(sequence_length),
            },
            blocks=(
                NeuralBlockSpec(
                    kind="tcn_block",
                    params={
                        "channels": tuple(int(v) for v in channels),
                        "kernel_size": int(kernel_size),
                        "dilation_base": int(dilation_base),
                        "activation": str(activation),
                        "dropout": float(dropout),
                    },
                    metadata={"family": "temporal", "route": "temporal_tcn"},
                ),
            ),
            heads=tuple(NeuralHeadSpec.from_value(item) for item in head_specs),
            metadata={"family": "neural", "route": "temporal_tcn"},
        )

    @classmethod
    def temporal_transformer(
        cls,
        *,
        input_dim: int,
        sequence_length: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        ffn_expansion_ratio: float = 4.0,
        output_dim: int = 1,
        activation: str = "gelu",
        dropout: float = 0.0,
        heads: Sequence[NeuralHeadSpec | Mapping[str, Any] | str] | None = None,
        name: str = "temporal_transformer_graph",
    ) -> "NeuralGraphSpec":
        head_specs = heads or ({"kind": "forecast", "name": "forecast", "params": {"output_dim": int(output_dim)}},)
        return cls(
            name=name,
            input={
                "kind": "sequence",
                "input_dim": int(input_dim),
                "sequence_length": int(sequence_length),
                "hidden_dim": int(hidden_dim),
            },
            blocks=(
                NeuralBlockSpec(
                    kind="temporal_transformer_encoder_block",
                    repeat=int(num_layers),
                    mechanisms={
                        "attention": NeuralComponentSpec(
                            kind="temporal_self_attention",
                            params={"num_heads": int(num_heads), "dropout": float(dropout)},
                        ),
                        "ffn": NeuralComponentSpec(
                            kind="mlp",
                            params={"expansion_ratio": float(ffn_expansion_ratio), "activation": str(activation), "dropout": float(dropout)},
                        ),
                    },
                    norm={"kind": "layer_norm", "position": "pre"},
                    metadata={"family": "temporal", "route": "temporal_transformer"},
                ),
            ),
            heads=tuple(NeuralHeadSpec.from_value(item) for item in head_specs),
            metadata={"family": "neural", "route": "temporal_transformer"},
        )

    @classmethod
    def temporal_nbeats(
        cls,
        *,
        input_dim: int,
        sequence_length: int,
        hidden_dim: int = 64,
        theta_dim: int = 8,
        num_stacks: int = 2,
        num_blocks: int = 3,
        output_dim: int = 1,
        share_weights: bool = False,
        dropout: float = 0.0,
        heads: Sequence[NeuralHeadSpec | Mapping[str, Any] | str] | None = None,
        name: str = "temporal_nbeats_graph",
    ) -> "NeuralGraphSpec":
        head_specs = heads or ({"kind": "forecast", "name": "forecast", "params": {"output_dim": int(output_dim)}},)
        return cls(
            name=name,
            input={
                "kind": "sequence",
                "input_dim": int(input_dim),
                "sequence_length": int(sequence_length),
            },
            blocks=(
                NeuralBlockSpec(
                    kind="nbeats_block",
                    repeat=int(num_stacks),
                    params={
                        "hidden_dim": int(hidden_dim),
                        "theta_dim": int(theta_dim),
                        "num_blocks": int(num_blocks),
                        "output_dim": int(output_dim),
                        "share_weights": bool(share_weights),
                        "dropout": float(dropout),
                    },
                    metadata={"family": "temporal", "route": "temporal_nbeats"},
                ),
            ),
            heads=tuple(NeuralHeadSpec.from_value(item) for item in head_specs),
            metadata={"family": "neural", "route": "temporal_nbeats"},
        )

    @classmethod
    def temporal_deepar(
        cls,
        *,
        input_dim: int,
        sequence_length: int,
        hidden_dim: int = 32,
        num_layers: int = 1,
        output_dim: int = 1,
        dropout: float = 0.0,
        bidirectional: bool = False,
        heads: Sequence[NeuralHeadSpec | Mapping[str, Any] | str] | None = None,
        name: str = "temporal_deepar_graph",
    ) -> "NeuralGraphSpec":
        head_specs = heads or ({"kind": "deepar", "name": "deepar", "params": {"output_dim": int(output_dim)}},)
        return cls(
            name=name,
            input={
                "kind": "sequence",
                "input_dim": int(input_dim),
                "sequence_length": int(sequence_length),
            },
            blocks=(
                NeuralBlockSpec(
                    kind="deepar_block",
                    params={
                        "hidden_dim": int(hidden_dim),
                        "num_layers": int(num_layers),
                        "dropout": float(dropout),
                        "bidirectional": bool(bidirectional),
                    },
                    metadata={"family": "temporal", "route": "temporal_deepar"},
                ),
            ),
            heads=tuple(NeuralHeadSpec.from_value(item) for item in head_specs),
            metadata={"family": "neural", "route": "temporal_deepar"},
        )

    @classmethod
    def temporal_patchtst(
        cls,
        *,
        input_dim: int,
        sequence_length: int,
        patch_len: int = 8,
        stride: int | None = None,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        ffn_dim: int | None = None,
        output_dim: int = 1,
        dropout: float = 0.0,
        heads: Sequence[NeuralHeadSpec | Mapping[str, Any] | str] | None = None,
        name: str = "temporal_patchtst_graph",
    ) -> "NeuralGraphSpec":
        head_specs = heads or ({"kind": "forecast", "name": "forecast", "params": {"output_dim": int(output_dim)}},)
        eff_ffn = int(ffn_dim or hidden_dim * 4)
        return cls(
            name=name,
            input={
                "kind": "sequence",
                "input_dim": int(input_dim),
                "sequence_length": int(sequence_length),
            },
            blocks=(
                NeuralBlockSpec(
                    kind="patchtst_block",
                    params={
                        "patch_len": int(patch_len),
                        "stride": int(stride or patch_len),
                        "hidden_dim": int(hidden_dim),
                        "num_layers": int(num_layers),
                        "num_heads": int(num_heads),
                        "ffn_dim": eff_ffn,
                        "output_dim": int(output_dim),
                        "dropout": float(dropout),
                    },
                    metadata={"family": "temporal", "route": "temporal_patchtst"},
                ),
            ),
            heads=tuple(NeuralHeadSpec.from_value(item) for item in head_specs),
            metadata={"family": "neural", "route": "temporal_patchtst"},
        )

    @classmethod
    def temporal_tft(
        cls,
        *,
        input_dim: int,
        sequence_length: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        output_dim: int = 1,
        dropout: float = 0.0,
        heads: Sequence[NeuralHeadSpec | Mapping[str, Any] | str] | None = None,
        name: str = "temporal_tft_graph",
    ) -> "NeuralGraphSpec":
        head_specs = heads or ({"kind": "forecast", "name": "forecast", "params": {"output_dim": int(output_dim)}},)
        return cls(
            name=name,
            input={
                "kind": "sequence",
                "input_dim": int(input_dim),
                "sequence_length": int(sequence_length),
            },
            blocks=(
                NeuralBlockSpec(
                    kind="tft_block",
                    params={
                        "hidden_dim": int(hidden_dim),
                        "num_layers": int(num_layers),
                        "num_heads": int(num_heads),
                        "output_dim": int(output_dim),
                        "dropout": float(dropout),
                    },
                    metadata={"family": "temporal", "route": "temporal_tft"},
                ),
            ),
            heads=tuple(NeuralHeadSpec.from_value(item) for item in head_specs),
            metadata={"family": "neural", "route": "temporal_tft"},
        )

    @classmethod
    def tabular_tabnet(
        cls,
        *,
        input_dim: int,
        hidden_dim: int = 64,
        n_steps: int = 4,
        relaxation_factor: float = 1.5,
        ghost_bn: bool = True,
        dropout: float = 0.0,
        heads: Sequence[NeuralHeadSpec | Mapping[str, Any] | str] | None = None,
        name: str = "tabular_tabnet_graph",
    ) -> "NeuralGraphSpec":
        head_specs = heads or ({"kind": "classification", "name": "classification", "params": {"num_classes": 2}},)
        return cls(
            name=name,
            input={"kind": "flat", "input_dim": int(input_dim)},
            blocks=(
                NeuralBlockSpec(
                    kind="tabnet_block",
                    params={
                        "hidden_dim": int(hidden_dim),
                        "n_steps": int(n_steps),
                        "relaxation_factor": float(relaxation_factor),
                        "ghost_bn": bool(ghost_bn),
                        "dropout": float(dropout),
                    },
                    metadata={"family": "tabular", "route": "tabular_tabnet"},
                ),
            ),
            heads=tuple(NeuralHeadSpec.from_value(item) for item in head_specs),
            metadata={"family": "neural", "route": "tabular_tabnet"},
        )

    def block_specs(self) -> tuple[NeuralBlockSpec, ...]:
        return tuple(NeuralBlockSpec.from_value(item) for item in self.blocks)

    def head_specs(self) -> tuple[NeuralHeadSpec, ...]:
        return tuple(NeuralHeadSpec.from_value(item) for item in self.heads)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "input": dict(self.input),
            "blocks": [item.as_dict() for item in self.block_specs()],
            "heads": [item.as_dict() for item in self.head_specs()],
            "parameterization": dict(self.parameterization),
            "audit": dict(self.audit),
            "metadata": dict(self.metadata),
        }


__all__ = [
    "NeuralBlockSpec",
    "NeuralComponentSpec",
    "NeuralGraphSpec",
    "NeuralHeadSpec",
]
