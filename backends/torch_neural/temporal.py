from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from mlblack.representations.codecs.neural.specs import NeuralGraphSpec, NeuralHeadSpec


TEMPORAL_LSTM_BLOCK_KINDS = {"lstm", "lstm_block", "temporal_lstm"}
TEMPORAL_TCN_BLOCK_KINDS = {"tcn", "tcn_block", "temporal_tcn"}
TEMPORAL_TRANSFORMER_BLOCK_KINDS = {
    "temporal_transformer",
    "temporal_transformer_block",
    "temporal_transformer_encoder_block",
}
TEMPORAL_NBEATS_BLOCK_KINDS = {"nbeats", "nbeats_block", "temporal_nbeats"}
TEMPORAL_DEEPAR_BLOCK_KINDS = {"deepar", "deepar_block", "temporal_deepar"}
TEMPORAL_PATCHTST_BLOCK_KINDS = {"patchtst", "patchtst_block", "temporal_patchtst"}
TEMPORAL_TFT_BLOCK_KINDS = {"tft", "tft_block", "temporal_tft"}
TEMPORAL_ROUTES = {"temporal_lstm", "temporal_tcn", "temporal_transformer", "temporal_nbeats", "temporal_deepar", "temporal_patchtst", "temporal_tft"}


def temporal_route(spec: NeuralGraphSpec) -> str:
    route = str(spec.metadata.get("route", "")).strip().lower()
    if route in TEMPORAL_ROUTES:
        return route
    blocks = spec.block_specs()
    if not blocks:
        return "unknown"
    kinds = {str(block.kind).lower() for block in blocks}
    if kinds.issubset(TEMPORAL_LSTM_BLOCK_KINDS):
        return "temporal_lstm"
    if kinds.issubset(TEMPORAL_TCN_BLOCK_KINDS):
        return "temporal_tcn"
    if kinds.issubset(TEMPORAL_TRANSFORMER_BLOCK_KINDS):
        return "temporal_transformer"
    if kinds.issubset(TEMPORAL_NBEATS_BLOCK_KINDS):
        return "temporal_nbeats"
    if kinds.issubset(TEMPORAL_DEEPAR_BLOCK_KINDS):
        return "temporal_deepar"
    if kinds.issubset(TEMPORAL_PATCHTST_BLOCK_KINDS):
        return "temporal_patchtst"
    if kinds.issubset(TEMPORAL_TFT_BLOCK_KINDS):
        return "temporal_tft"
    return "unknown"


def is_temporal_spec(spec: NeuralGraphSpec) -> bool:
    return temporal_route(spec) in TEMPORAL_ROUTES


def temporal_parameter_layout(spec: NeuralGraphSpec) -> tuple[tuple[tuple[int, ...], ...], tuple[str, ...]]:
    module = build_temporal_module(spec, random_seed=0)
    shapes = tuple(tuple(int(v) for v in param.detach().cpu().shape) for _, param in module.named_parameters())
    names = tuple(str(name) for name, _ in module.named_parameters())
    return shapes, names


def temporal_initial_values(spec: NeuralGraphSpec, *, random_seed: int = 42) -> np.ndarray:
    module = build_temporal_module(spec, random_seed=random_seed)
    arrays = [param.detach().cpu().numpy().reshape(-1) for _, param in module.named_parameters()]
    return np.concatenate(arrays).astype(float) if arrays else np.zeros(0, dtype=float)


def decode_temporal(values: np.ndarray, spec: NeuralGraphSpec, *, random_seed: int = 42) -> Any:
    module = build_temporal_module(spec, random_seed=random_seed)
    _load_flat_parameters(module, np.asarray(values, dtype=float).reshape(-1))
    return module


def build_temporal_module(spec: NeuralGraphSpec, *, random_seed: int = 42) -> Any:
    try:
        import torch
        from torch import nn
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("temporal neural graph routes require optional dependency 'torch'") from exc
    torch.manual_seed(int(random_seed))
    input_cfg = _input_config(spec)
    input_cfg["graph_spec"] = spec.as_dict()
    block_cfg = _block_config(spec)
    route = temporal_route(spec)
    return TemporalGraphModule(nn=nn, torch=torch, route=route, input_cfg=input_cfg, block_cfg=block_cfg, head_specs=spec.head_specs())


class TemporalGraphModule:
    def __new__(
        cls,
        *,
        nn: Any,
        torch: Any,
        route: str,
        input_cfg: Mapping[str, Any],
        block_cfg: Mapping[str, Any],
        head_specs: tuple[NeuralHeadSpec, ...],
    ) -> Any:
        class _TemporalGraphModule(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.route = str(route)
                self.graph_spec = dict(input_cfg.get("graph_spec", {}) or {})
                self.input_dim = int(input_cfg["input_dim"])
                self.sequence_length = int(input_cfg["sequence_length"])
                self.head_specs = tuple(head_specs)
                self.heads = nn.ModuleDict()

                if self.route == "temporal_lstm":
                    self.hidden_dim = int(block_cfg["hidden_dim"]) * (2 if bool(block_cfg.get("bidirectional", False)) else 1)
                    self.backbone = nn.LSTM(
                        input_size=self.input_dim,
                        hidden_size=int(block_cfg["hidden_dim"]),
                        num_layers=int(block_cfg["num_layers"]),
                        dropout=float(block_cfg.get("dropout", 0.0)) if int(block_cfg["num_layers"]) > 1 else 0.0,
                        bidirectional=bool(block_cfg.get("bidirectional", False)),
                        batch_first=True,
                    )
                    self.input_projection = None
                    self.position_embedding = None
                elif self.route == "temporal_tcn":
                    channels = tuple(int(v) for v in block_cfg["channels"])
                    self.hidden_dim = int(channels[-1] if channels else self.input_dim)
                    layers: list[Any] = []
                    in_ch = self.input_dim
                    for idx, out_ch in enumerate(channels):
                        dilation = int(block_cfg["dilation_base"]) ** int(idx)
                        padding = (int(block_cfg["kernel_size"]) - 1) * dilation
                        layers.append(nn.Conv1d(in_ch, out_ch, kernel_size=int(block_cfg["kernel_size"]), padding=padding, dilation=dilation))
                        layers.append(_activation_module(nn, str(block_cfg.get("activation", "relu"))))
                        if float(block_cfg.get("dropout", 0.0)) > 0.0:
                            layers.append(nn.Dropout(float(block_cfg["dropout"])))
                        in_ch = out_ch
                    self.backbone = nn.Sequential(*layers)
                    self.input_projection = None
                    self.position_embedding = None
                elif self.route == "temporal_transformer":
                    self.hidden_dim = int(input_cfg.get("hidden_dim", block_cfg["hidden_dim"]))
                    self.input_projection = nn.Linear(self.input_dim, self.hidden_dim)
                    self.position_embedding = nn.Embedding(self.sequence_length, self.hidden_dim)
                    layer = nn.TransformerEncoderLayer(
                        d_model=self.hidden_dim,
                        nhead=int(block_cfg["num_heads"]),
                        dim_feedforward=int(block_cfg["ffn_dim"]),
                        dropout=float(block_cfg.get("dropout", 0.0)),
                        activation=str(block_cfg.get("activation", "gelu")),
                        batch_first=True,
                    )
                    self.backbone = nn.TransformerEncoder(layer, num_layers=int(block_cfg["num_layers"]))
                elif self.route == "temporal_nbeats":
                    self.hidden_dim = int(block_cfg["hidden_dim"])
                    self.theta_dim = int(block_cfg["theta_dim"])
                    self.num_stacks = int(block_cfg.get("num_stacks", 2))
                    self.num_blocks_per_stack = int(block_cfg.get("num_blocks", 3))
                    self.nbeats_share_weights = bool(block_cfg.get("share_weights", False))
                    self.backcast_dim = int(self.input_dim) * int(self.sequence_length)
                    _forecast_dim = int(block_cfg.get("output_dim", 1))
                    for h in self.head_specs:
                        if str(h.kind).lower() in {"forecast", "point", "regression"}:
                            _forecast_dim = int(dict(h.params).get("output_dim", _forecast_dim))
                            break
                    total_blocks = self.num_stacks * self.num_blocks_per_stack
                    if self.nbeats_share_weights:
                        total_blocks = 1
                    self.backbone = nn.ModuleList([
                        _NBeatsBlock(
                            nn=nn,
                            input_dim=self.backcast_dim,
                            hidden_dim=self.hidden_dim,
                            theta_dim=self.theta_dim,
                            output_dim=_forecast_dim,
                            dropout=float(block_cfg.get("dropout", 0.0)),
                        )
                        for _ in range(total_blocks)
                    ])
                    self.hidden_dim = _forecast_dim
                    self.input_projection = None
                    self.position_embedding = None
                elif self.route == "temporal_deepar":
                    self.hidden_dim = int(block_cfg["hidden_dim"]) * (2 if bool(block_cfg.get("bidirectional", False)) else 1)
                    self.backbone = nn.LSTM(
                        input_size=self.input_dim,
                        hidden_size=int(block_cfg["hidden_dim"]),
                        num_layers=int(block_cfg["num_layers"]),
                        dropout=float(block_cfg.get("dropout", 0.0)) if int(block_cfg["num_layers"]) > 1 else 0.0,
                        bidirectional=bool(block_cfg.get("bidirectional", False)),
                        batch_first=True,
                    )
                    self.mu_projection = nn.Linear(self.hidden_dim, 1)
                    self.log_sigma_projection = nn.Linear(self.hidden_dim, 1)
                    self.input_projection = None
                    self.position_embedding = None
                elif self.route == "temporal_patchtst":
                    self.patch_len = int(block_cfg["patch_len"])
                    self.stride = int(block_cfg["stride"])
                    self.hidden_dim = int(block_cfg["hidden_dim"])
                    num_patches = (self.sequence_length - self.patch_len) // self.stride + 1
                    self.num_patches = max(1, num_patches)
                    self.patch_projection = nn.Linear(self.patch_len * self.input_dim, self.hidden_dim)
                    self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, self.hidden_dim) * 0.02)
                    self.pre_norm = nn.LayerNorm(self.hidden_dim)
                    layer = nn.TransformerEncoderLayer(
                        d_model=self.hidden_dim,
                        nhead=int(block_cfg["num_heads"]),
                        dim_feedforward=int(block_cfg["ffn_dim"]),
                        dropout=float(block_cfg.get("dropout", 0.0)),
                        activation="gelu",
                        batch_first=True,
                    )
                    self.backbone = nn.TransformerEncoder(layer, num_layers=int(block_cfg["num_layers"]))
                    self.input_projection = None
                    self.position_embedding = None
                elif self.route == "temporal_tft":
                    self.hidden_dim = int(block_cfg["hidden_dim"])
                    self.num_layers_tft = int(block_cfg.get("num_layers", 2))
                    self.num_heads_tft = int(block_cfg.get("num_heads", 4))
                    self.dropout_tft = float(block_cfg.get("dropout", 0.0))
                    self.input_projection = nn.Linear(self.input_dim, self.hidden_dim)
                    self.position_embedding = nn.Embedding(self.sequence_length, self.hidden_dim)
                    # GRN blocks
                    self._tft_grn = nn.ModuleList([
                        _GRN(nn=nn, hidden_dim=self.hidden_dim, dropout=self.dropout_tft)
                        for _ in range(self.num_layers_tft)
                    ])
                    # Multi-head self-attention
                    self._tft_attn = nn.MultiheadAttention(
                        self.hidden_dim, self.num_heads_tft,
                        dropout=self.dropout_tft, batch_first=True,
                    )
                    self._tft_attn_gate = nn.Sequential(
                        nn.Linear(self.hidden_dim * 2, self.hidden_dim), nn.GELU(), nn.Linear(self.hidden_dim, 1), nn.Sigmoid(),
                    )
                    self._tft_output_grn = _GRN(nn=nn, hidden_dim=self.hidden_dim, dropout=self.dropout_tft)
                    self.backbone = None  # TFT uses its own forward path
                else:
                    raise ValueError(f"unsupported temporal route: {self.route}")

                for head in self.head_specs:
                    key = _head_name(head)
                    params = dict(head.params)
                    kind = str(head.kind).lower()
                    if kind in {"forecast", "point", "regression", "classification", "classifier", "class"}:
                        out_dim = int(params.get("num_classes", params.get("output_dim", 1 if kind != "classification" else 2)))
                        self.heads[key] = nn.Linear(self.hidden_dim, out_dim)
                    elif kind in {"deepar"}:
                        pass
                    elif kind in {"embedding", "embedding_head"}:
                        self.heads[key] = nn.Linear(self.hidden_dim, int(params.get("output_dim", self.hidden_dim)))
                    else:
                        raise ValueError(f"unsupported temporal head kind: {head.kind}")

            def forward(self, sequence: Any, *, return_audit: bool = False) -> dict[str, Any]:
                if self.route == "temporal_nbeats":
                    x_flat = _coerce_sequence(torch, sequence, self.input_dim, self.sequence_length)
                    x_flat = x_flat.reshape(x_flat.shape[0], -1)
                    forecast_total = torch.zeros(x_flat.shape[0], self.hidden_dim, device=x_flat.device)
                    residual = x_flat
                    hidden_audit = x_flat
                    for block in self.backbone:
                        residual, block_forecast, block_hidden = block(residual)
                        if self.nbeats_share_weights:
                            forecast_total = forecast_total + block_forecast
                        else:
                            forecast_total = forecast_total + block_forecast
                        hidden_audit = block_hidden
                    hidden = hidden_audit
                    pooled = forecast_total
                elif self.route == "temporal_deepar":
                    x = _coerce_sequence(torch, sequence, self.input_dim, self.sequence_length)
                    hidden, _state = self.backbone(x)
                    pooled = hidden[:, -1, :]
                    mu = self.mu_projection(pooled)
                    log_sigma = self.log_sigma_projection(pooled)
                    head_outputs_dp: dict[str, Any] = {}
                    for head in self.head_specs:
                        key = _head_name(head)
                        kind = str(head.kind).lower()
                        if kind == "deepar":
                            head_outputs_dp[key] = {"mu": mu, "log_sigma": log_sigma}
                        else:
                            head_outputs_dp[key] = self.heads[key](pooled)
                    first_key = _head_name(self.head_specs[0]) if self.head_specs else ""
                    embedding_key = _first_head_key(self.head_specs, {"embedding", "embedding_head"})
                    return {
                        "hidden_states": hidden,
                        "head_outputs": head_outputs_dp,
                        "logits": head_outputs_dp.get(first_key),
                        "forecast": mu,
                        "mu": mu,
                        "log_sigma": log_sigma,
                        "embeddings": head_outputs_dp.get(embedding_key) if embedding_key else None,
                        "audit": {"route": self.route, "hidden_mean": torch.mean(hidden).detach()} if return_audit else {},
                    }
                elif self.route == "temporal_patchtst":
                    x = _coerce_sequence(torch, sequence, self.input_dim, self.sequence_length)
                    batch_size = x.shape[0]
                    patches = x.unfold(1, self.patch_len, self.stride)  # [B, n_patches, input_dim, patch_len]
                    patches = patches.contiguous().view(batch_size, self.num_patches, -1)  # [B, n_patches, input_dim*patch_len]
                    patch_emb = self.patch_projection(patches) + self.pos_embedding[:, :self.num_patches, :]
                    patch_emb = self.pre_norm(patch_emb)
                    hidden = self.backbone(patch_emb)
                    pooled = hidden.mean(dim=1)
                elif self.route == "temporal_tft":
                    x_base = _coerce_sequence(torch, sequence, self.input_dim, self.sequence_length)
                    positions = torch.arange(0, x_base.shape[1], device=x_base.device).unsqueeze(0).expand(x_base.shape[0], x_base.shape[1])
                    h = self.input_projection(x_base) + self.position_embedding(positions)  # [B, seq_len, hidden]
                    # Apply GRN blocks
                    for grn in self._tft_grn:
                        h = grn(h)
                    # Self-attention with gating
                    attn_out, _ = self._tft_attn(h, h, h)
                    gate_input = torch.cat([h, attn_out], dim=-1)
                    gate = self._tft_attn_gate(gate_input)
                    h = gate * attn_out + (1.0 - gate) * h
                    h = self._tft_output_grn(h)
                    hidden = h
                    pooled = hidden[:, -1, :]
                else:
                    x = _coerce_sequence(torch, sequence, self.input_dim, self.sequence_length)
                    if self.route == "temporal_lstm":
                        hidden, _state = self.backbone(x)
                        pooled = hidden[:, -1, :]
                    elif self.route == "temporal_tcn":
                        conv = self.backbone(x.transpose(1, 2))
                        conv = conv[:, :, -self.sequence_length :]
                        hidden = conv.transpose(1, 2)
                        pooled = hidden[:, -1, :]
                    elif self.route == "temporal_transformer":
                        positions = torch.arange(0, x.shape[1], device=x.device).unsqueeze(0).expand(x.shape[0], x.shape[1])
                        hidden = self.input_projection(x) + self.position_embedding(positions)
                        hidden = self.backbone(hidden)
                        pooled = hidden[:, -1, :]
                    else:
                        raise ValueError(f"unsupported temporal route: {self.route}")

                head_outputs: dict[str, Any] = {}
                for head in self.head_specs:
                    key = _head_name(head)
                    head_outputs[key] = self.heads[key](pooled)
                first_key = _head_name(self.head_specs[0]) if self.head_specs else ""
                embedding_key = _first_head_key(self.head_specs, {"embedding", "embedding_head"})
                return {
                    "hidden_states": hidden,
                    "head_outputs": head_outputs,
                    "logits": head_outputs.get(first_key),
                    "forecast": head_outputs.get(first_key),
                    "embeddings": head_outputs.get(embedding_key) if embedding_key else None,
                    "audit": {"route": self.route, "hidden_mean": torch.mean(hidden).detach()} if return_audit else {},
                }

            def describe(self) -> dict[str, Any]:
                return {
                    "kind": self.route,
                    "input_dim": int(self.input_dim),
                    "sequence_length": int(self.sequence_length),
                    "hidden_dim": int(self.hidden_dim),
                    "heads": tuple(_head_name(head) for head in self.head_specs),
                }

        return _TemporalGraphModule()


def _input_config(spec: NeuralGraphSpec) -> dict[str, Any]:
    cfg = dict(spec.input)
    for key in ("input_dim", "sequence_length"):
        if int(cfg.get(key, 0)) <= 0:
            raise ValueError(f"temporal spec requires input.{key}")
    return cfg


def _block_config(spec: NeuralGraphSpec) -> dict[str, Any]:
    blocks = spec.block_specs()
    if len(blocks) != 1:
        raise ValueError("temporal neural graph expects exactly one temporal block spec")
    block = blocks[0]
    params = dict(block.params)
    route = temporal_route(spec)
    if route == "temporal_lstm":
        return {
            "hidden_dim": int(params.get("hidden_dim", 32)),
            "num_layers": int(params.get("num_layers", 1)),
            "dropout": float(params.get("dropout", 0.0)),
            "bidirectional": bool(params.get("bidirectional", False)),
        }
    if route == "temporal_tcn":
        return {
            "channels": tuple(int(v) for v in params.get("channels", (32, 32))),
            "kernel_size": int(params.get("kernel_size", 3)),
            "dilation_base": int(params.get("dilation_base", 2)),
            "activation": str(params.get("activation", "relu")),
            "dropout": float(params.get("dropout", 0.0)),
        }
    if route == "temporal_transformer":
        mechanisms = block.mechanism_specs()
        attention = dict(mechanisms.get("attention", {}).params) if mechanisms.get("attention") is not None else {}
        ffn = dict(mechanisms.get("ffn", {}).params) if mechanisms.get("ffn") is not None else {}
        hidden_dim = int(dict(spec.input).get("hidden_dim", params.get("hidden_dim", 64)))
        return {
            "hidden_dim": hidden_dim,
            "num_layers": int(block.repeat),
            "num_heads": int(attention.get("num_heads", params.get("num_heads", 4))),
            "ffn_dim": int(round(hidden_dim * float(ffn.get("expansion_ratio", params.get("ffn_expansion_ratio", 4.0))))),
            "activation": str(ffn.get("activation", params.get("activation", "gelu"))),
            "dropout": float(ffn.get("dropout", attention.get("dropout", params.get("dropout", 0.0)))),
        }
    if route == "temporal_nbeats":
        return {
            "hidden_dim": int(params.get("hidden_dim", 64)),
            "theta_dim": int(params.get("theta_dim", 8)),
            "num_stacks": int(params.get("num_stacks", 2)),
            "num_blocks": int(params.get("num_blocks", 3)),
            "share_weights": bool(params.get("share_weights", False)),
            "output_dim": int(params.get("output_dim", 1)),
            "dropout": float(params.get("dropout", 0.0)),
        }
    if route == "temporal_deepar":
        return {
            "hidden_dim": int(params.get("hidden_dim", 32)),
            "num_layers": int(params.get("num_layers", 1)),
            "dropout": float(params.get("dropout", 0.0)),
            "bidirectional": bool(params.get("bidirectional", False)),
        }
    if route == "temporal_patchtst":
        return {
            "patch_len": int(params.get("patch_len", 8)),
            "stride": int(params.get("stride", params.get("patch_len", 8))),
            "hidden_dim": int(params.get("hidden_dim", 64)),
            "num_layers": int(params.get("num_layers", 2)),
            "num_heads": int(params.get("num_heads", 4)),
            "ffn_dim": int(params.get("ffn_dim", int(params.get("hidden_dim", 64)) * 4)),
            "dropout": float(params.get("dropout", 0.0)),
        }
    if route == "temporal_tft":
        return {
            "hidden_dim": int(params.get("hidden_dim", 64)),
            "num_layers": int(params.get("num_layers", 2)),
            "num_heads": int(params.get("num_heads", 4)),
            "dropout": float(params.get("dropout", 0.0)),
        }
    raise ValueError(f"unsupported temporal route: {route}")


class _GRN:
    """Gated Residual Network: LayerNorm(x + Dropout(ELU(Linear2(Dropout(ReLU(Linear1(x)))))))."""

    def __new__(cls, *, nn: Any, hidden_dim: int, dropout: float = 0.0) -> Any:
        class _GatedResidual(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(int(hidden_dim), int(hidden_dim))
                self.fc2 = nn.Linear(int(hidden_dim), int(hidden_dim))
                self.drop = nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity()
                self.norm = nn.LayerNorm(int(hidden_dim))

            def forward(self, x):
                residual = x
                h = self.drop(nn.functional.relu(self.fc1(x)))
                h = nn.functional.elu(self.fc2(h))
                h = self.drop(h)
                return self.norm(residual + h)

        return _GatedResidual()


class _NBeatsBlock:
    """Single N-BEATS residual block: FC stack → theta → backcast + forecast."""

    def __new__(
        cls,
        *,
        nn: Any,
        input_dim: int,
        hidden_dim: int,
        theta_dim: int,
        output_dim: int,
        dropout: float = 0.0,
    ) -> Any:
        class _Block(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.input_dim = int(input_dim)
                self.hidden_dim = int(hidden_dim)
                self.theta_dim = int(theta_dim)
                self.output_dim = int(output_dim)
                self.fc_stack = nn.Sequential(
                    nn.Linear(self.input_dim, self.hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity(),
                    nn.Linear(self.hidden_dim, self.hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity(),
                    nn.Linear(self.hidden_dim, self.hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity(),
                    nn.Linear(self.hidden_dim, self.theta_dim),
                )
                self.backcast_projection = nn.Linear(self.theta_dim, self.input_dim)
                self.forecast_projection = nn.Linear(self.theta_dim, self.output_dim)

            def forward(self, x: Any) -> tuple[Any, Any, Any]:
                theta = self.fc_stack(x)
                backcast = self.backcast_projection(theta)
                forecast = self.forecast_projection(theta)
                return backcast, forecast, theta

        return _Block()


def _coerce_sequence(torch: Any, sequence: Any, input_dim: int, sequence_length: int) -> Any:
    x = sequence.float() if hasattr(sequence, "float") else torch.as_tensor(sequence, dtype=torch.float32)
    if x.ndim == 2:
        expected = int(input_dim) * int(sequence_length)
        if x.shape[1] != expected:
            raise ValueError(f"2D temporal input expects {expected} flattened features")
        x = x.reshape(x.shape[0], int(sequence_length), int(input_dim))
    if x.ndim != 3:
        raise ValueError("temporal input must have shape [batch, sequence_length, input_dim] or flattened [batch, sequence_length * input_dim]")
    if x.shape[1] != int(sequence_length) or x.shape[2] != int(input_dim):
        raise ValueError("temporal input shape does not match NeuralGraphSpec input")
    return x


def _activation_module(nn: Any, activation: str) -> Any:
    key = str(activation or "relu").lower()
    if key == "relu":
        return nn.ReLU()
    if key == "gelu":
        return nn.GELU()
    if key in {"silu", "swish"}:
        return nn.SiLU()
    if key == "tanh":
        return nn.Tanh()
    if key in {"identity", "linear", "none"}:
        return nn.Identity()
    raise ValueError(f"unsupported temporal activation: {activation}")


def _head_name(head: NeuralHeadSpec) -> str:
    return str(head.name or head.kind or "head")


def _first_head_key(heads: tuple[NeuralHeadSpec, ...], kinds: set[str]) -> str:
    for head in heads:
        if str(head.kind).lower() in kinds:
            return _head_name(head)
    return ""


def _load_flat_parameters(module: Any, values: np.ndarray) -> None:
    arr = np.asarray(values, dtype=float).reshape(-1)
    expected = int(sum(param.numel() for _, param in module.named_parameters()))
    if arr.shape[0] != expected:
        raise ValueError(f"parameter vector has {arr.shape[0]} values but temporal graph expects {expected}")
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("temporal neural graph routes require optional dependency 'torch'") from exc
    offset = 0
    with torch.no_grad():
        for _name, param in module.named_parameters():
            size = int(param.numel())
            block = arr[offset : offset + size].reshape(tuple(param.shape))
            offset += size
            param.copy_(torch.as_tensor(block, dtype=param.dtype, device=param.device))


__all__ = [
    "build_temporal_module",
    "decode_temporal",
    "is_temporal_spec",
    "temporal_initial_values",
    "temporal_parameter_layout",
    "temporal_route",
]
