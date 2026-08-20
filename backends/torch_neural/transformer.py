from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from mlblack.representations.codecs.neural.specs import NeuralBlockSpec, NeuralGraphSpec, NeuralHeadSpec


TRANSFORMER_BLOCK_KINDS = {
    "transformer_decoder_block",
    "decoder_transformer_block",
    "transformer_encoder_block",
    "tiny_transformer_block",
}


def is_tiny_transformer_spec(spec: NeuralGraphSpec) -> bool:
    blocks = spec.block_specs()
    return bool(blocks) and all(str(block.kind).lower() in TRANSFORMER_BLOCK_KINDS for block in blocks)


def transformer_parameter_layout(spec: NeuralGraphSpec) -> tuple[tuple[tuple[int, ...], ...], tuple[str, ...]]:
    module = build_tiny_transformer_module(spec, random_seed=0)
    shapes = tuple(tuple(int(v) for v in param.detach().cpu().shape) for _, param in module.named_parameters())
    names = tuple(str(name) for name, _ in module.named_parameters())
    return shapes, names


def transformer_initial_values(spec: NeuralGraphSpec, *, random_seed: int = 42) -> np.ndarray:
    module = build_tiny_transformer_module(spec, random_seed=random_seed)
    arrays = [param.detach().cpu().numpy().reshape(-1) for _, param in module.named_parameters()]
    return np.concatenate(arrays).astype(float) if arrays else np.zeros(0, dtype=float)


def decode_tiny_transformer(values: np.ndarray, spec: NeuralGraphSpec, *, random_seed: int = 42) -> Any:
    module = build_tiny_transformer_module(spec, random_seed=random_seed)
    _load_flat_parameters(module, np.asarray(values, dtype=float).reshape(-1))
    return module


def build_tiny_transformer_module(spec: NeuralGraphSpec, *, random_seed: int = 42) -> Any:
    try:
        import torch
        from torch import nn
    except Exception as exc:  # pragma: no cover - exercised only when torch is missing.
        raise RuntimeError("tiny Transformer neural graph requires optional dependency 'torch'") from exc

    torch.manual_seed(int(random_seed))
    input_cfg = _input_config(spec)
    input_cfg["graph_spec"] = spec.as_dict()
    block_cfg = _block_config(spec)
    head_specs = spec.head_specs()
    return TinyTransformerGraphModule(nn=nn, torch=torch, input_cfg=input_cfg, block_cfg=block_cfg, head_specs=head_specs)


class TinyTransformerGraphModule:  # intentionally assigned nn.Module at runtime
    def __new__(cls, *, nn: Any, torch: Any, input_cfg: Mapping[str, Any], block_cfg: Mapping[str, Any], head_specs: tuple[NeuralHeadSpec, ...]) -> Any:
        class _TinyTransformerGraphModule(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.graph_name = str(input_cfg.get("name", "tiny_transformer"))
                self.route = "tiny_transformer"
                self.graph_spec = dict(input_cfg.get("graph_spec", {}) or {})
                self.vocab_size = int(input_cfg["vocab_size"])
                self.max_length = int(input_cfg["max_length"])
                self.hidden_dim = int(input_cfg["hidden_dim"])
                self.position_encoding = str(input_cfg.get("position_encoding", "learned")).lower()
                self.token_embedding = nn.Embedding(self.vocab_size, self.hidden_dim)
                self.position_embedding = (
                    nn.Embedding(self.max_length, self.hidden_dim)
                    if self.position_encoding in {"learned", "learned_absolute", "absolute"}
                    else None
                )
                self.dropout = nn.Dropout(float(block_cfg.get("dropout", 0.0)))
                self.blocks = nn.ModuleList(
                    [
                        TinyTransformerBlock(
                            nn=nn,
                            torch=torch,
                            hidden_dim=self.hidden_dim,
                            num_heads=int(block_cfg["num_heads"]),
                            ffn_dim=int(block_cfg["ffn_dim"]),
                            activation=str(block_cfg.get("activation", "gelu")),
                            ffn_kind=str(block_cfg.get("ffn_kind", "mlp")),
                            dropout=float(block_cfg.get("dropout", 0.0)),
                            norm_kind=str(block_cfg.get("norm_kind", "layer_norm")),
                            norm_position=str(block_cfg.get("norm_position", "pre")),
                            causal=bool(block_cfg.get("causal", True)),
                            position_encoding=str(block_cfg.get("position_encoding", self.position_encoding)),
                            lora=dict(block_cfg.get("lora", {}) or {}),
                        )
                        for _ in range(int(block_cfg["num_layers"]))
                    ]
                )
                self.final_norm = _make_norm(nn, torch, self.hidden_dim, str(block_cfg.get("norm_kind", "layer_norm")))
                self.heads = nn.ModuleDict()
                self.head_specs = tuple(head_specs)
                for head in self.head_specs:
                    key = _head_name(head)
                    kind = str(head.kind).lower()
                    params = dict(head.params)
                    if kind in {"classification", "classifier", "class"}:
                        self.heads[key] = nn.Linear(self.hidden_dim, int(params.get("num_classes", 2)))
                    elif kind in {"language_modeling", "lm", "causal_lm"}:
                        self.heads[key] = nn.Linear(self.hidden_dim, int(params.get("vocab_size", self.vocab_size)))
                    elif kind in {"embedding", "embedding_head"}:
                        self.heads[key] = nn.Linear(self.hidden_dim, int(params.get("output_dim", self.hidden_dim)))
                    elif kind in {"ranking", "ranker", "rank"}:
                        self.heads[key] = nn.Linear(self.hidden_dim, int(params.get("output_dim", 1)))
                    elif kind in {"preference", "preference_score", "reward"}:
                        self.heads[key] = nn.Linear(self.hidden_dim, int(params.get("output_dim", 1)))
                    else:
                        raise ValueError(f"unsupported tiny Transformer head kind: {head.kind}")

            def forward(
                self,
                input_ids: Any,
                attention_mask: Any | None = None,
                *,
                return_audit: bool = False,
                use_cache: bool = False,
                past_key_values: Any | None = None,
            ) -> dict[str, Any]:
                if input_ids.ndim != 2:
                    raise ValueError("input_ids must have shape [batch, seq_len]")
                batch_size, seq_len = input_ids.shape
                position_offset = _past_length(past_key_values)
                if int(seq_len) + int(position_offset) > self.max_length:
                    raise ValueError(f"sequence length {seq_len + position_offset} exceeds max_length {self.max_length}")
                positions = (
                    torch.arange(int(position_offset), int(position_offset) + int(seq_len), device=input_ids.device)
                    .unsqueeze(0)
                    .expand(int(batch_size), int(seq_len))
                )
                x = self.token_embedding(input_ids)
                if self.position_embedding is not None:
                    x = x + self.position_embedding(positions)
                x = self.dropout(x)
                audit: dict[str, Any] = {"attention_maps": [], "ffn_activations": []}
                present_key_values: list[Any] = []
                past_values = tuple(past_key_values or ())
                for block_idx, block in enumerate(self.blocks):
                    block_past = past_values[block_idx] if block_idx < len(past_values) else None
                    x, block_audit, block_present = block(
                        x,
                        attention_mask=attention_mask,
                        return_audit=return_audit,
                        past_key_value=block_past,
                        use_cache=bool(use_cache),
                        position_offset=int(position_offset),
                    )
                    if use_cache:
                        present_key_values.append(block_present)
                    if return_audit:
                        audit["attention_maps"].append(block_audit.get("attention"))
                        audit["ffn_activations"].append(block_audit.get("ffn_activation"))
                hidden = self.final_norm(x)
                head_outputs: dict[str, Any] = {}
                for head in self.head_specs:
                    key = _head_name(head)
                    kind = str(head.kind).lower()
                    layer = self.heads[key]
                    if kind in {"classification", "classifier", "class"}:
                        pooled = _pool_hidden(torch, hidden, attention_mask, str(dict(head.params).get("pooling", "mean")))
                        head_outputs[key] = layer(pooled)
                    elif kind in {"language_modeling", "lm", "causal_lm"}:
                        head_outputs[key] = layer(hidden)
                    elif kind in {"embedding", "embedding_head"}:
                        pooled = _pool_hidden(torch, hidden, attention_mask, str(dict(head.params).get("pooling", "mean")))
                        head_outputs[key] = layer(pooled)
                    elif kind in {"ranking", "ranker", "rank"}:
                        pooled = _pool_hidden(torch, hidden, attention_mask, str(dict(head.params).get("pooling", "mean")))
                        head_outputs[key] = layer(pooled).reshape(hidden.shape[0], -1)
                    elif kind in {"preference", "preference_score", "reward"}:
                        pooled = _pool_hidden(torch, hidden, attention_mask, str(dict(head.params).get("pooling", "mean")))
                        head_outputs[key] = layer(pooled).reshape(hidden.shape[0], -1)
                first_key = _head_name(self.head_specs[0]) if self.head_specs else ""
                embedding_key = _first_head_key(self.head_specs, {"embedding", "embedding_head"})
                ranking_key = _first_head_key(self.head_specs, {"ranking", "ranker", "rank"})
                preference_key = _first_head_key(self.head_specs, {"preference", "preference_score", "reward"})
                return {
                    "hidden_states": hidden,
                    "head_outputs": head_outputs,
                    "logits": head_outputs.get(first_key),
                    "embeddings": head_outputs.get(embedding_key) if embedding_key else None,
                    "ranking_scores": head_outputs.get(ranking_key) if ranking_key else None,
                    "preference_scores": head_outputs.get(preference_key) if preference_key else None,
                    "audit": audit if return_audit else {},
                    "past_key_values": tuple(present_key_values) if use_cache else tuple(),
                    "kv_cache": _make_kv_cache(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        past_key_values=tuple(present_key_values),
                        position_offset=int(position_offset),
                    )
                    if use_cache
                    else {},
                }

            def prefill_cache(self, input_ids: Any, attention_mask: Any | None = None) -> dict[str, Any]:
                self.eval()
                with torch.no_grad():
                    output = self(input_ids, attention_mask=attention_mask, use_cache=True)
                return dict(output.get("kv_cache", {}) or {})

            def generate(
                self,
                input_ids: Any,
                *,
                max_new_tokens: int = 8,
                attention_mask: Any | None = None,
                temperature: float = 0.0,
                top_k: int | None = None,
                eos_token_id: int | None = None,
                use_cache: bool = True,
                return_cache: bool = False,
            ) -> Any:
                self.eval()
                generated = input_ids.clone()
                mask = attention_mask.clone() if attention_mask is not None else torch.ones_like(generated, dtype=torch.long)
                cache: dict[str, Any] = {}
                past: Any | None = None
                with torch.no_grad():
                    if bool(use_cache):
                        output = self(generated, attention_mask=mask, use_cache=True)
                        past = output.get("past_key_values", tuple())
                        cache = dict(output.get("kv_cache", {}) or {})
                        logits = output["logits"]
                    else:
                        output = self(generated, attention_mask=mask)
                        logits = output["logits"]
                    for _ in range(int(max_new_tokens)):
                        next_token = _select_next_token(torch, logits[:, -1, :], temperature=float(temperature), top_k=top_k)
                        generated = torch.cat((generated, next_token), dim=1)
                        next_mask = torch.ones((generated.shape[0], 1), dtype=mask.dtype, device=mask.device)
                        mask = torch.cat((mask, next_mask), dim=1)
                        if eos_token_id is not None and bool(torch.all(next_token.reshape(-1) == int(eos_token_id)).item()):
                            break
                        if bool(use_cache):
                            output = self(next_token, attention_mask=mask, use_cache=True, past_key_values=past)
                            past = output.get("past_key_values", tuple())
                            cache = dict(output.get("kv_cache", {}) or {})
                            logits = output["logits"]
                        else:
                            output = self(generated, attention_mask=mask)
                            logits = output["logits"]
                if return_cache:
                    cache["tokens"] = generated.detach()
                    cache["attention_mask"] = mask.detach()
                    return generated, cache
                return generated

            def describe(self) -> dict[str, Any]:
                return {
                    "kind": "tiny_transformer",
                    "vocab_size": int(self.vocab_size),
                    "max_length": int(self.max_length),
                    "hidden_dim": int(self.hidden_dim),
                    "num_blocks": int(len(self.blocks)),
                    "position_encoding": str(self.position_encoding),
                    "heads": tuple(_head_name(head) for head in self.head_specs),
                }

        return _TinyTransformerGraphModule()


class TinySelfAttention:
    def __new__(
        cls,
        *,
        nn: Any,
        torch: Any,
        hidden_dim: int,
        num_heads: int,
        dropout: float,
        causal: bool,
        position_encoding: str,
        lora: Mapping[str, Any],
    ) -> Any:
        class _TinySelfAttention(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.hidden_dim = int(hidden_dim)
                self.num_heads = int(num_heads)
                if self.hidden_dim % max(1, self.num_heads) != 0:
                    raise ValueError("hidden_dim must be divisible by num_heads")
                self.head_dim = self.hidden_dim // self.num_heads
                self.causal = bool(causal)
                self.position_encoding = str(position_encoding).lower()
                self.q_proj = _make_linear(nn, torch, self.hidden_dim, self.hidden_dim, lora, target="attention.q")
                self.k_proj = _make_linear(nn, torch, self.hidden_dim, self.hidden_dim, lora, target="attention.k")
                self.v_proj = _make_linear(nn, torch, self.hidden_dim, self.hidden_dim, lora, target="attention.v")
                self.out_proj = _make_linear(nn, torch, self.hidden_dim, self.hidden_dim, lora, target="attention.out")
                self.dropout = nn.Dropout(float(dropout))

            def forward(
                self,
                x: Any,
                attention_mask: Any | None = None,
                *,
                need_weights: bool = False,
                past_key_value: Any | None = None,
                use_cache: bool = False,
                position_offset: int = 0,
            ) -> tuple[Any, Any | None, Any | None]:
                batch_size, seq_len, _hidden_dim = x.shape
                q = self._split_heads(self.q_proj(x))
                k = self._split_heads(self.k_proj(x))
                v = self._split_heads(self.v_proj(x))
                if self.position_encoding in {"rope", "rotary", "rotary_position"}:
                    q, k = _apply_rope(torch, q, k, position_offset=int(position_offset))
                if past_key_value is not None:
                    past_k, past_v = past_key_value
                    k = torch.cat((past_k.to(device=k.device, dtype=k.dtype), k), dim=-2)
                    v = torch.cat((past_v.to(device=v.device, dtype=v.dtype), v), dim=-2)
                scale = float(self.head_dim) ** -0.5
                scores = torch.matmul(q, k.transpose(-2, -1)) * scale
                if self.causal:
                    key_len = int(k.shape[-2])
                    query_positions = torch.arange(
                        int(position_offset),
                        int(position_offset) + int(seq_len),
                        device=x.device,
                        dtype=torch.long,
                    ).unsqueeze(-1)
                    key_positions = torch.arange(0, key_len, device=x.device, dtype=torch.long).unsqueeze(0)
                    causal_mask = torch.triu(
                        torch.ones((int(seq_len), int(key_len)), device=x.device, dtype=torch.bool),
                        diagonal=max(1, key_len - int(seq_len) + 1),
                    )
                    causal_mask = causal_mask | (key_positions > query_positions)
                    scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
                if attention_mask is not None:
                    key_padding_mask = attention_mask.to(dtype=torch.bool) == 0
                    if key_padding_mask.shape[-1] != scores.shape[-1]:
                        key_padding_mask = key_padding_mask[:, -scores.shape[-1] :]
                    scores = scores.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf"))
                weights = torch.softmax(scores, dim=-1)
                weights = torch.nan_to_num(weights, nan=0.0)
                out = torch.matmul(self.dropout(weights), v)
                out = out.transpose(1, 2).contiguous().reshape(int(batch_size), int(seq_len), self.hidden_dim)
                present = (k.detach(), v.detach()) if use_cache else None
                return self.out_proj(out), weights if need_weights else None, present

            def _split_heads(self, x: Any) -> Any:
                batch_size, seq_len, _hidden_dim = x.shape
                return x.reshape(int(batch_size), int(seq_len), self.num_heads, self.head_dim).transpose(1, 2)

        return _TinySelfAttention()


class TinyRMSNorm:
    def __new__(cls, *, nn: Any, torch: Any, hidden_dim: int, eps: float = 1e-6) -> Any:
        class _TinyRMSNorm(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = nn.Parameter(torch.ones(int(hidden_dim)))
                self.eps = float(eps)

            def forward(self, x: Any) -> Any:
                scale = torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
                return (x * scale) * self.weight

        return _TinyRMSNorm()


class TinyLoRALinear:
    def __new__(
        cls,
        *,
        nn: Any,
        torch: Any,
        in_features: int,
        out_features: int,
        bias: bool,
        rank: int,
        alpha: float,
        dropout: float,
        freeze_base: bool,
        quantize_base: bool,
        quantization_bits: int,
    ) -> Any:
        class _TinyLoRALinear(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.base = nn.Linear(int(in_features), int(out_features), bias=bool(bias))
                self.rank = int(rank)
                self.scaling = float(alpha) / max(1, self.rank)
                self.quantize_base = bool(quantize_base)
                self.quantization_bits = int(quantization_bits)
                self.dropout = nn.Dropout(float(dropout))
                self.lora_a = nn.Parameter(torch.empty(self.rank, int(in_features)))
                self.lora_b = nn.Parameter(torch.zeros(int(out_features), self.rank))
                nn.init.kaiming_uniform_(self.lora_a, a=5**0.5)
                if self.quantize_base:
                    with torch.no_grad():
                        self.base.weight.copy_(_fake_quantize_tensor(torch, self.base.weight, bits=self.quantization_bits))
                        if self.base.bias is not None:
                            self.base.bias.copy_(_fake_quantize_tensor(torch, self.base.bias, bits=self.quantization_bits))
                if bool(freeze_base) or self.quantize_base:
                    for param in self.base.parameters():
                        param.requires_grad_(False)

            def forward(self, x: Any) -> Any:
                delta = self.dropout(x).matmul(self.lora_a.transpose(0, 1)).matmul(self.lora_b.transpose(0, 1))
                return self.base(x) + (delta * self.scaling)

        return _TinyLoRALinear()


class TinyTransformerBlock:
    def __new__(
        cls,
        *,
        nn: Any,
        torch: Any,
        hidden_dim: int,
        num_heads: int,
        ffn_dim: int,
        activation: str,
        ffn_kind: str,
        dropout: float,
        norm_kind: str,
        norm_position: str,
        causal: bool,
        position_encoding: str,
        lora: Mapping[str, Any],
    ) -> Any:
        class _TinyTransformerBlock(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.hidden_dim = int(hidden_dim)
                self.num_heads = int(num_heads)
                self.ffn_dim = int(ffn_dim)
                self.ffn_kind = str(ffn_kind).lower()
                self.norm_position = str(norm_position).lower()
                self.causal = bool(causal)
                self.attention = TinySelfAttention(
                    nn=nn,
                    torch=torch,
                    hidden_dim=self.hidden_dim,
                    num_heads=self.num_heads,
                    dropout=float(dropout),
                    causal=self.causal,
                    position_encoding=str(position_encoding),
                    lora=dict(lora or {}),
                )
                self.norm1 = _make_norm(nn, torch, self.hidden_dim, str(norm_kind))
                self.norm2 = _make_norm(nn, torch, self.hidden_dim, str(norm_kind))
                self.ffn_up = _make_linear(nn, torch, self.hidden_dim, self.ffn_dim, lora, target="ffn.up")
                self.ffn_gate = (
                    _make_linear(nn, torch, self.hidden_dim, self.ffn_dim, lora, target="ffn.gate")
                    if self.ffn_kind in {"swiglu", "geglu", "gated_mlp", "gated"}
                    else None
                )
                self.ffn_down = _make_linear(nn, torch, self.ffn_dim, self.hidden_dim, lora, target="ffn.down")
                self.dropout = nn.Dropout(float(dropout))
                self.activation_name = str(activation).lower()

            def forward(
                self,
                x: Any,
                attention_mask: Any | None = None,
                *,
                return_audit: bool = False,
                past_key_value: Any | None = None,
                use_cache: bool = False,
                position_offset: int = 0,
            ) -> tuple[Any, dict[str, Any], Any | None]:
                if self.norm_position == "pre":
                    attn_input = self.norm1(x)
                    attn_out, attn_weights, present = self.attention(
                        attn_input,
                        attention_mask=attention_mask,
                        need_weights=bool(return_audit),
                        past_key_value=past_key_value,
                        use_cache=bool(use_cache),
                        position_offset=int(position_offset),
                    )
                    x = x + self.dropout(attn_out)
                    ffn_input = self.norm2(x)
                    ffn_hidden = self._ffn_hidden(ffn_input)
                    x = x + self.dropout(self.ffn_down(ffn_hidden))
                else:
                    attn_out, attn_weights, present = self.attention(
                        x,
                        attention_mask=attention_mask,
                        need_weights=bool(return_audit),
                        past_key_value=past_key_value,
                        use_cache=bool(use_cache),
                        position_offset=int(position_offset),
                    )
                    x = self.norm1(x + self.dropout(attn_out))
                    ffn_hidden = self._ffn_hidden(x)
                    x = self.norm2(x + self.dropout(self.ffn_down(ffn_hidden)))
                audit = {}
                if return_audit:
                    audit = {"attention": attn_weights, "ffn_activation": ffn_hidden.detach()}
                return x, audit, present

            def _ffn_hidden(self, x: Any) -> Any:
                up = self.ffn_up(x)
                if self.ffn_gate is not None:
                    gate = self.ffn_gate(x)
                    gate_activation = "gelu" if self.ffn_kind == "geglu" else "silu"
                    return _activate(torch, up, gate_activation) * gate
                return _activate(torch, up, self.activation_name)

        return _TinyTransformerBlock()


def _input_config(spec: NeuralGraphSpec) -> dict[str, Any]:
    cfg = dict(spec.input)
    if int(cfg.get("vocab_size", 0)) <= 0:
        raise ValueError("tiny Transformer spec requires input.vocab_size")
    if int(cfg.get("max_length", 0)) <= 0:
        raise ValueError("tiny Transformer spec requires input.max_length")
    hidden_dim = int(cfg.get("hidden_dim", cfg.get("embedding_dim", 0)))
    if hidden_dim <= 0:
        raise ValueError("tiny Transformer spec requires input.hidden_dim")
    cfg["hidden_dim"] = hidden_dim
    cfg.setdefault("position_encoding", "learned")
    cfg.setdefault("name", spec.name)
    return cfg


def _block_config(spec: NeuralGraphSpec) -> dict[str, Any]:
    blocks = spec.block_specs()
    if len(blocks) != 1:
        raise ValueError("tiny Transformer currently expects one repeated block spec")
    block: NeuralBlockSpec = blocks[0]
    mechanisms = block.mechanism_specs()
    attention = mechanisms.get("attention")
    ffn = mechanisms.get("ffn")
    input_cfg = _input_config(spec)
    hidden_dim = int(input_cfg["hidden_dim"])
    attn_params = dict(attention.params if attention is not None else {})
    ffn_params = dict(ffn.params if ffn is not None else {})
    num_heads = int(attn_params.get("num_heads", 1))
    if hidden_dim % max(1, num_heads) != 0:
        raise ValueError("hidden_dim must be divisible by num_heads")
    expansion = float(ffn_params.get("expansion_ratio", 4.0))
    ffn_dim = int(ffn_params.get("ffn_dim", max(1, round(hidden_dim * expansion))))
    attention_kind = str(attention.kind if attention is not None else "causal_self_attention").lower()
    ffn_kind = str(ffn.kind if ffn is not None else ffn_params.get("kind", "mlp")).lower()
    lora_cfg = dict(spec.parameterization.get("lora", {}) or {})
    qlora_cfg = dict(spec.parameterization.get("qlora", {}) or {})
    if qlora_cfg:
        lora_cfg = {
            "enabled": True,
            "rank": int(qlora_cfg.get("rank", lora_cfg.get("rank", 4))),
            "alpha": float(qlora_cfg.get("alpha", lora_cfg.get("alpha", 8.0))),
            "targets": tuple(qlora_cfg.get("targets", lora_cfg.get("targets", ("attention.q", "attention.v")))),
            "dropout": float(qlora_cfg.get("dropout", lora_cfg.get("dropout", 0.0))),
            "freeze_base": True,
            "quantize_base": True,
            "qlora": dict(qlora_cfg),
            **lora_cfg,
        }
    return {
        "num_layers": int(block.repeat),
        "num_heads": num_heads,
        "ffn_dim": ffn_dim,
        "ffn_kind": ffn_kind,
        "activation": str(ffn_params.get("activation", "gelu")),
        "dropout": float(ffn_params.get("dropout", attn_params.get("dropout", 0.0))),
        "norm_kind": str(dict(block.norm).get("kind", "layer_norm")),
        "norm_position": str(dict(block.norm).get("position", "pre")),
        "position_encoding": str(attn_params.get("position_encoding", input_cfg.get("position_encoding", "learned"))),
        "lora": lora_cfg,
        "causal": attention_kind in {"causal_self_attention", "causal", "decoder_self_attention"},
    }


def _head_name(head: NeuralHeadSpec) -> str:
    return str(head.name or head.kind or "head")


def _first_head_key(heads: tuple[NeuralHeadSpec, ...], kinds: set[str]) -> str:
    for head in heads:
        if str(head.kind).lower() in kinds:
            return _head_name(head)
    return ""


def _make_norm(nn: Any, torch: Any, hidden_dim: int, norm_kind: str) -> Any:
    key = str(norm_kind or "layer_norm").lower()
    if key in {"layer_norm", "layernorm", "ln"}:
        return nn.LayerNorm(int(hidden_dim))
    if key in {"rms_norm", "rmsnorm", "rms"}:
        return TinyRMSNorm(nn=nn, torch=torch, hidden_dim=int(hidden_dim))
    raise ValueError(f"unsupported tiny Transformer norm kind: {norm_kind}")


def _make_linear(nn: Any, torch: Any, in_features: int, out_features: int, lora: Mapping[str, Any], *, target: str) -> Any:
    cfg = dict(lora or {})
    enabled = bool(cfg.get("enabled", False))
    rank = int(cfg.get("rank", 0) or 0)
    targets = tuple(str(item) for item in cfg.get("targets", ("attention.q", "attention.v")) or ())
    if enabled and rank > 0 and (str(target) in targets or "*" in targets):
        quantization = dict(cfg.get("quantization", {}) or {})
        qlora = dict(cfg.get("qlora", {}) or {})
        quantize_base = bool(cfg.get("quantize_base", False) or qlora or quantization)
        return TinyLoRALinear(
            nn=nn,
            torch=torch,
            in_features=int(in_features),
            out_features=int(out_features),
            bias=bool(cfg.get("bias", True)),
            rank=rank,
            alpha=float(cfg.get("alpha", rank)),
            dropout=float(cfg.get("dropout", 0.0)),
            freeze_base=bool(cfg.get("freeze_base", False)),
            quantize_base=quantize_base,
            quantization_bits=int(qlora.get("bits", quantization.get("bits", cfg.get("quantization_bits", 4)))),
        )
    return nn.Linear(int(in_features), int(out_features))


def _pool_hidden(torch: Any, hidden: Any, attention_mask: Any | None, pooling: str) -> Any:
    key = str(pooling or "mean").lower()
    if key in {"cls", "first"}:
        return hidden[:, 0, :]
    if key in {"last"}:
        if attention_mask is None:
            return hidden[:, -1, :]
        lengths = torch.clamp(torch.sum(attention_mask.to(dtype=torch.long), dim=1), min=1) - 1
        return hidden[torch.arange(hidden.shape[0], device=hidden.device), lengths, :]
    if attention_mask is None:
        return torch.mean(hidden, dim=1)
    mask = attention_mask.to(dtype=hidden.dtype).unsqueeze(-1)
    denom = torch.clamp(torch.sum(mask, dim=1), min=1.0)
    return torch.sum(hidden * mask, dim=1) / denom


def _activate(torch: Any, x: Any, activation: str) -> Any:
    key = str(activation or "gelu").lower()
    if key == "relu":
        return torch.relu(x)
    if key == "gelu":
        return torch.nn.functional.gelu(x)
    if key in {"silu", "swish"}:
        return torch.nn.functional.silu(x)
    if key == "tanh":
        return torch.tanh(x)
    if key in {"identity", "linear", "none"}:
        return x
    raise ValueError(f"unsupported tiny Transformer activation: {activation}")


def _past_length(past_key_values: Any | None) -> int:
    values = tuple(past_key_values or ())
    if not values:
        return 0
    first = values[0]
    if first is None:
        return 0
    key = first[0] if isinstance(first, (tuple, list)) and first else None
    return 0 if key is None else int(key.shape[-2])


def _make_kv_cache(*, input_ids: Any, attention_mask: Any | None, past_key_values: Any, position_offset: int) -> dict[str, Any]:
    return {
        "schema": "mlblack.tiny_transformer.kv_cache.v1",
        "position_offset": int(position_offset),
        "length": int(position_offset) + int(input_ids.shape[1]),
        "batch_size": int(input_ids.shape[0]),
        "tokens": input_ids.detach(),
        "attention_mask": None if attention_mask is None else attention_mask.detach(),
        "past_key_values": tuple(past_key_values or ()),
        "num_layers": int(len(tuple(past_key_values or ()))),
    }


def _select_next_token(torch: Any, logits: Any, *, temperature: float, top_k: int | None) -> Any:
    if float(temperature) <= 0.0:
        return torch.argmax(logits, dim=-1, keepdim=True)
    scaled = logits / max(float(temperature), 1e-8)
    if top_k is not None and int(top_k) > 0 and int(top_k) < int(scaled.shape[-1]):
        values, indices = torch.topk(scaled, k=int(top_k), dim=-1)
        probs = torch.softmax(values, dim=-1)
        selected = torch.multinomial(probs, num_samples=1)
        return torch.gather(indices, dim=-1, index=selected)
    probs = torch.softmax(scaled, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def _fake_quantize_tensor(torch: Any, tensor: Any, *, bits: int = 4) -> Any:
    levels = max(2, int(2 ** int(bits)) - 1)
    min_value = torch.min(tensor)
    max_value = torch.max(tensor)
    span = torch.clamp(max_value - min_value, min=1e-8)
    scaled = torch.round((tensor - min_value) / span * float(levels))
    return (scaled / float(levels) * span) + min_value


def _apply_rope(torch: Any, q: Any, k: Any, *, base: float = 10000.0, position_offset: int = 0) -> tuple[Any, Any]:
    head_dim = int(q.shape[-1])
    rot_dim = head_dim - (head_dim % 2)
    if rot_dim <= 0:
        return q, k
    seq_len = int(q.shape[-2])
    inv_freq = 1.0 / (float(base) ** (torch.arange(0, rot_dim, 2, device=q.device, dtype=q.dtype) / float(rot_dim)))
    positions = torch.arange(int(position_offset), int(position_offset) + seq_len, device=q.device, dtype=q.dtype)
    angles = torch.einsum("t,d->td", positions, inv_freq)
    cos = torch.cos(angles).unsqueeze(0).unsqueeze(0)
    sin = torch.sin(angles).unsqueeze(0).unsqueeze(0)
    return _rotate_with_cos_sin(torch, q, cos, sin, rot_dim), _rotate_with_cos_sin(torch, k, cos, sin, rot_dim)


def _rotate_with_cos_sin(torch: Any, x: Any, cos: Any, sin: Any, rot_dim: int) -> Any:
    x_rot = x[..., :rot_dim]
    x_pass = x[..., rot_dim:]
    x1 = x_rot[..., 0::2]
    x2 = x_rot[..., 1::2]
    rotated = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2)
    if x_pass.numel() == 0:
        return rotated
    return torch.cat((rotated, x_pass), dim=-1)


def _load_flat_parameters(module: Any, values: np.ndarray) -> None:
    arr = np.asarray(values, dtype=float).reshape(-1)
    expected = int(sum(param.numel() for _, param in module.named_parameters()))
    if arr.shape[0] != expected:
        raise ValueError(f"parameter vector has {arr.shape[0]} values but tiny Transformer expects {expected}")
    offset = 0
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("tiny Transformer neural graph requires optional dependency 'torch'") from exc
    with torch.no_grad():
        for _name, param in module.named_parameters():
            size = int(param.numel())
            block = arr[offset : offset + size].reshape(tuple(param.shape))
            offset += size
            param.copy_(
                torch.as_tensor(
                    np.array(block, copy=True),
                    dtype=param.dtype,
                    device=param.device,
                )
            )


__all__ = [
    "build_tiny_transformer_module",
    "decode_tiny_transformer",
    "is_tiny_transformer_spec",
    "transformer_initial_values",
    "transformer_parameter_layout",
]
