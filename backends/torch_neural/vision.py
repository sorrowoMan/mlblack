from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from mlblack.representations.codecs.neural.specs import NeuralGraphSpec, NeuralHeadSpec


CNN_BLOCK_KINDS = {"cnn", "cnn_block", "convnet", "tiny_cnn"}


def is_tiny_cnn_spec(spec: NeuralGraphSpec) -> bool:
    blocks = spec.block_specs()
    return bool(blocks) and all(str(block.kind).lower() in CNN_BLOCK_KINDS for block in blocks)


def cnn_parameter_layout(spec: NeuralGraphSpec) -> tuple[tuple[tuple[int, ...], ...], tuple[str, ...]]:
    module = build_tiny_cnn_module(spec, random_seed=0)
    shapes = tuple(tuple(int(v) for v in param.detach().cpu().shape) for _, param in module.named_parameters())
    names = tuple(str(name) for name, _ in module.named_parameters())
    return shapes, names


def cnn_initial_values(spec: NeuralGraphSpec, *, random_seed: int = 42) -> np.ndarray:
    module = build_tiny_cnn_module(spec, random_seed=random_seed)
    arrays = [param.detach().cpu().numpy().reshape(-1) for _, param in module.named_parameters()]
    return np.concatenate(arrays).astype(float) if arrays else np.zeros(0, dtype=float)


def decode_tiny_cnn(values: np.ndarray, spec: NeuralGraphSpec, *, random_seed: int = 42) -> Any:
    module = build_tiny_cnn_module(spec, random_seed=random_seed)
    _load_flat_parameters(module, np.asarray(values, dtype=float).reshape(-1))
    return module


def build_tiny_cnn_module(spec: NeuralGraphSpec, *, random_seed: int = 42) -> Any:
    try:
        import torch
        from torch import nn
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("tiny CNN neural graph requires optional dependency 'torch'") from exc
    torch.manual_seed(int(random_seed))
    input_cfg = _input_config(spec)
    input_cfg["graph_spec"] = spec.as_dict()
    block_cfg = _block_config(spec)
    return TinyCNNGraphModule(nn=nn, torch=torch, input_cfg=input_cfg, block_cfg=block_cfg, head_specs=spec.head_specs())


class TinyCNNGraphModule:
    def __new__(cls, *, nn: Any, torch: Any, input_cfg: Mapping[str, Any], block_cfg: Mapping[str, Any], head_specs: tuple[NeuralHeadSpec, ...]) -> Any:
        class _TinyCNNGraphModule(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.route = "tiny_cnn"
                self.graph_spec = dict(input_cfg.get("graph_spec", {}) or {})
                self.channels = int(input_cfg["channels"])
                self.height = int(input_cfg["height"])
                self.width = int(input_cfg["width"])
                conv_channels = tuple(int(v) for v in block_cfg["conv_channels"])
                layers: list[Any] = []
                in_ch = self.channels
                for out_ch in conv_channels:
                    layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=int(block_cfg["kernel_size"]), padding=int(block_cfg["kernel_size"]) // 2))
                    layers.append(_activation_module(nn, str(block_cfg.get("activation", "relu"))))
                    if float(block_cfg.get("dropout", 0.0)) > 0.0:
                        layers.append(nn.Dropout2d(float(block_cfg["dropout"])))
                    in_ch = out_ch
                self.backbone = nn.Sequential(*layers)
                self.pool = nn.AdaptiveAvgPool2d((1, 1))
                self.hidden_dim = int(conv_channels[-1] if conv_channels else self.channels)
                self.heads = nn.ModuleDict()
                self.head_specs = tuple(head_specs)
                for head in self.head_specs:
                    key = _head_name(head)
                    kind = str(head.kind).lower()
                    params = dict(head.params)
                    if kind in {"classification", "classifier", "class"}:
                        self.heads[key] = nn.Linear(self.hidden_dim, int(params.get("num_classes", 2)))
                    elif kind in {"embedding", "embedding_head", "contrastive", "retrieval"}:
                        self.heads[key] = nn.Linear(self.hidden_dim, int(params.get("output_dim", self.hidden_dim)))
                    else:
                        raise ValueError(f"unsupported tiny CNN head kind: {head.kind}")

            def forward(self, images: Any, *, return_audit: bool = False) -> dict[str, Any]:
                if images.ndim != 4:
                    raise ValueError("images must have shape [batch, channels, height, width]")
                features = self.backbone(images)
                pooled = self.pool(features).flatten(1)
                head_outputs: dict[str, Any] = {}
                for head in self.head_specs:
                    key = _head_name(head)
                    head_outputs[key] = self.heads[key](pooled)
                first_key = _head_name(self.head_specs[0]) if self.head_specs else ""
                embedding_key = _first_head_key(self.head_specs, {"embedding", "embedding_head", "contrastive", "retrieval"})
                return {
                    "hidden_states": pooled,
                    "feature_maps": features,
                    "head_outputs": head_outputs,
                    "logits": head_outputs.get(first_key),
                    "embeddings": head_outputs.get(embedding_key) if embedding_key else None,
                    "audit": {"feature_mean": torch.mean(features).detach(), "feature_std": torch.std(features).detach()} if return_audit else {},
                }

            def describe(self) -> dict[str, Any]:
                return {
                    "kind": "tiny_cnn",
                    "channels": int(self.channels),
                    "height": int(self.height),
                    "width": int(self.width),
                    "hidden_dim": int(self.hidden_dim),
                    "heads": tuple(_head_name(head) for head in self.head_specs),
                }

        return _TinyCNNGraphModule()


def _input_config(spec: NeuralGraphSpec) -> dict[str, Any]:
    cfg = dict(spec.input)
    for key in ("channels", "height", "width"):
        if int(cfg.get(key, 0)) <= 0:
            raise ValueError(f"tiny CNN spec requires input.{key}")
    return cfg


def _block_config(spec: NeuralGraphSpec) -> dict[str, Any]:
    blocks = spec.block_specs()
    if len(blocks) != 1:
        raise ValueError("tiny CNN expects exactly one cnn block spec")
    params = dict(blocks[0].params)
    return {
        "conv_channels": tuple(int(v) for v in params.get("conv_channels", (8, 16))),
        "kernel_size": int(params.get("kernel_size", 3)),
        "activation": str(params.get("activation", "relu")),
        "dropout": float(params.get("dropout", 0.0)),
    }


def _head_name(head: NeuralHeadSpec) -> str:
    return str(head.name or head.kind or "head")


def _first_head_key(heads: tuple[NeuralHeadSpec, ...], kinds: set[str]) -> str:
    for head in heads:
        if str(head.kind).lower() in kinds:
            return _head_name(head)
    return ""


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
    raise ValueError(f"unsupported tiny CNN activation: {activation}")


def _load_flat_parameters(module: Any, values: np.ndarray) -> None:
    arr = np.asarray(values, dtype=float).reshape(-1)
    expected = int(sum(param.numel() for _, param in module.named_parameters()))
    if arr.shape[0] != expected:
        raise ValueError(f"parameter vector has {arr.shape[0]} values but tiny CNN expects {expected}")
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("tiny CNN neural graph requires optional dependency 'torch'") from exc
    offset = 0
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


__all__ = ["build_tiny_cnn_module", "cnn_initial_values", "cnn_parameter_layout", "decode_tiny_cnn", "is_tiny_cnn_spec"]
