from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from mlblack.representations.codecs.neural.specs import NeuralGraphSpec, NeuralHeadSpec


GNN_BLOCK_KINDS = {"gnn", "gnn_block", "gcn", "tiny_gnn"}


def is_tiny_gnn_spec(spec: NeuralGraphSpec) -> bool:
    blocks = spec.block_specs()
    return bool(blocks) and all(str(block.kind).lower() in GNN_BLOCK_KINDS for block in blocks)


def gnn_parameter_layout(spec: NeuralGraphSpec) -> tuple[tuple[tuple[int, ...], ...], tuple[str, ...]]:
    module = build_tiny_gnn_module(spec, random_seed=0)
    shapes = tuple(tuple(int(v) for v in param.detach().cpu().shape) for _, param in module.named_parameters())
    names = tuple(str(name) for name, _ in module.named_parameters())
    return shapes, names


def gnn_initial_values(spec: NeuralGraphSpec, *, random_seed: int = 42) -> np.ndarray:
    module = build_tiny_gnn_module(spec, random_seed=random_seed)
    arrays = [param.detach().cpu().numpy().reshape(-1) for _, param in module.named_parameters()]
    return np.concatenate(arrays).astype(float) if arrays else np.zeros(0, dtype=float)


def decode_tiny_gnn(values: np.ndarray, spec: NeuralGraphSpec, *, random_seed: int = 42) -> Any:
    module = build_tiny_gnn_module(spec, random_seed=random_seed)
    _load_flat_parameters(module, np.asarray(values, dtype=float).reshape(-1))
    return module


def build_tiny_gnn_module(spec: NeuralGraphSpec, *, random_seed: int = 42) -> Any:
    try:
        import torch
        from torch import nn
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("tiny GNN neural graph requires optional dependency 'torch'") from exc
    torch.manual_seed(int(random_seed))
    input_cfg = _input_config(spec)
    input_cfg["graph_spec"] = spec.as_dict()
    block_cfg = _block_config(spec)
    return TinyGNNGraphModule(nn=nn, torch=torch, input_cfg=input_cfg, block_cfg=block_cfg, head_specs=spec.head_specs())


class TinyGNNGraphModule:
    def __new__(cls, *, nn: Any, torch: Any, input_cfg: Mapping[str, Any], block_cfg: Mapping[str, Any], head_specs: tuple[NeuralHeadSpec, ...]) -> Any:
        class _TinyGNNGraphModule(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.route = "tiny_gnn"
                self.graph_spec = dict(input_cfg.get("graph_spec", {}) or {})
                self.node_feature_dim = int(input_cfg["node_feature_dim"])
                self.num_nodes = int(input_cfg["num_nodes"])
                self.hidden_dim = int(block_cfg["hidden_dim"])
                self.pooling = str(block_cfg.get("pooling", "mean")).lower()
                dims = [self.node_feature_dim] + [self.hidden_dim] * int(block_cfg["num_layers"])
                self.layers = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)])
                self.dropout = nn.Dropout(float(block_cfg.get("dropout", 0.0)))
                self.activation_name = str(block_cfg.get("activation", "relu")).lower()
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
                        raise ValueError(f"unsupported tiny GNN head kind: {head.kind}")

            def forward(self, node_features: Any, adjacency: Any, *, return_audit: bool = False) -> dict[str, Any]:
                if node_features.ndim != 3:
                    raise ValueError("node_features must have shape [batch, nodes, features]")
                if adjacency.ndim != 3:
                    raise ValueError("adjacency must have shape [batch, nodes, nodes]")
                x = node_features
                norm_adj = _normalize_adjacency(torch, adjacency)
                activations = []
                for layer in self.layers:
                    x = torch.matmul(norm_adj, x)
                    x = layer(x)
                    x = _activate(torch, x, self.activation_name)
                    x = self.dropout(x)
                    if return_audit:
                        activations.append(x.detach())
                pooled = _pool_nodes(torch, x, self.pooling)
                head_outputs: dict[str, Any] = {}
                for head in self.head_specs:
                    key = _head_name(head)
                    head_outputs[key] = self.heads[key](pooled)
                first_key = _head_name(self.head_specs[0]) if self.head_specs else ""
                embedding_key = _first_head_key(self.head_specs, {"embedding", "embedding_head", "contrastive", "retrieval"})
                return {
                    "hidden_states": pooled,
                    "node_states": x,
                    "head_outputs": head_outputs,
                    "logits": head_outputs.get(first_key),
                    "embeddings": head_outputs.get(embedding_key) if embedding_key else None,
                    "audit": {"layer_activations": activations} if return_audit else {},
                }

            def describe(self) -> dict[str, Any]:
                return {
                    "kind": "tiny_gnn",
                    "node_feature_dim": int(self.node_feature_dim),
                    "num_nodes": int(self.num_nodes),
                    "hidden_dim": int(self.hidden_dim),
                    "num_layers": int(len(self.layers)),
                    "heads": tuple(_head_name(head) for head in self.head_specs),
                }

        return _TinyGNNGraphModule()


def _input_config(spec: NeuralGraphSpec) -> dict[str, Any]:
    cfg = dict(spec.input)
    if int(cfg.get("node_feature_dim", 0)) <= 0:
        raise ValueError("tiny GNN spec requires input.node_feature_dim")
    if int(cfg.get("num_nodes", 0)) <= 0:
        raise ValueError("tiny GNN spec requires input.num_nodes")
    return cfg


def _block_config(spec: NeuralGraphSpec) -> dict[str, Any]:
    blocks = spec.block_specs()
    if len(blocks) != 1:
        raise ValueError("tiny GNN expects exactly one gnn block spec")
    block = blocks[0]
    params = dict(block.params)
    return {
        "num_layers": int(block.repeat),
        "hidden_dim": int(params.get("hidden_dim", 16)),
        "activation": str(params.get("activation", "relu")),
        "dropout": float(params.get("dropout", 0.0)),
        "pooling": str(params.get("pooling", "mean")),
    }


def _normalize_adjacency(torch: Any, adjacency: Any) -> Any:
    eye = torch.eye(adjacency.shape[-1], device=adjacency.device, dtype=adjacency.dtype).unsqueeze(0)
    adj = adjacency + eye
    degree = torch.clamp(torch.sum(adj, dim=-1, keepdim=True), min=1.0)
    return adj / degree


def _pool_nodes(torch: Any, node_states: Any, pooling: str) -> Any:
    key = str(pooling or "mean").lower()
    if key == "sum":
        return torch.sum(node_states, dim=1)
    if key == "max":
        return torch.max(node_states, dim=1).values
    return torch.mean(node_states, dim=1)


def _activate(torch: Any, x: Any, activation: str) -> Any:
    key = str(activation or "relu").lower()
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
    raise ValueError(f"unsupported tiny GNN activation: {activation}")


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
        raise ValueError(f"parameter vector has {arr.shape[0]} values but tiny GNN expects {expected}")
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("tiny GNN neural graph requires optional dependency 'torch'") from exc
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


__all__ = ["build_tiny_gnn_module", "decode_tiny_gnn", "gnn_initial_values", "gnn_parameter_layout", "is_tiny_gnn_spec"]
