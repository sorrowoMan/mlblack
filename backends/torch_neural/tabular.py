from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from mlblack.representations.codecs.neural.specs import NeuralGraphSpec, NeuralHeadSpec


TABNET_BLOCK_KINDS = {"tabnet", "tabnet_block", "tabular_tabnet"}


def is_tabular_tabnet_spec(spec: NeuralGraphSpec) -> bool:
    blocks = spec.block_specs()
    if not blocks:
        return False
    kinds = {str(block.kind).lower() for block in blocks}
    return bool(kinds.issubset(TABNET_BLOCK_KINDS)
                or any(str(spec.metadata.get("route", "")).lower() == "tabular_tabnet" for _ in (1,)))


def tabnet_parameter_layout(spec: NeuralGraphSpec) -> tuple[tuple[tuple[int, ...], ...], tuple[str, ...]]:
    module = build_tabnet_module(spec, random_seed=0)
    shapes = tuple(tuple(int(v) for v in param.detach().cpu().shape) for _, param in module.named_parameters())
    names = tuple(str(name) for name, _ in module.named_parameters())
    return shapes, names


def tabnet_initial_values(spec: NeuralGraphSpec, *, random_seed: int = 42) -> np.ndarray:
    module = build_tabnet_module(spec, random_seed=random_seed)
    arrays = [param.detach().cpu().numpy().reshape(-1) for _, param in module.named_parameters()]
    return np.concatenate(arrays).astype(float) if arrays else np.zeros(0, dtype=float)


def decode_tabnet(values: np.ndarray, spec: NeuralGraphSpec, *, random_seed: int = 42) -> Any:
    module = build_tabnet_module(spec, random_seed=random_seed)
    _load_flat_parameters(module, np.asarray(values, dtype=float).reshape(-1))
    return module


def build_tabnet_module(spec: NeuralGraphSpec, *, random_seed: int = 42) -> Any:
    try:
        import torch
        from torch import nn
    except Exception as exc:
        raise RuntimeError("TabNet neural graph requires optional dependency 'torch'") from exc
    torch.manual_seed(int(random_seed))
    input_cfg = _input_config(spec)
    input_cfg["graph_spec"] = spec.as_dict()
    block_cfg = _block_config(spec)
    route = str(spec.metadata.get("route", "tabular_tabnet")).strip().lower()
    return TabNetModule(
        nn=nn, torch=torch, route=route, input_cfg=input_cfg,
        block_cfg=block_cfg, head_specs=spec.head_specs(),
    )


class TabNetModule:
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
        class _TabNetModule(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.route = str(route)
                self.graph_spec = dict(input_cfg.get("graph_spec", {}) or {})
                self.input_dim = int(input_cfg["input_dim"])
                self.hidden_dim = int(block_cfg["hidden_dim"])
                self.n_steps = int(block_cfg["n_steps"])
                self.relaxation = float(block_cfg.get("relaxation_factor", 1.5))
                self.epsilon = 1e-5
                self.head_specs = tuple(head_specs)

                # Initial batch norm
                self.initial_bn = nn.BatchNorm1d(self.input_dim)
                # Project raw features to hidden dim before feature transformer
                self.input_proj = nn.Linear(self.input_dim, self.hidden_dim)

                # Shared feature transformer (used across steps)
                self.shared_ft = nn.ModuleList([
                    _FeatureTransformerBlock(nn=nn, input_dim=self.hidden_dim, output_dim=self.hidden_dim,
                                             ghost_bn=bool(block_cfg.get("ghost_bn", True)),
                                             dropout=float(block_cfg.get("dropout", 0.0)))
                ])

                # Per-step feature transformers
                self.step_ft = nn.ModuleList([
                    _FeatureTransformerBlock(nn=nn, input_dim=self.hidden_dim, output_dim=self.hidden_dim,
                                             ghost_bn=bool(block_cfg.get("ghost_bn", True)),
                                             dropout=float(block_cfg.get("dropout", 0.0)))
                    for _ in range(self.n_steps)
                ])

                # Attentive transformers (one per step, produces mask)
                self.attentive_transformers = nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(self.hidden_dim, self.input_dim),
                        nn.BatchNorm1d(self.input_dim),
                    )
                    for _ in range(self.n_steps)
                ])

                self.heads = nn.ModuleDict()
                for head in self.head_specs:
                    key = str(head.name or head.kind or "head")
                    params = dict(head.params)
                    kind = str(head.kind).lower()
                    if kind in {"classification", "classifier", "class"}:
                        out_dim = int(params.get("num_classes", 2))
                        self.heads[key] = nn.Linear(self.hidden_dim, out_dim)
                    elif kind in {"forecast", "point", "regression"}:
                        out_dim = int(params.get("output_dim", 1))
                        self.heads[key] = nn.Linear(self.hidden_dim, out_dim)
                    elif kind in {"embedding", "embedding_head"}:
                        self.heads[key] = nn.Linear(self.hidden_dim, int(params.get("output_dim", self.hidden_dim)))
                    else:
                        self.heads[key] = nn.Linear(self.hidden_dim, self.hidden_dim)

            def forward(self, x: Any, *, return_audit: bool = False) -> dict[str, Any]:
                x_flat = x.float() if hasattr(x, "float") else torch.as_tensor(x, dtype=torch.float32)
                if x_flat.ndim != 2:
                    raise ValueError("TabNet expects tabular input [batch, features]")
                x_bn = self.initial_bn(x_flat)
                h_proj = self.input_proj(x_bn)
                batch_size = x_bn.shape[0]
                prior = torch.ones(batch_size, self.input_dim, device=x_bn.device)
                step_outputs: list[Any] = []
                masks: list[Any] = []

                for step_idx in range(self.n_steps):
                    # Feature transformer (shared + step-specific)
                    h = h_proj
                    for shared_block in self.shared_ft:
                        h = shared_block(h)
                    h = self.step_ft[step_idx](h)

                    # Attentive transformer → mask
                    mask_logits = self.attentive_transformers[step_idx](h)
                    mask = _sparsemax(mask_logits * prior, dim=-1)
                    masks.append(mask)
                    prior = prior * (self.relaxation - mask)

                    # Apply mask to input features and re-project for next step
                    x_bn = x_bn * mask
                    h_proj = self.input_proj(x_bn)
                    step_outputs.append(h)

                # Aggregate step outputs (sum)
                aggregated = torch.stack(step_outputs, dim=0).sum(dim=0)

                head_outputs: dict[str, Any] = {}
                for head in self.head_specs:
                    key = str(head.name or head.kind or "head")
                    head_outputs[key] = self.heads[key](aggregated)
                first_key = str(self.head_specs[0].name or self.head_specs[0].kind or "head") if self.head_specs else ""
                return {
                    "hidden_states": aggregated,
                    "head_outputs": head_outputs,
                    "logits": head_outputs.get(first_key),
                    "forecast": head_outputs.get(first_key),
                    "mask_stack": torch.stack(masks, dim=1) if masks else None,
                    "audit": {"route": self.route, "hidden_mean": torch.mean(aggregated).detach(),
                              "n_steps": self.n_steps} if return_audit else {},
                }

            def describe(self) -> dict[str, Any]:
                return {
                    "kind": self.route,
                    "input_dim": int(self.input_dim),
                    "hidden_dim": int(self.hidden_dim),
                    "n_steps": int(self.n_steps),
                    "heads": tuple(str(h.name or h.kind or "head") for h in self.head_specs),
                }

        return _TabNetModule()


class _FeatureTransformerBlock:
    """GLU block with ghost batch norm for TabNet feature transformer."""

    def __new__(
        cls,
        *,
        nn: Any,
        input_dim: int,
        output_dim: int,
        ghost_bn: bool = True,
        dropout: float = 0.0,
    ) -> Any:
        class _Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(int(input_dim), int(output_dim) * 2)
                self.bn = nn.BatchNorm1d(int(output_dim) * 2, momentum=0.01) if ghost_bn else nn.Identity()
                self.drop = nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity()
                self.output_dim = int(output_dim)

            def forward(self, x):
                import torch as _torch
                h = self.fc(x)
                h = self.bn(h)
                h = self.drop(h)
                return h[:, :self.output_dim] * _torch.sigmoid(h[:, self.output_dim:])

        return _Block()


def _sparsemax(logits: Any, dim: int = -1) -> Any:
    """Sparsemax activation: Euclidean projection onto the probability simplex."""
    import torch

    z = logits - logits.max(dim=dim, keepdim=True).values
    z_sorted, _ = torch.sort(z, dim=dim, descending=True)
    z_cumsum = torch.cumsum(z_sorted, dim=dim)
    k = torch.arange(1, z.shape[dim] + 1, device=z.device, dtype=z.dtype).unsqueeze(0)
    condition = 1.0 + k * z_sorted > z_cumsum
    k_max = condition.long().sum(dim=dim, keepdim=True).float()
    tau = (z_cumsum.gather(dim, (k_max - 1).long().clamp(min=0)) - 1.0) / k_max.clamp(min=1.0)
    return torch.clamp(z - tau, min=0.0)


def _input_config(spec: NeuralGraphSpec) -> dict[str, Any]:
    cfg = dict(spec.input)
    if int(cfg.get("input_dim", 0)) <= 0:
        raise ValueError("TabNet spec requires input.input_dim")
    return cfg


def _block_config(spec: NeuralGraphSpec) -> dict[str, Any]:
    blocks = spec.block_specs()
    if len(blocks) != 1:
        raise ValueError("TabNet neural graph expects exactly one tabnet block spec")
    block = blocks[0]
    params = dict(block.params)
    return {
        "hidden_dim": int(params.get("hidden_dim", 64)),
        "n_steps": int(params.get("n_steps", 4)),
        "relaxation_factor": float(params.get("relaxation_factor", 1.5)),
        "ghost_bn": bool(params.get("ghost_bn", True)),
        "dropout": float(params.get("dropout", 0.0)),
    }


def _load_flat_parameters(module: Any, values: np.ndarray) -> None:
    arr = np.asarray(values, dtype=float).reshape(-1)
    expected = int(sum(param.numel() for _, param in module.named_parameters()))
    if arr.shape[0] != expected:
        raise ValueError(f"parameter vector has {arr.shape[0]} values but TabNet expects {expected}")
    try:
        import torch
    except Exception as exc:
        raise RuntimeError("TabNet requires optional dependency 'torch'") from exc
    offset = 0
    with torch.no_grad():
        for _name, param in module.named_parameters():
            size = int(param.numel())
            block_data = arr[offset : offset + size].reshape(tuple(param.shape))
            offset += size
            param.copy_(torch.as_tensor(block_data, dtype=param.dtype, device=param.device))


__all__ = [
    "build_tabnet_module",
    "decode_tabnet",
    "is_tabular_tabnet_spec",
    "tabnet_initial_values",
    "tabnet_parameter_layout",
]
