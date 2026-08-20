"""Torch lowering for the backend-neutral ``NeuralGraphSpec.mlp`` route."""

from __future__ import annotations

from typing import Any

import numpy as np

from mlblack.models import mlp_parameter_shapes
from mlblack.representations.codecs.neural.specs import NeuralGraphSpec


MLP_BLOCK_KINDS = {"mlp", "mlp_block", "feed_forward"}


def is_mlp_spec(spec: NeuralGraphSpec) -> bool:
    blocks = spec.block_specs()
    return len(blocks) == 1 and str(blocks[0].kind).lower() in MLP_BLOCK_KINDS


def mlp_parameter_layout(
    spec: NeuralGraphSpec,
) -> tuple[tuple[tuple[int, ...], ...], tuple[str, ...]]:
    input_dim, hidden_layers, output_dim, _activation, _dropout = _mlp_parts(spec)
    shapes = mlp_parameter_shapes(input_dim, hidden_layers, output_dim)
    names: list[str] = []
    for index in range(len(shapes) // 2):
        names.extend(
            (
                f"mlp.layers.{index}.weight",
                f"mlp.layers.{index}.bias",
            )
        )
    return tuple(shapes), tuple(names)


def mlp_initial_values(
    spec: NeuralGraphSpec,
    *,
    random_seed: int = 42,
) -> np.ndarray:
    module = build_mlp_module(spec, random_seed=random_seed)
    arrays = [
        parameter.detach().cpu().numpy().reshape(-1)
        for _name, parameter in module.named_parameters()
    ]
    return np.concatenate(arrays).astype(float) if arrays else np.zeros(0, dtype=float)


def decode_mlp(
    values: np.ndarray,
    spec: NeuralGraphSpec,
    *,
    random_seed: int = 42,
) -> Any:
    module = build_mlp_module(spec, random_seed=random_seed)
    _load_flat_parameters(module, np.asarray(values, dtype=float).reshape(-1))
    return module


def build_mlp_module(spec: NeuralGraphSpec, *, random_seed: int = 42) -> Any:
    try:
        import torch
        from torch import nn
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("MLP neural lowering requires optional dependency 'torch'") from exc

    torch.manual_seed(int(random_seed))
    input_dim, hidden_layers, output_dim, activation, dropout = _mlp_parts(spec)

    class _CanonicalLinear(nn.Module):
        """Linear layer whose public weight layout is canonical ``[in, out]``."""

        def __init__(self, left: int, right: int) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.empty(left, right))
            self.bias = nn.Parameter(torch.empty(right))
            nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5.0))
            bound = 1.0 / np.sqrt(max(1, left))
            nn.init.uniform_(self.bias, -bound, bound)

        def forward(self, values: Any) -> Any:
            return values @ self.weight + self.bias

    class _MLPBackbone(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            dims = (input_dim, *hidden_layers, output_dim)
            self.layers = nn.ModuleList(
                _CanonicalLinear(left, right)
                for left, right in zip(dims[:-1], dims[1:])
            )

        def forward(self, values: Any, *, training: bool) -> tuple[Any, Any]:
            out = values
            hidden = values
            for index, layer in enumerate(self.layers):
                out = layer(out)
                if index < len(self.layers) - 1:
                    out = _activate(torch, out, activation)
                    if dropout > 0.0:
                        out = torch.nn.functional.dropout(
                            out,
                            p=min(dropout, 0.95),
                            training=training,
                        )
                    hidden = out
            return out, hidden

    class _MLPGraphModule(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.route = "mlp"
            self.graph_spec = spec.as_dict()
            self.mlp = _MLPBackbone()

        def forward(self, values: Any, *, return_audit: bool = False) -> dict[str, Any]:
            if values.ndim != 2:
                raise ValueError("MLP input must have shape [batch, features]")
            output, hidden = self.mlp(values, training=bool(self.training))
            return {
                "hidden_states": hidden,
                "head_outputs": {"point": output},
                "logits": output,
                "audit": (
                    {
                        "output_mean": torch.mean(output).detach(),
                        "output_std": torch.std(output).detach(),
                    }
                    if return_audit
                    else {}
                ),
            }

        def predict(self, values: np.ndarray) -> np.ndarray:
            self.eval()
            with torch.no_grad():
                tensor = torch.as_tensor(values, dtype=self.mlp.layers[0].weight.dtype)
                output = self.forward(tensor)["head_outputs"]["point"]
            return output.detach().cpu().numpy().reshape(len(values), -1)[:, 0]

        def describe(self) -> dict[str, Any]:
            return {
                "kind": "mlp",
                "input_dim": input_dim,
                "hidden_layers": hidden_layers,
                "output_dim": output_dim,
                "activation": activation,
                "dropout": dropout,
            }

    return _MLPGraphModule()


def _mlp_parts(
    spec: NeuralGraphSpec,
) -> tuple[int, tuple[int, ...], int, str, float]:
    input_dim = int(dict(spec.input).get("input_dim", 0))
    if input_dim <= 0:
        raise ValueError("MLP spec requires input.input_dim")
    blocks = spec.block_specs()
    if len(blocks) != 1 or str(blocks[0].kind).lower() not in MLP_BLOCK_KINDS:
        raise ValueError("MLP spec requires exactly one mlp block")
    params = dict(blocks[0].params)
    hidden_layers = tuple(int(value) for value in params.get("hidden_layers", (64, 32)))
    if any(value <= 0 for value in hidden_layers):
        raise ValueError("MLP hidden layer sizes must be positive")
    heads = spec.head_specs()
    output_dim = 1 if not heads else int(dict(heads[0].params).get("output_dim", 1))
    if output_dim <= 0:
        raise ValueError("MLP output dimension must be positive")
    return (
        input_dim,
        hidden_layers,
        output_dim,
        str(params.get("activation", "relu")),
        float(params.get("dropout", 0.0) or 0.0),
    )


def _activate(torch: Any, values: Any, activation: str) -> Any:
    key = str(activation or "relu").strip().lower()
    if key == "relu":
        return torch.relu(values)
    if key == "tanh":
        return torch.tanh(values)
    if key in {"sigmoid", "logistic"}:
        return torch.sigmoid(values)
    if key == "gelu":
        return torch.nn.functional.gelu(values)
    if key in {"silu", "swish"}:
        return torch.nn.functional.silu(values)
    if key in {"identity", "linear", "none"}:
        return values
    raise ValueError(f"unsupported MLP activation: {activation}")


def _load_flat_parameters(module: Any, values: np.ndarray) -> None:
    expected = int(sum(parameter.numel() for _, parameter in module.named_parameters()))
    if values.shape[0] != expected:
        raise ValueError(
            f"parameter vector has {values.shape[0]} values but MLP expects {expected}"
        )
    try:
        import torch
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("MLP neural lowering requires optional dependency 'torch'") from exc
    offset = 0
    with torch.no_grad():
        for _name, parameter in module.named_parameters():
            size = int(parameter.numel())
            block = values[offset : offset + size].reshape(tuple(parameter.shape))
            offset += size
            parameter.copy_(
                torch.as_tensor(
                    np.array(block, copy=True),
                    dtype=parameter.dtype,
                    device=parameter.device,
                )
            )


__all__ = [
    "build_mlp_module",
    "decode_mlp",
    "is_mlp_spec",
    "mlp_initial_values",
    "mlp_parameter_layout",
]
