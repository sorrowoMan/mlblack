from __future__ import annotations

from typing import Sequence


try:
    import torch
    from torch import nn
except Exception as exc:  # pragma: no cover - environment guard
    raise ImportError(
        "PyTorch is required for torch-based trainers. Install torch before using mlp_torch."
    ) from exc


def _activation(name: str) -> nn.Module:
    key = str(name or "relu").strip().lower()
    if key == "relu":
        return nn.ReLU()
    if key == "gelu":
        return nn.GELU()
    if key == "tanh":
        return nn.Tanh()
    if key == "silu":
        return nn.SiLU()
    raise ValueError(f"Unsupported activation: {name}")


class TorchMLPRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        hidden_dims: Sequence[int] = (128, 64),
        activation: str = "relu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if int(input_dim) <= 0:
            raise ValueError("input_dim must be positive")
        if int(output_dim) <= 0:
            raise ValueError("output_dim must be positive")

        dims = [int(input_dim)] + [int(h) for h in hidden_dims if int(h) > 0] + [int(output_dim)]
        if len(dims) < 2:
            raise ValueError("Invalid MLP dimensions")

        layers: list[nn.Module] = []
        act_name = str(activation)
        p = float(dropout)

        for i in range(len(dims) - 1):
            in_d = dims[i]
            out_d = dims[i + 1]
            layers.append(nn.Linear(in_d, out_d))
            if i < len(dims) - 2:
                layers.append(_activation(act_name))
                if p > 0.0:
                    layers.append(nn.Dropout(p=p))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
