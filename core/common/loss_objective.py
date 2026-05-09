from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

try:
    import torch
except Exception:  # pragma: no cover - optional dependency at import time
    torch = None  # type: ignore[assignment]


class TrainingObjective(Protocol):
    """Loss/objective abstraction for supervised training."""

    name: str

    def loss(self, pred: Any, target: Any, *, sample_weight: Any | None = None) -> Any:
        ...


def _ensure_torch() -> None:
    if torch is None:
        raise RuntimeError("PyTorch is required for torch objectives")


def _weighted_reduce(per_sample_loss, sample_weight):
    if sample_weight is None:
        return torch.mean(per_sample_loss)
    w = sample_weight.reshape(-1)
    denom = torch.clamp(torch.sum(w), min=1e-8)
    return torch.sum(per_sample_loss * w) / denom


@dataclass(frozen=True)
class MSEObjective:
    name: str = "mse"

    def loss(self, pred, target, *, sample_weight=None):
        _ensure_torch()
        sq = (pred - target) ** 2
        per_sample = torch.mean(sq, dim=1)
        return _weighted_reduce(per_sample, sample_weight)


@dataclass(frozen=True)
class PinballObjective:
    quantile: float
    name: str = "pinball"

    def __post_init__(self) -> None:
        q = float(self.quantile)
        if not (0.0 < q < 1.0):
            raise ValueError("pinball quantile must be in (0, 1)")

    def loss(self, pred, target, *, sample_weight=None):
        _ensure_torch()
        q = float(self.quantile)
        err = target - pred
        loss_elem = torch.maximum(q * err, (q - 1.0) * err)
        per_sample = torch.mean(loss_elem, dim=1)
        return _weighted_reduce(per_sample, sample_weight)


def create_regression_objective(name: str = "mse") -> TrainingObjective:
    key = str(name or "mse").strip().lower()
    if key in {"mse", "l2", "mean_squared_error"}:
        return MSEObjective()
    raise ValueError(f"Unsupported regression objective: {name}")


def create_quantile_objective(name: str = "pinball", *, quantile: float) -> TrainingObjective:
    key = str(name or "pinball").strip().lower()
    if key in {"pinball", "quantile"}:
        return PinballObjective(quantile=float(quantile))
    raise ValueError(f"Unsupported quantile objective: {name}")
