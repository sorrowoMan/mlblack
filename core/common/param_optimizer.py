from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

try:
    import torch
except Exception:  # pragma: no cover - optional dependency at import time
    torch = None  # type: ignore[assignment]


@dataclass(frozen=True)
class OptimizerSpec:
    key: str = "adamw"
    params: Mapping[str, Any] = field(default_factory=dict)


def _ensure_torch() -> None:
    if torch is None:
        raise RuntimeError("PyTorch is required for optimizer creation")


def create_torch_optimizer(
    parameters: Iterable[Any],
    *,
    spec: OptimizerSpec,
    lr: float,
    weight_decay: float,
):
    _ensure_torch()
    key = str(spec.key or "adamw").strip().lower()
    kwargs = dict(spec.params or {})
    kwargs.setdefault("lr", float(lr))
    kwargs.setdefault("weight_decay", float(weight_decay))

    if key == "adamw":
        return torch.optim.AdamW(parameters, **kwargs)
    if key == "adam":
        return torch.optim.Adam(parameters, **kwargs)
    if key == "sgd":
        kwargs.setdefault("momentum", 0.0)
        return torch.optim.SGD(parameters, **kwargs)
    if key == "rmsprop":
        return torch.optim.RMSprop(parameters, **kwargs)

    raise ValueError(f"Unsupported optimizer: {spec.key}")
