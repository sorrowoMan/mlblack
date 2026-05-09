from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence

try:
    from torch.utils.data import DataLoader, TensorDataset
except Exception:  # pragma: no cover - optional dependency at import time
    DataLoader = None  # type: ignore[assignment]
    TensorDataset = None  # type: ignore[assignment]


class BatchStream(Protocol):
    """Data-loader abstraction that yields training mini-batches."""

    def __iter__(self):
        ...

    def __len__(self) -> int:
        ...


@dataclass(frozen=True)
class BatchStreamSpec:
    batch_size: int = 64
    shuffle: bool = True
    drop_last: bool = False
    num_workers: int = 0
    pin_memory: bool = False


def create_torch_batch_stream(
    tensors: Sequence[Any],
    *,
    spec: BatchStreamSpec,
):
    if TensorDataset is None or DataLoader is None:
        raise RuntimeError("PyTorch DataLoader is required for batch stream creation")

    dataset = TensorDataset(*tensors)
    return DataLoader(
        dataset,
        batch_size=int(max(1, spec.batch_size)),
        shuffle=bool(spec.shuffle),
        drop_last=bool(spec.drop_last),
        num_workers=int(max(0, spec.num_workers)),
        pin_memory=bool(spec.pin_memory),
    )
