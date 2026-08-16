from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from core.common.contracts import ProcessedDataset


@dataclass(frozen=True)
class TrainDataBundle:
    train: ProcessedDataset
    test: ProcessedDataset | None = None
    valid: ProcessedDataset | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

