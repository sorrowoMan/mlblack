from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from core.common.contracts import ProcessedDataset, SampleDataset


TrainingData = ProcessedDataset | SampleDataset


@dataclass(frozen=True)
class TrainTask:
    data: TrainingData
    schema: Mapping[str, Any] | None = None
    objective: Mapping[str, Any] | str | None = None
    sample_weight: np.ndarray | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    task_id: str = "train_task"

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", dict(self.metadata))
        if self.sample_weight is not None:
            object.__setattr__(self, "sample_weight", np.asarray(self.sample_weight, dtype=float).reshape(-1))

    @classmethod
    def from_data(
        cls,
        data: TrainingData,
        *,
        schema: Mapping[str, Any] | None = None,
        objective: Mapping[str, Any] | str | None = None,
        sample_weight: np.ndarray | None = None,
        metadata: Mapping[str, Any] | None = None,
        task_id: str = "train_task",
    ) -> "TrainTask":
        return cls(
            data=data,
            schema=None if schema is None else dict(schema),
            objective=objective,
            sample_weight=sample_weight,
            metadata=dict(metadata or {}),
            task_id=str(task_id),
        )


__all__ = [
    "TrainingData",
    "TrainTask",
]
