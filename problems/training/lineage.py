from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class TrainingLineage:
    parent_run_id: str = ""
    task_id: str = ""
    outer_candidate_id: str = ""
    namespace: str = "mlblack.inner"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "parent_run_id": self.parent_run_id,
            "task_id": self.task_id,
            "outer_candidate_id": self.outer_candidate_id,
            "namespace": self.namespace,
            "metadata": dict(self.metadata),
        }
