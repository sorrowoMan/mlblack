from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class TrainingResultRecord:
    task_id: str
    objectives: Sequence[float]
    constraints: Sequence[float] = tuple()
    metrics: Mapping[str, Any] = field(default_factory=dict)
    report: Mapping[str, Any] = field(default_factory=dict)
    artifact_refs: Mapping[str, Any] = field(default_factory=dict)
    resource_context: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_trainer_result(cls, task_id: str, result: Any, *, artifact_refs: Mapping[str, Any] | None = None, resource_context: Mapping[str, Any] | None = None) -> "TrainingResultRecord":
        feedback = getattr(result, "best_feedback", None)
        objectives = [] if feedback is None else np.asarray(feedback.objectives, dtype=float).reshape(-1).tolist()
        constraints = [] if feedback is None else np.asarray(feedback.constraints, dtype=float).reshape(-1).tolist()
        metrics = {} if feedback is None else dict(feedback.metrics)
        report = dict(getattr(result, "report", {}) or {})
        return cls(
            task_id=str(task_id),
            objectives=tuple(float(v) for v in objectives),
            constraints=tuple(float(v) for v in constraints),
            metrics=metrics,
            report=report,
            artifact_refs=dict(artifact_refs or {}),
            resource_context=dict(resource_context or {}),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "objectives": [float(v) for v in self.objectives],
            "constraints": [float(v) for v in self.constraints],
            "metrics": dict(self.metrics),
            "report": dict(self.report),
            "artifact_refs": dict(self.artifact_refs),
            "resource_context": dict(self.resource_context),
        }
