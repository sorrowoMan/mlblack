from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence
from uuid import uuid4


@dataclass(frozen=True)
class TrainingTask:
    """JSON-compatible request for one inner mlblack training run."""

    task_id: str = field(default_factory=lambda: f"task_{uuid4().hex[:12]}")
    trainer_spec: Mapping[str, Any] = field(default_factory=dict)
    max_steps: int = 100
    resource_context: Mapping[str, Any] = field(default_factory=dict)
    outer_candidate: Sequence[float] = tuple()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_value(cls, value: Mapping[str, Any] | "TrainingTask") -> "TrainingTask":
        if isinstance(value, TrainingTask):
            return value
        payload = dict(value)
        return cls(
            task_id=str(payload.get("task_id", f"task_{uuid4().hex[:12]}")),
            trainer_spec=dict(payload.get("trainer_spec", {}) or {}),
            max_steps=int(payload.get("max_steps", 100)),
            resource_context=dict(payload.get("resource_context", {}) or {}),
            outer_candidate=tuple(float(v) for v in payload.get("outer_candidate", ())),
            metadata=dict(payload.get("metadata", {}) or {}),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "trainer_spec": dict(self.trainer_spec),
            "max_steps": int(self.max_steps),
            "resource_context": dict(self.resource_context),
            "outer_candidate": [float(v) for v in self.outer_candidate],
            "metadata": dict(self.metadata),
        }
