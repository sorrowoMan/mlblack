from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class TrainingLineage:
    mode: str = "fresh"
    trainer_name: str = "unknown_trainer"
    parent_artifact_id: str | None = None
    parent_state_trainer: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", str(self.mode).strip().lower() or "fresh")
        object.__setattr__(self, "trainer_name", str(self.trainer_name))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": str(self.mode),
            "trainer_name": str(self.trainer_name),
            "parent_artifact_id": None if self.parent_artifact_id is None else str(self.parent_artifact_id),
            "parent_state_trainer": None if self.parent_state_trainer is None else str(self.parent_state_trainer),
            "metadata": dict(self.metadata),
        }


__all__ = ["TrainingLineage"]
