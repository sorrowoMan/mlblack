from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from core.common.contracts import SurrogateArtifact

from .lineage import TrainingLineage
from .state import TrainerState


@dataclass(frozen=True)
class FitResult:
    artifact: SurrogateArtifact
    trainer_state: TrainerState | None = None
    report: Mapping[str, Any] = field(default_factory=dict)
    lineage: TrainingLineage = field(default_factory=TrainingLineage)

    def __post_init__(self) -> None:
        object.__setattr__(self, "report", dict(self.report))


__all__ = ["FitResult"]
