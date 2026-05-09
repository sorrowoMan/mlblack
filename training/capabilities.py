from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


TrainingMode = str


@dataclass(frozen=True)
class TrainerCapabilities:
    supports_fresh: bool = True
    supports_resume: bool = False
    supports_warm_start: bool = False
    supports_incremental: bool = False
    supports_recalibration: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def supports(self, mode: TrainingMode) -> bool:
        key = str(mode).strip().lower()
        if key == "fresh":
            return bool(self.supports_fresh)
        if key == "resume":
            return bool(self.supports_resume)
        if key == "warm_start":
            return bool(self.supports_warm_start)
        if key == "incremental":
            return bool(self.supports_incremental)
        if key == "recalibrate":
            return bool(self.supports_recalibration)
        return False

    def as_dict(self) -> dict[str, Any]:
        return {
            "supports_fresh": bool(self.supports_fresh),
            "supports_resume": bool(self.supports_resume),
            "supports_warm_start": bool(self.supports_warm_start),
            "supports_incremental": bool(self.supports_incremental),
            "supports_recalibration": bool(self.supports_recalibration),
            "metadata": dict(self.metadata),
        }


def coerce_trainer_capabilities(value: TrainerCapabilities | Mapping[str, Any] | None) -> TrainerCapabilities:
    if value is None:
        return TrainerCapabilities()
    if isinstance(value, TrainerCapabilities):
        return value
    raw = dict(value)
    return TrainerCapabilities(
        supports_fresh=bool(raw.get("supports_fresh", True)),
        supports_resume=bool(raw.get("supports_resume", raw.get("resume", False))),
        supports_warm_start=bool(raw.get("supports_warm_start", raw.get("warm_start", False))),
        supports_incremental=bool(raw.get("supports_incremental", raw.get("incremental", False))),
        supports_recalibration=bool(raw.get("supports_recalibration", raw.get("recalibration", False))),
        metadata=dict(raw),
    )


__all__ = [
    "TrainingMode",
    "TrainerCapabilities",
    "coerce_trainer_capabilities",
]
