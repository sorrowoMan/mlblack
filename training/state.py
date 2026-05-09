from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class TrainerState:
    trainer_name: str
    payload: Any
    schema_signature: str | None = None
    feature_signature: str | None = None
    target_signature: str | None = None
    objective_signature: str | None = None
    pipeline_signature: str | None = None
    numericizer_signature: str | None = None
    regime_signature: str | None = None
    symbolic_family_signature: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "trainer_name", str(self.trainer_name))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def as_dict(self) -> dict[str, Any]:
        return {
            "trainer_name": str(self.trainer_name),
            "schema_signature": self.schema_signature,
            "feature_signature": self.feature_signature,
            "target_signature": self.target_signature,
            "objective_signature": self.objective_signature,
            "pipeline_signature": self.pipeline_signature,
            "numericizer_signature": self.numericizer_signature,
            "regime_signature": self.regime_signature,
            "symbolic_family_signature": self.symbolic_family_signature,
            "metadata": dict(self.metadata),
        }


__all__ = ["TrainerState"]
