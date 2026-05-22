from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class TrainingContract:
    """Boundary contract for outer optimizers invoking mlblack as an inner run."""

    requires: Sequence[str] = ("data", "trainer_spec")
    provides: Sequence[str] = ("objectives", "constraints", "metrics", "report")
    objective_names: Sequence[str] = ("loss",)
    constraint_names: Sequence[str] = tuple()
    metric_names: Sequence[str] = tuple()
    resource_keys: Sequence[str] = ("threads", "device_tokens", "gpu_tokens")
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "requires": list(self.requires),
            "provides": list(self.provides),
            "objective_names": list(self.objective_names),
            "constraint_names": list(self.constraint_names),
            "metric_names": list(self.metric_names),
            "resource_keys": list(self.resource_keys),
            "metadata": dict(self.metadata),
        }
