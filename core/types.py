from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class UnknownState:
    """Optimizable unknown state, equivalent to an nsgablack candidate."""

    values: np.ndarray
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def as_array(self) -> np.ndarray:
        return np.asarray(self.values, dtype=float).reshape(-1)

    def with_values(self, values: Sequence[float] | np.ndarray, **metadata: Any) -> "UnknownState":
        merged = dict(self.metadata)
        merged.update(metadata)
        return UnknownState(values=np.asarray(values, dtype=float).reshape(-1), metadata=merged)


@dataclass(frozen=True)
class Feedback:
    """Data-dependent optimization feedback returned by a LearningProblem."""

    objectives: np.ndarray
    constraints: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=float))
    loss: float | None = None
    gradients: np.ndarray | None = None
    residuals: np.ndarray | None = None
    metrics: Mapping[str, Any] = field(default_factory=dict)
    signals: Mapping[str, Any] = field(default_factory=dict)

    def scalar_score(self, *, constraint_penalty: float = 1e6) -> float:
        obj = np.asarray(self.objectives, dtype=float).reshape(-1)
        cons = np.asarray(self.constraints, dtype=float).reshape(-1)
        violation = float(np.sum(np.maximum(cons, 0.0))) if cons.size else 0.0
        return float(np.sum(obj)) + (float(constraint_penalty) * violation)


@dataclass(frozen=True)
class PopulationSnapshot:
    """Snapshot of unknown states and evaluation feedback."""

    states: tuple[UnknownState, ...]
    feedback: tuple[Feedback, ...]
    step: int
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TrainerResult:
    """Result returned by Trainer.fit()."""

    best_state: UnknownState | None
    best_model: Any | None
    best_feedback: Feedback | None
    history: tuple[Mapping[str, Any], ...]
    report: Mapping[str, Any]
