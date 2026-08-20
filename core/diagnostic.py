"""One-shot ML diagnostic semantics on the canonical NSGABlack lifecycle."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from blackbase.contracts import ComponentContract
from .problem import LearningProblem
from .representation import ModelRepresentation
from .types import Feedback, UnknownState


DiagnosticRunner = Callable[[Mapping[str, Any]], Feedback | Mapping[str, Any] | None]


class DiagnosticProblem(LearningProblem):
    """Problem surface for a single non-search ML diagnostic evaluation."""

    name = "diagnostic_problem"
    context_requires = ()
    context_optional = ("resource_context",)
    context_provides = ("feedback.objectives", "feedback.metrics")
    context_mutates = ()
    context_cache = ()
    context_notes = "Runs one diagnostic and returns compact Feedback; large outputs stay on the problem instance."
    contract = ComponentContract(
        name=name,
        requires=(),
        provides=("feedback.objectives", "feedback.metrics"),
        supports_batch=False,
        supports_resume=False,
        metadata={"family": "diagnostic", "execution": "one_shot"},
    )

    def __init__(self, runner: DiagnosticRunner, *, name: str = name) -> None:
        if not callable(runner):
            raise TypeError("DiagnosticProblem runner must be callable")
        self.runner = runner
        self.name = str(name)
        self.last_result: Feedback | Mapping[str, Any] | None = None

    def evaluate(
        self,
        candidate: Any,
        state: UnknownState | Mapping[str, Any] | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> Feedback:
        del candidate
        if context is None and isinstance(state, Mapping):
            context = state
        result = self.runner(dict(context or {}))
        self.last_result = result
        if isinstance(result, Feedback):
            return result
        payload = dict(result or {}) if isinstance(result, Mapping) else {}
        objective = float(payload.get("objective", 0.0))
        metrics = _compact_metrics(payload.get("metrics", payload))
        metrics.setdefault("diagnostic.status", str(payload.get("status", "ok")))
        return Feedback(
            objectives=np.asarray([objective], dtype=float),
            constraints=np.zeros(0, dtype=float),
            metrics=metrics,
            info={"result_type": type(result).__name__},
        )

    def describe(self) -> Mapping[str, Any]:
        runner_name = getattr(self.runner, "__name__", type(self.runner).__name__)
        return {
            "name": self.name,
            "family": "diagnostic",
            "execution": "one_shot",
            "runner": str(runner_name),
            "contract": self.get_context_contract(),
        }


class DiagnosticRepresentation(ModelRepresentation):
    """A semantic placeholder candidate for one-shot diagnostic evaluation."""

    name = "diagnostic_constant"

    def init(self, context: Mapping[str, Any]) -> UnknownState:
        return UnknownState(
            values=np.zeros(1, dtype=float),
            metadata={
                "source": "diagnostic",
                "run_name": str(context.get("run_name", "diagnostic_run")),
            },
        )

    def decode(
        self,
        state: UnknownState,
        context: Mapping[str, Any] | None = None,
    ) -> None:
        del state, context
        return None


def _compact_metrics(payload: Mapping[str, Any]) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            metrics[str(key)] = value
    return metrics


__all__ = [
    "DiagnosticProblem",
    "DiagnosticRepresentation",
    "DiagnosticRunner",
]
