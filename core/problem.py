from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping

from .contracts import ComponentContract, ContractMixin
from .types import Feedback, UnknownState


class LearningProblem(ContractMixin, ABC):
    """Data-dependent evaluator, equivalent to nsgablack Problem."""

    name = "learning_problem"
    context_requires = ('candidate.model', 'data')
    context_optional = ()
    context_provides = ('feedback.objectives', 'feedback.metrics')
    context_mutates = ()
    context_cache = ()
    requires_metrics = ()
    metrics_fallback = "strict"
    context_notes = 'Reads context fields: candidate.model, data; provides feedback.objectives, feedback.metrics.'
    contract = ComponentContract(
        name=name,
        requires=("candidate.model", "data"),
        provides=("feedback.objectives", "feedback.metrics"),
        supports_batch=False,
    )

    @abstractmethod
    def evaluate(self, model: Any, state: UnknownState, context: Mapping[str, Any]) -> Feedback:
        """Evaluate decoded model and return objectives/constraints/signals."""

    def describe(self) -> Mapping[str, Any]:
        return {"name": self.name}
