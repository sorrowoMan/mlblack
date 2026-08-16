"""Learning problem for mlblack trainers.

Inherits the unified ProblemBase from blackbase and adds mlblack-specific
features (Feedback type, ContractMixin, model+state evaluation).
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Mapping

from blackbase.abc import ProblemBase

from .contracts import ComponentContract, ContractMixin
from .types import Feedback, UnknownState


class LearningProblem(ProblemBase, ContractMixin):
    """Data-dependent evaluator, equivalent to nsgablack Problem.

    Inherits ProblemBase (unified interface) and ContractMixin
    (mlblack metadata protocol).
    """

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

    # --- ProblemBase abstract method ---

    @abstractmethod
    def evaluate(self, candidate: Any, context: Mapping[str, Any] | None = None) -> Feedback:
        """Evaluate decoded model and return objectives/constraints/signals.

        In mlblack, candidate is typically (model, state) or just model.
        The context parameter provides data and other runtime info.
        """

    # --- Override: describe uses ContractMixin ---

    def describe(self) -> Mapping[str, Any]:
        return {"name": self.name, "contract": self.get_context_contract()}
