"""Learning problem for mlblack trainers.

Inherits the unified ProblemBase from blackbase and adds mlblack-specific
features (Feedback type, ContractMixin, model+state evaluation).
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Mapping

from blackbase.abc import ProblemBase

from blackbase.contracts import ComponentContract, ContractMixin
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
    objective_count = 1

    def get_num_objectives(self) -> int:
        """Return the stable optimization objective arity.

        Objective shape is part of the Problem contract, not something the
        control plane may guess from the first evaluation. Problems with a
        configurable objective vector should override this method.
        """

        count = int(self.objective_count)
        if count <= 0:
            raise ValueError("LearningProblem objective_count must be positive")
        return count

    def prepare_model_for_evaluation(
        self,
        model: Any,
        state: UnknownState,
        context: Mapping[str, Any],
    ) -> Any:
        """Materialize one decoded candidate for semantic evaluation.

        Representations own candidate encoding and decoding.  Problems and
        Providers own data-dependent work such as closed-form fitting or an
        external estimator's ``fit(...)`` call.  The default is the identity
        projection used by already-executable models.
        """

        del state, context
        return model

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
