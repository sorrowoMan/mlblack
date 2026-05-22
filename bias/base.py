from __future__ import annotations

from abc import ABC
from typing import Any, Mapping, Sequence

from mlblack.core.contracts import ComponentContract, ContractMixin
from mlblack.core.types import Feedback, UnknownState


class OptimizationBias(ContractMixin, ABC):
    """Soft preference component for optimization feedback/context.

    Bias is not a hard constraint and is not a trainer. It can expose hints to
    adapters through context, or softly adjust feedback before adapter.update().
    """

    name = "optimization_bias"
    context_requires = ()
    context_optional = ('feedback.objectives', 'candidate.unknown_state', 'trainer.context')
    context_provides = ('bias.soft_preference',)
    context_mutates = ('feedback.objectives', 'trainer.context')
    context_cache = ()
    requires_metrics = ('objective',)
    metrics_fallback = "strict"
    context_notes = 'provides bias.soft_preference; mutates feedback.objectives, trainer.context.'
    contract = ComponentContract(
        name=name,
        optional=("feedback.objectives", "candidate.unknown_state", "trainer.context"),
        provides=("bias.soft_preference",),
        mutates=("feedback.objectives", "trainer.context"),
        supports_batch=True,
        metadata={"component": "bias"},
    )

    def project_context(self, trainer: Any, context: Mapping[str, Any]) -> Mapping[str, Any]:
        _ = trainer
        return dict(context)

    def adjust_feedback(
        self,
        trainer: Any,
        states: Sequence[UnknownState],
        feedback: Sequence[Feedback],
        context: Mapping[str, Any],
    ) -> tuple[Feedback, ...]:
        _ = trainer
        _ = states
        _ = context
        return tuple(feedback)

    def describe(self) -> dict[str, Any]:
        return {"name": self.name}

