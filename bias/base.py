"""Optimization bias for mlblack.

Inherits the unified BiasBase from blackbase. In mlblack, bias primarily
injects preference signals via project_context. The adjust_feedback method
that modified loss has been removed — loss weighting belongs in Problem.

Note: For single-loss training, bias is rarely needed. It becomes relevant
in multi-loss scenarios where preference between objectives is expressed
without changing the evaluation itself.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from blackbase.abc import BiasBase

from mlblack.core.contracts import ComponentContract, ContractMixin
from mlblack.core.types import Feedback, UnknownState


class OptimizationBias(BiasBase, ContractMixin):
    """Soft preference component for optimization feedback/context.

    Inherits BiasBase (unified interface) and ContractMixin
    (mlblack metadata protocol).

    In mlblack, bias primarily uses project_context to inject preference
    signals. The adjust() method is available for multi-loss preference
    expression, but does NOT change the evaluation itself.
    """

    name = "optimization_bias"
    context_requires = ()
    context_optional = ('feedback.objectives', 'candidate.unknown_state', 'trainer.context')
    context_provides = ('bias.soft_preference',)
    context_mutates = ('trainer.context',)
    context_cache = ()
    requires_metrics = ('objective',)
    metrics_fallback = "strict"
    context_notes = 'provides bias.soft_preference; mutates trainer.context.'
    contract = ComponentContract(
        name=name,
        optional=("feedback.objectives", "candidate.unknown_state", "trainer.context"),
        provides=("bias.soft_preference",),
        mutates=("trainer.context",),
        supports_batch=True,
        metadata={"component": "bias"},
    )

    # --- Override: project_context injects preference signals ---

    def project_context(self, context: Mapping[str, Any]) -> Mapping[str, Any]:
        """Inject preference signals into context.

        Default: pass through without modification.
        Subclasses override to express preferences.
        """
        return dict(context)

    # --- Override: adjust for multi-loss preference ---

    def adjust(self, feedback: Any, context: Mapping[str, Any]) -> Any:
        """Apply preference adjustment in multi-loss scenarios.

        Does NOT change the evaluation itself — only expresses relative
        preference between objectives. Default: pass through.
        """
        return feedback

    # --- Backward compatibility: adjust_feedback delegates to adjust ---

    def adjust_feedback(
        self,
        control: Any,
        states: Sequence[UnknownState],
        feedback: Sequence[Feedback],
        context: Mapping[str, Any],
    ) -> tuple[Feedback, ...]:
        """Legacy method: delegates to adjust() for each feedback item.

        Deprecated: prefer using adjust() directly.
        """
        _ = control
        _ = states
        return tuple(self.adjust(fb, context) for fb in feedback)

    # --- Override: describe uses ContractMixin ---

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "contract": self.get_context_contract()}
