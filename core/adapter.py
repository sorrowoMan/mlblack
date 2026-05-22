from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping, Sequence

from .contracts import ComponentContract, ContractMixin
from .types import Feedback, UnknownState


class OptimizerAdapter(ContractMixin, ABC):
    """Optimization strategy, equivalent to nsgablack's Adapter."""

    name = "optimizer_adapter"
    priority = 0
    context_requires = ('population.feedback',)
    context_optional = ()
    context_provides = ('population.candidates',)
    context_mutates = ('adapter.state',)
    context_cache = ()
    requires_metrics = ()
    metrics_fallback = "strict"
    context_notes = 'Reads context fields: population.feedback; provides population.candidates; mutates adapter.state.'
    contract = ComponentContract(
        name=name,
        requires=("population.feedback",),
        provides=("population.candidates",),
        mutates=("adapter.state",),
        supports_batch=True,
    )

    def setup(self, trainer: Any) -> None:
        return None

    @abstractmethod
    def propose(self, trainer: Any, context: Mapping[str, Any]) -> Sequence[UnknownState]:
        """Return unknown states to evaluate."""

    @abstractmethod
    def update(
        self,
        trainer: Any,
        states: Sequence[UnknownState],
        feedback: Sequence[Feedback],
        context: Mapping[str, Any],
    ) -> None:
        """Update adapter state from evaluation feedback."""

    def teardown(self, trainer: Any) -> None:
        return None

    def get_state(self) -> Mapping[str, Any]:
        return {}

    def set_state(self, state: Mapping[str, Any]) -> None:
        _ = state
        return None

    def get_context_projection(self, trainer: Any) -> Mapping[str, Any]:
        _ = trainer
        return {}

    def coerce_states(self, states: Any) -> tuple[UnknownState, ...]:
        if states is None:
            return tuple()
        if isinstance(states, UnknownState):
            return (states,)
        return tuple(states)


AlgorithmAdapter = OptimizerAdapter
