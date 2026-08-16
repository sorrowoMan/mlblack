"""Optimizer adapter for mlblack trainers.

Inherits the unified AdapterBase from blackbase and adds mlblack-specific
features (UnknownState/Feedback types, ContractMixin, coerce_states).
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Mapping, Sequence

from blackbase.abc import AdapterBase

from .contracts import ComponentContract, ContractMixin
from .types import Feedback, UnknownState


class OptimizerAdapter(AdapterBase, ContractMixin):
    """Optimization strategy, equivalent to nsgablack's Adapter.

    Inherits AdapterBase (unified interface) and ContractMixin
    (mlblack metadata protocol).
    """

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

    # --- AdapterBase abstract methods ---

    @abstractmethod
    def propose(self, control: Any, context: Mapping[str, Any]) -> Sequence[UnknownState]:
        """Return unknown states to evaluate."""

    @abstractmethod
    def update(
        self,
        control: Any,
        candidates: Sequence[UnknownState],
        feedback: Any,
        context: Mapping[str, Any],
    ) -> None:
        """Update adapter state from evaluation feedback."""

    # --- Lifecycle ---

    def setup(self, control: Any) -> None:
        return None

    def teardown(self, control: Any) -> None:
        return None

    # --- State persistence ---

    def get_state(self) -> Mapping[str, Any]:
        return {}

    def set_state(self, state: Mapping[str, Any]) -> None:
        _ = state
        return None

    def get_population(self) -> tuple[UnknownState, ...] | None:
        """Return adapter-authoritative runtime states after update, if owned."""
        return None

    def set_population(self, population: Sequence[UnknownState]) -> bool:
        """Restore adapter-authoritative states when the adapter supports it."""
        _ = population
        return False

    # --- mlblack-specific: coerce_states ---

    def coerce_states(self, states: Any) -> tuple[UnknownState, ...]:
        """Normalize states to a tuple."""
        if states is None:
            return tuple()
        if isinstance(states, UnknownState):
            return (states,)
        return tuple(states)

    # --- mlblack-specific: context projection ---

    def get_context_projection(self, control: Any) -> Mapping[str, Any]:
        _ = control
        return {}

    # --- Override: describe uses ContractMixin ---

    def describe(self) -> Mapping[str, Any]:
        return {"name": self.name, "contract": self.get_context_contract()}


AlgorithmAdapter = OptimizerAdapter
