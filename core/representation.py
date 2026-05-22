from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping, Sequence

from .contracts import ComponentContract, ContractMixin
from .types import UnknownState


class ModelRepresentation(ContractMixin, ABC):
    """Unknown-state encoder/decoder, equivalent to nsgablack Representation."""

    name = "model_representation"
    context_requires = ('candidate.unknown_state',)
    context_optional = ()
    context_provides = ('candidate.model',)
    context_mutates = ('candidate.repaired_state',)
    context_cache = ()
    requires_metrics = ()
    metrics_fallback = "strict"
    context_notes = 'Reads context fields: candidate.unknown_state; provides candidate.model; mutates candidate.repaired_state.'
    contract = ComponentContract(
        name=name,
        requires=("candidate.unknown_state",),
        provides=("candidate.model",),
        mutates=("candidate.repaired_state",),
        supports_batch=True,
    )

    @abstractmethod
    def init(self, context: Mapping[str, Any]) -> UnknownState:
        """Create an initial unknown state."""

    def init_batch(self, n: int, context: Mapping[str, Any] | None = None) -> tuple[UnknownState, ...]:
        ctx = dict(context or {})
        return tuple(self.init(ctx) for _ in range(int(n)))

    def encode(self, model: Any, context: Mapping[str, Any] | None = None) -> UnknownState:
        _ = context
        raise NotImplementedError(f"{type(self).__name__}.encode(...) is not implemented")

    @abstractmethod
    def decode(self, state: UnknownState, context: Mapping[str, Any] | None = None) -> Any:
        """Decode unknown state into a model/function."""

    def decode_batch(self, states: Sequence[UnknownState], context: Mapping[str, Any] | None = None) -> tuple[Any, ...]:
        ctx = dict(context or {})
        return tuple(self.decode(state, ctx) for state in tuple(states))

    def repair(self, state: UnknownState, context: Mapping[str, Any] | None = None) -> UnknownState:
        _ = context
        return state

    def repair_batch(self, states: Sequence[UnknownState], context: Mapping[str, Any] | None = None) -> tuple[UnknownState, ...]:
        ctx = dict(context or {})
        return tuple(self.repair(state, ctx) for state in tuple(states))

    def mutate(self, state: UnknownState, context: Mapping[str, Any] | None = None) -> UnknownState:
        _ = context
        return state

    def describe(self) -> Mapping[str, Any]:
        return {"name": self.name}


RepresentationPipeline = ModelRepresentation
