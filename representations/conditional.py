from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from blackbase.contracts import ComponentContract
from mlblack.core.representation import ModelRepresentation
from mlblack.core.types import UnknownState
from mlblack.models import PiecewiseModel, Router, ThresholdRouter


@dataclass(frozen=True)
class PiecewiseRepresentationConfig:
    branch_names: tuple[str, ...] = tuple()
    default_branch: int = 0


class PiecewiseRepresentation(ModelRepresentation):
    """Concatenate branch unknown states and decode a routed piecewise model."""

    name = "piecewise"
    context_requires = ('candidate.unknown_state', 'router', 'branch_representations')
    context_optional = ()
    context_provides = ('candidate.model', 'model.predict')
    context_mutates = ('candidate.repaired_state',)
    context_cache = ()
    requires_metrics = ()
    metrics_fallback = "strict"
    context_notes = 'Reads context fields: candidate.unknown_state, router, branch_representations; provides candidate.model, model.predict; mutates candidate.repaired_state.'
    contract = ComponentContract(
        name=name,
        requires=("candidate.unknown_state", "router", "branch_representations"),
        provides=("candidate.model", "model.predict"),
        mutates=("candidate.repaired_state",),
        supports_batch=True,
        supports_resume=True,
        metadata={"family": "conditional", "model": "piecewise"},
    )

    def __init__(
        self,
        *,
        router: Router | None = None,
        branches: Sequence[ModelRepresentation],
        config: PiecewiseRepresentationConfig | None = None,
    ) -> None:
        self.router = router or ThresholdRouter(feature_index=0, thresholds=(0.0,))
        self.branches = tuple(branches)
        if not self.branches:
            raise ValueError("PiecewiseRepresentation requires at least one branch")
        self.config = config or PiecewiseRepresentationConfig()
        self.dimensions = tuple(int(getattr(branch, "dimension", getattr(branch, "base_dimension", 0))) for branch in self.branches)
        if any(dim <= 0 for dim in self.dimensions):
            raise ValueError("all branch representations must expose positive dimension")
        self.dimension = int(sum(self.dimensions))

    def init(self, context: Mapping[str, Any]) -> UnknownState:
        values = np.concatenate([branch.init(context).as_array() for branch in self.branches])
        return UnknownState(values=values, metadata={"source": "piecewise_init"})

    def decode(self, state: UnknownState, context: Mapping[str, Any] | None = None) -> PiecewiseModel:
        ctx = dict(context or {})
        parts = self._split(state)
        branch_models = [
            branch.decode(UnknownState(values=values, metadata=dict(state.metadata)), ctx)
            for branch, values in zip(self.branches, parts)
        ]
        return PiecewiseModel(
            router=self.router,
            branch_models=tuple(branch_models),
            default_branch=int(self.config.default_branch),
            metadata={"representation": self.name},
        )

    def repair(self, state: UnknownState, context: Mapping[str, Any] | None = None) -> UnknownState:
        ctx = dict(context or {})
        repaired: list[np.ndarray] = []
        for branch, values in zip(self.branches, self._split(state)):
            fixed = branch.repair(UnknownState(values=values, metadata=dict(state.metadata)), ctx)
            repaired.append(fixed.as_array())
        return state.with_values(np.concatenate(repaired), repaired=True)

    def _split(self, state: UnknownState) -> tuple[np.ndarray, ...]:
        arr = state.as_array()
        if arr.shape[0] != self.dimension:
            fixed = np.zeros(self.dimension, dtype=float)
            fixed[: min(self.dimension, arr.shape[0])] = arr[: min(self.dimension, arr.shape[0])]
            arr = fixed
        parts = []
        offset = 0
        for dim in self.dimensions:
            parts.append(arr[offset : offset + dim])
            offset += dim
        return tuple(parts)

    def describe(self) -> Mapping[str, Any]:
        return {
            "name": self.name,
            "dimension": int(self.dimension),
            "router": self.router.describe(),
            "branches": [branch.describe() for branch in self.branches],
            "default_branch": int(self.config.default_branch),
        }

