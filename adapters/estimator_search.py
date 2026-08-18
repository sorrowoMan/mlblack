from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from mlblack.core.adapter import OptimizerAdapter
from blackbase.contracts import ComponentContract
from mlblack.core.types import Feedback, UnknownState


@dataclass(frozen=True)
class EstimatorSpecSearchConfig:
    """Black-box search config for decoded external estimator specs."""

    population_size: int = 12
    mutation_scale: float = 0.2
    random_seed: int = 42
    exploit_best: bool = True
    include_center_candidate: bool = True


class EstimatorSpecSearchAdapter(OptimizerAdapter):
    """Search adapter for tree/boosting/sklearn-style estimator specs.

    The adapter does not fit estimators. It proposes UnknownState values that
    decode into EstimatorSpecModel; the problem/provider side performs fitting
    and returns feedback.
    """

    name = "estimator_spec_search"
    context_requires = ('candidate.model_spec', 'feedback.objectives')
    context_optional = ('estimator.factory', 'feedback.metrics', 'feedback.residuals')
    context_provides = ('population.candidates',)
    context_mutates = ('adapter.best_state', 'adapter.search_state')
    context_cache = ()
    requires_metrics = ('objective',)
    metrics_fallback = "strict"
    context_notes = 'Reads context fields: candidate.model_spec, feedback.objectives; provides population.candidates; mutates adapter.best_state, adapter.search_state.'
    contract = ComponentContract(
        name=name,
        requires=("candidate.model_spec", "feedback.objectives"),
        optional=("estimator.factory", "feedback.metrics", "feedback.residuals"),
        provides=("population.candidates",),
        mutates=("adapter.best_state", "adapter.search_state"),
        supports_gradient=False,
        supports_batch=True,
        supports_resume=True,
        metadata={"families": ("tree", "tree_boosting", "neural"), "routes": ("external_estimator",)},
    )

    def __init__(self, config: EstimatorSpecSearchConfig | None = None, **kwargs: Any) -> None:
        if config is not None and kwargs:
            raise ValueError("provide config or kwargs, not both")
        self.config = config or EstimatorSpecSearchConfig(**kwargs)
        self._rng = np.random.default_rng(int(self.config.random_seed))
        self.best_state: UnknownState | None = None
        self.best_score: float | None = None
        self.step_index = 0

    def propose(self, control: Any, context: Mapping[str, Any]) -> Sequence[UnknownState]:
        n = max(1, int(self.config.population_size))
        states: list[UnknownState] = []

        if self.best_state is None:
            initial = control.init_candidate(context)
            if self.config.include_center_candidate:
                states.append(initial)
            base = initial.as_array()
        else:
            base = self.best_state.as_array()
            if self.config.exploit_best:
                states.append(self.best_state)

        while len(states) < n:
            noise = self._rng.normal(0.0, float(self.config.mutation_scale), size=base.shape[0])
            source = "mutated_best" if self.best_state is not None else "random_around_center"
            states.append(UnknownState(values=base + noise, metadata={"adapter": self.name, "source": source}))
        return tuple(states)

    def update(
        self,
        control: Any,
        states: Sequence[UnknownState],
        feedback: Sequence[Feedback],
        context: Mapping[str, Any],
    ) -> None:
        _ = control
        _ = context
        self.step_index += 1
        for state, fb in zip(states, feedback):
            score = fb.scalar_score()
            if self.best_score is None or score < self.best_score:
                self.best_score = float(score)
                self.best_state = state

    def get_state(self) -> Mapping[str, Any]:
        return {
            "best_state": None if self.best_state is None else self.best_state.as_array().tolist(),
            "best_score": self.best_score,
            "step_index": int(self.step_index),
            "population_size": int(self.config.population_size),
            "mutation_scale": float(self.config.mutation_scale),
        }

    def get_population(self) -> tuple[UnknownState, ...] | None:
        return None if self.best_state is None else (self.best_state,)

    def set_population(self, population: Sequence[UnknownState]) -> bool:
        states = tuple(population)
        self.best_state = states[0] if states else None
        return True

    def set_state(self, state: Mapping[str, Any]) -> None:
        values = state.get("best_state")
        self.best_state = None if values is None else UnknownState(values=np.asarray(values, dtype=float))
        score = state.get("best_score")
        self.best_score = None if score is None else float(score)
        self.step_index = int(state.get("step_index", self.step_index))

