from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from mlblack.core.adapter import OptimizerAdapter
from mlblack.core.contracts import ComponentContract
from mlblack.core.types import Feedback, UnknownState


@dataclass(frozen=True)
class RandomSearchConfig:
    population_size: int = 16
    mutation_scale: float = 0.25
    random_seed: int = 42
    exploit_best: bool = True


class RandomSearchAdapter(OptimizerAdapter):
    """Black-box adapter for non-gradient decoded models/specs."""

    name = "random_search"
    context_requires = ('feedback.objectives', 'candidate.unknown_state')
    context_optional = ()
    context_provides = ('population.candidates',)
    context_mutates = ('adapter.best_state',)
    context_cache = ()
    requires_metrics = ('objective',)
    metrics_fallback = "strict"
    context_notes = 'Reads context fields: feedback.objectives, candidate.unknown_state; provides population.candidates; mutates adapter.best_state.'
    contract = ComponentContract(
        name=name,
        requires=("feedback.objectives", "candidate.unknown_state"),
        provides=("population.candidates",),
        mutates=("adapter.best_state",),
        supports_gradient=False,
        supports_batch=True,
        supports_resume=True,
    )

    def __init__(self, config: RandomSearchConfig | None = None, **kwargs: Any) -> None:
        if config is not None and kwargs:
            raise ValueError("provide config or kwargs, not both")
        self.config = config or RandomSearchConfig(**kwargs)
        self._rng = np.random.default_rng(int(self.config.random_seed))
        self.best_state: UnknownState | None = None
        self.best_score: float | None = None

    def propose(self, trainer: Any, context: Mapping[str, Any]) -> Sequence[UnknownState]:
        n = max(1, int(self.config.population_size))
        states: list[UnknownState] = []
        if self.config.exploit_best and self.best_state is not None:
            states.append(self.best_state)
            base = self.best_state.as_array()
            while len(states) < n:
                noise = self._rng.normal(0.0, float(self.config.mutation_scale), size=base.shape[0])
                states.append(self.best_state.with_values(base + noise, adapter=self.name, source="mutated_best"))
            return tuple(states)
        return tuple(trainer.init_population(n, context))

    def update(
        self,
        trainer: Any,
        states: Sequence[UnknownState],
        feedback: Sequence[Feedback],
        context: Mapping[str, Any],
    ) -> None:
        _ = trainer
        _ = context
        for state, fb in zip(states, feedback):
            score = fb.scalar_score()
            if self.best_score is None or score < self.best_score:
                self.best_score = float(score)
                self.best_state = state

    def get_state(self) -> Mapping[str, Any]:
        return {
            "best_state": None if self.best_state is None else self.best_state.as_array().tolist(),
            "best_score": self.best_score,
            "population_size": int(self.config.population_size),
            "mutation_scale": float(self.config.mutation_scale),
        }

    def set_state(self, state: Mapping[str, Any]) -> None:
        values = state.get("best_state")
        self.best_state = None if values is None else UnknownState(values=np.asarray(values, dtype=float))
        score = state.get("best_score")
        self.best_score = None if score is None else float(score)

