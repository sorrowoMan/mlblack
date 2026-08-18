from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from mlblack.core.adapter import OptimizerAdapter
from blackbase.contracts import ComponentContract
from mlblack.core.types import Feedback, UnknownState


@dataclass(frozen=True)
class GradientDescentConfig:
    learning_rate: float = 0.05
    min_learning_rate: float = 1e-8
    max_grad_norm: float | None = 1e3
    require_gradient: bool = True


class GradientDescentAdapter(OptimizerAdapter):
    """Gradient-descent optimizer adapter.

    It does not consume data. It only consumes gradients returned by the
    LearningProblem feedback.
    """

    name = "gradient_descent"
    context_requires = ('feedback.gradients', 'candidate.unknown_state')
    context_optional = ()
    context_provides = ('population.candidates',)
    context_mutates = ('adapter.current_state',)
    context_cache = ()
    requires_metrics = ()
    metrics_fallback = "strict"
    context_notes = 'Reads context fields: feedback.gradients, candidate.unknown_state; provides population.candidates; mutates adapter.current_state.'
    contract = ComponentContract(
        name=name,
        requires=("feedback.gradients", "candidate.unknown_state"),
        provides=("population.candidates",),
        mutates=("adapter.current_state",),
        supports_gradient=True,
        supports_batch=False,
        supports_resume=True,
    )

    def __init__(self, config: GradientDescentConfig | None = None, **kwargs: Any) -> None:
        if config is not None and kwargs:
            raise ValueError("provide config or kwargs, not both")
        self.config = config or GradientDescentConfig(**kwargs)
        self.current_state: UnknownState | None = None
        self.last_gradient_norm: float | None = None

    def setup(self, control: Any) -> None:
        _ = control
        if self.current_state is None:
            self.last_gradient_norm = None

    def propose(self, control: Any, context: Mapping[str, Any]) -> Sequence[UnknownState]:
        if self.current_state is None:
            self.current_state = control.init_candidate(context)
        return (self.current_state,)

    def update(
        self,
        control: Any,
        states: Sequence[UnknownState],
        feedback: Sequence[Feedback],
        context: Mapping[str, Any],
    ) -> None:
        _ = control
        _ = context
        if not states or not feedback:
            return
        state = states[0]
        fb = feedback[0]
        grad = fb.gradients
        if grad is None:
            if self.config.require_gradient:
                raise ValueError("GradientDescentAdapter requires feedback.gradients")
            return
        grad_arr = np.asarray(grad, dtype=float).reshape(-1)
        values = state.as_array()
        if grad_arr.shape[0] != values.shape[0]:
            raise ValueError(f"gradient dimension {grad_arr.shape[0]} does not match state dimension {values.shape[0]}")
        norm = float(np.linalg.norm(grad_arr))
        self.last_gradient_norm = norm
        if self.config.max_grad_norm is not None and norm > float(self.config.max_grad_norm) and norm > 0.0:
            grad_arr = grad_arr * (float(self.config.max_grad_norm) / norm)
        lr = max(float(self.config.min_learning_rate), float(self.config.learning_rate))
        next_values = values - (lr * grad_arr)
        self.current_state = state.with_values(next_values, adapter=self.name, learning_rate=lr, gradient_norm=norm)

    def get_state(self) -> Mapping[str, Any]:
        return {
            "current_state": None if self.current_state is None else self.current_state.as_array().tolist(),
            "last_gradient_norm": self.last_gradient_norm,
            "learning_rate": float(self.config.learning_rate),
        }

    def get_population(self) -> tuple[UnknownState, ...] | None:
        return None if self.current_state is None else (self.current_state,)

    def set_population(self, population: Sequence[UnknownState]) -> bool:
        states = tuple(population)
        self.current_state = states[0] if states else None
        return True

    def set_state(self, state: Mapping[str, Any]) -> None:
        values = state.get("current_state")
        self.current_state = None if values is None else UnknownState(values=np.asarray(values, dtype=float))
        grad_norm = state.get("last_gradient_norm")
        self.last_gradient_norm = None if grad_norm is None else float(grad_norm)

