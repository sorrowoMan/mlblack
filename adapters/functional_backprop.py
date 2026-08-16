from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from mlblack.core.adapter import OptimizerAdapter
from mlblack.core.backend_session import get_compute_backend_from_context
from mlblack.core.contracts import ComponentContract
from mlblack.core.types import Feedback, UnknownState


@dataclass(frozen=True)
class FunctionalBackpropConfig:
    learning_rate: float = 0.05
    min_learning_rate: float = 1e-8
    max_grad_norm: float | None = 1e3
    random_seed: int = 42


class FunctionalBackpropAdapter(OptimizerAdapter):
    """Functional-gradient optimizer for backends such as JAX.

    The adapter owns the optimizer step, while the problem owns the data and
    exposes a problem-specific functional gradient hook. This keeps the
    boundary explicit: adapters do not read datasets directly, and backends do
    not pretend to expose torch-style backward when they are functional.
    """

    name = "functional_backprop"
    backend_requires = (
        "autograd.functional.grad",
        "autograd.gradients.flat_export",
        "optimizer.sgd_step",
    )
    context_requires = ("candidate.unknown_state", "candidate.model", "backend.contract")
    context_optional = ("feedback.loss", "feedback.metrics")
    context_provides = ("population.candidates", "feedback.gradients")
    context_mutates = ("adapter.current_state",)
    context_cache = ()
    requires_metrics = ("loss",)
    metrics_fallback = "strict"
    context_notes = "Uses a problem-owned functional gradient hook plus backend functional optimizer capabilities."
    contract = ComponentContract(
        name=name,
        requires=context_requires,
        optional=context_optional,
        provides=context_provides,
        mutates=context_mutates,
        supports_gradient=True,
        supports_batch=False,
        supports_resume=True,
        metadata={"family": "neural", "engine": "backend", "gradient_style": "functional", "backend_requires": backend_requires},
    )

    def __init__(self, config: FunctionalBackpropConfig | None = None, **kwargs: Any) -> None:
        if config is not None and kwargs:
            raise ValueError("provide config or kwargs, not both")
        self.config = config or FunctionalBackpropConfig(**kwargs)
        self.backend: Any | None = None
        self.current_state: UnknownState | None = None
        self.last_loss: float | None = None
        self.last_gradient_norm: float | None = None
        self.step_index = 0

    def setup(self, control: Any) -> None:
        if hasattr(control, "require_compute_backend"):
            self.backend = control.require_compute_backend(self.backend_requires, consumer=self.name)
        else:
            self.backend = None
        if self.current_state is None:
            self.last_loss = None
            self.last_gradient_norm = None
            self.step_index = 0

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
        if not states:
            return

        self.backend = get_compute_backend_from_context(context, self.backend_requires, consumer=self.name)
        state = states[0]
        eval_context = self._context_with_backend(context)
        model = control.decode_candidate(state, eval_context)

        problem = getattr(control, "problem", None)
        if problem is None:
            raise ValueError("FunctionalBackpropAdapter requires trainer.problem")
        gradient_fn = getattr(problem, "compute_functional_gradient", None)
        if not callable(gradient_fn):
            raise ValueError(
                "FunctionalBackpropAdapter requires problem.compute_functional_gradient(model, state, context, backend=...). "
                "Use a problem-owned hook so adapters do not read training data directly."
            )

        gradient = gradient_fn(model, state, eval_context, backend=self.backend)
        grad_arr = self.backend.autograd.flat_gradient(gradient)
        values = state.as_array()
        if grad_arr.shape[0] != values.shape[0]:
            raise ValueError(f"gradient dimension {grad_arr.shape[0]} does not match state dimension {values.shape[0]}")

        norm = float(np.linalg.norm(grad_arr))
        update_grad = grad_arr
        if self.config.max_grad_norm is not None and norm > float(self.config.max_grad_norm) and norm > 0.0:
            update_grad = grad_arr * (float(self.config.max_grad_norm) / norm)

        lr = max(float(self.config.min_learning_rate), float(self.config.learning_rate))
        next_values = self.backend.optimizers.sgd_step(values, update_grad, learning_rate=lr)

        fb = feedback[0] if feedback else None
        self.last_loss = None if fb is None or fb.loss is None else float(fb.loss)
        self.last_gradient_norm = norm
        self.current_state = state.with_values(
            next_values,
            adapter=self.name,
            backend=self.backend.contract().name,
            learning_rate=lr,
            gradient_norm=norm,
            gradient_size=int(grad_arr.size),
            gradient_style="functional",
            optimizer="sgd_step",
            optimizer_step=int(self.step_index + 1),
        )
        self.step_index += 1

    def get_state(self) -> Mapping[str, Any]:
        return {
            "current_state": None if self.current_state is None else self.current_state.as_array().tolist(),
            "last_loss": self.last_loss,
            "last_gradient_norm": self.last_gradient_norm,
            "step_index": int(self.step_index),
            "backend": None if self.backend is None else self.backend.contract().name,
            "backend_contract": None if self.backend is None else self.backend.contract().as_dict(),
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
        loss = state.get("last_loss")
        self.last_loss = None if loss is None else float(loss)
        grad_norm = state.get("last_gradient_norm")
        self.last_gradient_norm = None if grad_norm is None else float(grad_norm)
        self.step_index = int(state.get("step_index", self.step_index))

    def _context_with_backend(self, context: Mapping[str, Any]) -> dict[str, Any]:
        ctx = dict(context)
        if self.backend is not None:
            ctx["backend.name"] = self.backend.contract().name
            ctx["backend.contract"] = self.backend.contract().as_dict()
        return ctx


__all__ = ["FunctionalBackpropAdapter", "FunctionalBackpropConfig"]
