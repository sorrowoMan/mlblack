from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from mlblack.core.adapter import OptimizerAdapter
from mlblack.core.backend_session import get_compute_backend_from_context
from mlblack.core.contracts import ComponentContract
from mlblack.core.types import Feedback, UnknownState


@dataclass(frozen=True)
class NeuralGraphBackpropConfig:
    optimizer: str = "adamw"
    learning_rate: float = 1e-3
    min_learning_rate: float = 1e-9
    weight_decay: float = 0.0
    max_grad_norm: float | None = 10.0
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    random_seed: int = 42


class NeuralGraphBackpropAdapter(OptimizerAdapter):
    """Backend-dispatched neural graph backprop adapter.

    The adapter owns optimizer/update semantics. Concrete tensor operations,
    autograd, optimizer construction, and parameter export are delegated to the
    selected backend capability components.
    """

    name = "neural_graph_backprop"
    backend_requires = (
        "autograd.mode.train",
        "autograd.backward",
        "autograd.zero_grad",
        "autograd.gradients.flat_export",
        "optimizer.build",
        "optimizer.step",
        "parameters.flat_export",
        "parameters.state_json",
    )
    context_requires = ("candidate.unknown_state", "candidate.model", "neural.graph_spec", "backend.contract")
    context_optional = ("resource.device", "resource.context", "feedback.loss", "feedback.metrics")
    context_provides = ("population.candidates", "feedback.gradients", "neural.optimizer_state")
    context_mutates = ("adapter.current_state", "neural.optimizer_state")
    context_cache = ("neural.optimizer_state",)
    requires_metrics = ("objective", "loss")
    metrics_fallback = "strict"
    context_notes = "Uses selected backend autograd/optimizer capabilities to update a decoded neural graph module."
    contract = ComponentContract(
        name=name,
        requires=context_requires,
        optional=context_optional,
        provides=context_provides,
        mutates=context_mutates,
        cache=context_cache,
        supports_gradient=True,
        supports_batch=False,
        supports_resume=True,
        metadata={"family": "neural", "engine": "backend", "route": "neural_graph", "backend_requires": backend_requires},
    )

    def __init__(self, config: NeuralGraphBackpropConfig | None = None, **kwargs: Any) -> None:
        if config is not None and kwargs:
            raise ValueError("provide config or kwargs, not both")
        self.config = config or NeuralGraphBackpropConfig(**kwargs)
        self.backend: Any | None = None
        self.current_state: UnknownState | None = None
        self.last_loss: float | None = None
        self.last_gradient_norm: float | None = None
        self.step_index = 0
        self.actual_device = "cpu"
        self.optimizer_state_dict: Mapping[str, Any] | None = None

    def setup(self, trainer: Any) -> None:
        if hasattr(trainer, "require_compute_backend"):
            self.backend = trainer.require_compute_backend(self.backend_requires, consumer=self.name)
        else:
            self.backend = None
        if self.current_state is None:
            self.last_loss = None
            self.last_gradient_norm = None
            self.step_index = 0
            self.optimizer_state_dict = None

    def propose(self, trainer: Any, context: Mapping[str, Any]) -> Sequence[UnknownState]:
        if self.current_state is None:
            self.current_state = trainer.init_candidate(context)
        return (self.current_state,)

    def update(
        self,
        trainer: Any,
        states: Sequence[UnknownState],
        feedback: Sequence[Feedback],
        context: Mapping[str, Any],
    ) -> None:
        _ = feedback
        if not states:
            return

        self.backend = get_compute_backend_from_context(context, self.backend_requires, consumer=self.name)
        torch = getattr(self.backend.tensor, "torch", lambda: None)()
        if torch is not None and hasattr(torch, "manual_seed"):
            torch.manual_seed(int(self.config.random_seed) + int(self.step_index))

        state = states[0]
        model = trainer.decode_candidate(state, self._context_with_backend(context))
        strict = str(context.get("backend.device_policy", "fallback_cpu")).lower() == "strict"
        device = self.backend.tensor.device(context, fallback=str(context.get("backend.device", "cpu")), strict=strict)
        self.actual_device = str(device)
        self.backend.autograd.train(model, device=device)
        optimizer = self.backend.optimizers.build_optimizer(model, self.config)
        if self.optimizer_state_dict:
            optimizer.load_state_dict(self.backend.autograd.optimizer_state_to_device(self.optimizer_state_dict, device))
        self.backend.optimizers.zero_grad(optimizer)

        problem = getattr(trainer, "problem", None)
        if problem is None:
            raise ValueError("NeuralGraphBackpropAdapter requires trainer.problem")
        compute_backend_loss = getattr(problem, "compute_backend_loss", None)
        if not callable(compute_backend_loss):
            raise ValueError(
                "NeuralGraphBackpropAdapter requires problem.compute_backend_loss(...). "
                "Problem.evaluate(...) is reserved for no-backward evaluation."
            )
        eval_context = self._context_with_backend(context)
        eval_context["resource.device"] = self.actual_device
        evaluation = compute_backend_loss(model, state, eval_context, differentiable=True)
        if getattr(evaluation, "loss", None) is None:
            raise ValueError("problem.compute_backend_loss(...) must return a backend-native loss object")
        self.backend.autograd.backward(evaluation.loss)
        gradients = self.backend.autograd.flat_grads(model)

        params = self.backend.autograd.trainable_parameters(model)
        if self.config.max_grad_norm is None:
            grad_norm = self.backend.autograd.grad_norm(params)
        else:
            grad_norm = self.backend.autograd.clip_grad_norm(params, float(self.config.max_grad_norm))
        self.backend.optimizers.step(optimizer)

        next_values = self.backend.autograd.flat_parameters(model)
        self.optimizer_state_dict = self.backend.autograd.optimizer_state_to_cpu(optimizer.state_dict())
        self.last_loss = float(evaluation.loss_value)
        self.last_gradient_norm = float(grad_norm)
        self.current_state = state.with_values(
            next_values,
            adapter=self.name,
            backend=self.backend.contract().name,
            learning_rate=max(float(self.config.min_learning_rate), float(self.config.learning_rate)),
            gradient_norm=float(grad_norm),
            gradient_size=int(np.asarray(gradients, dtype=float).size),
            backend_loss=self.last_loss,
            optimizer=str(self.config.optimizer),
            optimizer_step=int(self.step_index + 1),
            actual_device=str(self.actual_device),
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
            "optimizer": str(self.config.optimizer),
            "optimizer_state": {}
            if self.backend is None
            else self.backend.autograd.jsonable_torch_state(self.optimizer_state_dict or {}),
            "learning_rate": float(self.config.learning_rate),
            "weight_decay": float(self.config.weight_decay),
            "actual_device": str(self.actual_device),
        }

    def set_state(self, state: Mapping[str, Any]) -> None:
        values = state.get("current_state")
        self.current_state = None if values is None else UnknownState(values=np.asarray(values, dtype=float))
        loss = state.get("last_loss")
        self.last_loss = None if loss is None else float(loss)
        grad_norm = state.get("last_gradient_norm")
        self.last_gradient_norm = None if grad_norm is None else float(grad_norm)
        self.step_index = int(state.get("step_index", self.step_index))
        opt_state = state.get("optimizer_state")
        if isinstance(opt_state, Mapping) and self.backend is not None:
            self.optimizer_state_dict = self.backend.autograd.restore_jsonable_torch_state(opt_state)
        self.actual_device = str(state.get("actual_device", self.actual_device))

    def _context_with_backend(self, context: Mapping[str, Any]) -> dict[str, Any]:
        ctx = dict(context)
        if self.backend is not None:
            ctx["backend.name"] = self.backend.contract().name
            ctx["backend.contract"] = self.backend.contract().as_dict()
        return ctx


__all__ = ["NeuralGraphBackpropAdapter", "NeuralGraphBackpropConfig"]
