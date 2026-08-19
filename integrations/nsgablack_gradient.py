"""Formal assembly of nsgablack gradient strategy with mlblack semantics."""

from __future__ import annotations

from typing import Any, Mapping

from blackbase.resources import ResourceContext
from mlblack.backends.torch_neural import (
    TorchEvaluationProvider,
    TorchEvaluationProviderConfig,
)
from mlblack.core import (
    ComputeBackendSpec,
    FunctionalGradientLearningProblem,
    ProviderBackedLearningProblem,
)
from mlblack.pipeline.datasets import NumericBatchSchedule
from .nsgablack_control import LearningSolver
from .nsgablack_optimization import build_optimization_adapter


def build_gradient_trainer(
    *,
    problem: Any,
    representation: Any,
    method: str = "gradient.adamw",
    compute_backend: str = "torch",
    learning_rate: float = 1e-3,
    min_learning_rate: float = 1e-9,
    weight_decay: float = 0.0,
    max_gradient_norm: float | None = 10.0,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1e-8,
    data_schedule: NumericBatchSchedule | None = None,
    resource_context: Mapping[str, Any] | ResourceContext | None = None,
    provider_config: TorchEvaluationProviderConfig | None = None,
    run_name: str = "torch_gradient_trainer",
) -> LearningSolver:
    """Build the stable method -> Adapter -> Provider -> L0 runtime path.

    ``compute_backend`` selects an ML evaluation implementation, not an
    optimization algorithm.  The stable ``gradient.*`` method remains owned
    by nsgablack regardless of which Provider executes the loss/autograd work.
    """

    method_id = _gradient_method_id(method)
    backend_name = str(compute_backend or "torch").strip().lower()
    backend_aliases = {
        "analytic": "problem",
        "inline": "problem",
        "native": "problem",
        "numpy": "problem",
    }
    backend_name = backend_aliases.get(backend_name, backend_name)
    if backend_name not in {"torch", "jax", "tensorflow", "problem"}:
        raise ValueError(
            f"no unified gradient Evaluation Provider is registered for "
            f"compute_backend={backend_name!r}; currently available: "
            "torch, jax, tensorflow, problem"
        )
    provider = None
    trainer_problem = problem
    state_gateway = None
    use_provider_transition = False
    compute_spec = ComputeBackendSpec(
        name="numpy",
        device="cpu",
        metadata={"evaluation_owner": "problem"},
    )
    if backend_name == "torch":
        effective_provider_config = provider_config or TorchEvaluationProviderConfig(
            publish_state_refs=True,
        )
        if (
            effective_provider_config.publish_state_refs
            and not effective_provider_config.inline_gradients
        ):
            raise ValueError(
                "the unified gradient Trainer currently requires inline_gradients=True "
                "when publish_state_refs=True so its checkpoint contains an exact "
                "optimizer-state shadow"
            )
        provider = TorchEvaluationProvider(
            problem,
            representation,
            data_schedule=data_schedule,
            config=effective_provider_config,
        )
        trainer_problem = ProviderBackedLearningProblem.from_provider(problem, provider)
        state_gateway = trainer_problem.gateway
        use_provider_transition = bool(effective_provider_config.publish_state_refs)
        compute_spec = ComputeBackendSpec(name="torch", device="cpu")
    elif backend_name in {"jax", "tensorflow"}:
        if provider_config is not None or data_schedule is not None:
            raise ValueError(
                "provider_config and data_schedule currently belong to the "
                "Torch Evaluation Provider"
            )
        trainer_problem = FunctionalGradientLearningProblem(problem)
        compute_spec = ComputeBackendSpec(name=backend_name, device="cpu")
    elif provider_config is not None or data_schedule is not None:
        raise ValueError(
            "provider_config and data_schedule require compute_backend='torch'; "
            "the problem route evaluates gradients directly"
        )
    adapter = build_optimization_adapter(
        method_id,
        learning_rate=float(learning_rate),
        min_learning_rate=float(min_learning_rate),
        weight_decay=float(weight_decay),
        max_gradient_norm=max_gradient_norm,
        beta1=float(beta1),
        beta2=float(beta2),
        epsilon=float(epsilon),
        state_gateway=state_gateway,
        prefer_provider_transition=use_provider_transition,
    )
    trainer = LearningSolver(
        problem=trainer_problem,
        representation=representation,
        adapter=adapter,
        run_name=run_name,
        resource_context=resource_context,
        compute_backend=compute_spec,
    )
    # Explicitly named public evidence; these are semantic/runtime components,
    # not a second resource allocator.
    if provider is not None:
        trainer.evaluation_provider = provider
        trainer.data_schedule = data_schedule
    trainer.optimizer_method = method_id
    trainer.evaluation_mode = backend_name
    return trainer


def _gradient_method_id(value: str) -> str:
    normalized = str(value or "gradient.adamw").strip().lower()
    aliases = {
        "sgd": "gradient.sgd",
        "gd": "gradient.sgd",
        "gradient_descent": "gradient.sgd",
        "adam": "gradient.adam",
        "adamw": "gradient.adamw",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in {
        "gradient.sgd",
        "gradient.adam",
        "gradient.adamw",
    }:
        raise ValueError(
            "method must be gradient.sgd, gradient.adam, or gradient.adamw"
        )
    return normalized


__all__ = ["build_gradient_trainer"]
