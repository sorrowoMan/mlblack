"""LearningProblem facade backed by the shared BlackBase evaluation gateway."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from blackbase.evaluation import (
    EvaluationGateway,
    EvaluationProviderRegistry,
    EvaluationRequest,
)
from blackbase.resources import ResourceContext
from blackbase.state_ref import StateRef
from blackbase.types import Feedback, UnknownState

from .problem import LearningProblem


class FunctionalGradientLearningProblem(LearningProblem):
    """Attach backend-functional gradients to an ML semantic Problem.

    JAX and TensorFlow expose functional differentiation rather than a
    Torch-style mutable ``backward()`` loop.  The gradient is evaluation
    evidence, so it is produced here at the Problem/Provider boundary and is
    then consumed by NSGABlack's provider-neutral ``gradient.*`` Adapter.
    """

    backend_requires = (
        "autograd.functional.grad",
        "autograd.gradients.flat_export",
    )
    thread_safe_evaluation = False

    def __init__(self, semantic_problem: LearningProblem) -> None:
        gradient = getattr(semantic_problem, "compute_functional_gradient", None)
        if not callable(gradient):
            raise TypeError(
                "functional-gradient execution requires "
                "problem.compute_functional_gradient(model, state, context, backend=...)"
            )
        self.semantic_problem = semantic_problem
        self.name = str(getattr(semantic_problem, "name", "functional_gradient_problem"))
        if hasattr(semantic_problem, "data"):
            self.data = semantic_problem.data

    def evaluate(
        self,
        model: Any,
        state: UnknownState,
        context: Mapping[str, Any],
    ) -> Feedback:
        from .backend_session import get_compute_backend_from_context

        feedback = self.semantic_problem.evaluate(model, state, context)
        if not isinstance(feedback, Feedback):
            raise TypeError("semantic Problem.evaluate(...) must return Feedback")
        backend = get_compute_backend_from_context(
            context,
            self.backend_requires,
            consumer=f"{self.name}.functional_gradient",
        )
        gradient = self.semantic_problem.compute_functional_gradient(
            model,
            state,
            context,
            backend=backend,
        )
        flat = np.asarray(backend.autograd.flat_gradient(gradient), dtype=float).reshape(-1)
        if flat.shape != state.as_array().shape:
            raise ValueError(
                "functional gradient dimension must match candidate state: "
                f"gradient={flat.shape}, candidate={state.as_array().shape}"
            )
        return Feedback(
            objectives=np.asarray(feedback.objectives, dtype=float),
            constraints=np.asarray(feedback.constraints, dtype=float),
            gradients=flat,
            gradient_ref=feedback.gradient_ref,
            loss=feedback.loss,
            metrics=dict(feedback.metrics or {}),
            residuals=(
                None
                if feedback.residuals is None
                else np.asarray(feedback.residuals, dtype=float)
            ),
            signals={
                **dict(feedback.signals or {}),
                "has_gradient": True,
                "gradient_owner": "mlblack.problem_provider",
                "gradient_backend": backend.contract().name,
            },
            info=dict(feedback.info or {}),
        )

    def prepare_model_for_evaluation(
        self,
        model: Any,
        state: UnknownState,
        context: Mapping[str, Any],
    ) -> Any:
        return self.semantic_problem.prepare_model_for_evaluation(
            model,
            state,
            context,
        )

    def describe(self) -> Mapping[str, Any]:
        describe = getattr(self.semantic_problem, "describe", None)
        semantic = dict(describe()) if callable(describe) else {"name": self.name}
        return {
            **semantic,
            "gradient_owner": "mlblack.problem_provider",
            "gradient_style": "functional",
        }

    def get_num_objectives(self) -> int:
        getter = getattr(self.semantic_problem, "get_num_objectives", None)
        if callable(getter):
            return int(getter())
        return int(getattr(self.semantic_problem, "objective_count", 1) or 1)

    def build_model_artifact(
        self,
        model: Any,
        context: Mapping[str, Any] | None = None,
    ) -> Any:
        materialize = getattr(self.semantic_problem, "build_model_artifact", None)
        if not callable(materialize):
            return model
        return materialize(model, dict(context or {}))


class ProviderBackedLearningProblem(LearningProblem):
    """Keep ML objective semantics while delegating compute to a Provider.

    The wrapped semantic Problem still defines objectives, metrics and loss.
    The gateway only selects an implementation inside the already-authorized L0
    grant. No ML control class or Provider is allowed to choose a device here.
    """

    name = "provider_backed_learning_problem"
    backend_requires: tuple[str, ...] = ()
    thread_safe_evaluation = False

    def __init__(
        self,
        semantic_problem: LearningProblem,
        gateway: EvaluationGateway,
        *,
        problem_id: str,
        capabilities: Sequence[str],
        provider: Any | None = None,
    ) -> None:
        self.semantic_problem = semantic_problem
        self.gateway = gateway
        self.problem_id = str(problem_id)
        self.capabilities = tuple(str(value) for value in capabilities)
        self.provider = provider
        self.name = str(getattr(semantic_problem, "name", self.name))
        if hasattr(semantic_problem, "data"):
            self.data = semantic_problem.data

    @classmethod
    def from_provider(
        cls,
        semantic_problem: LearningProblem,
        provider: Any,
        *,
        registry: EvaluationProviderRegistry | None = None,
    ) -> "ProviderBackedLearningProblem":
        active_registry = registry or EvaluationProviderRegistry()
        active_registry.register(provider)
        problem_id = str(getattr(provider, "problem_id", "") or "")
        capabilities = tuple(getattr(provider, "request_capabilities", ()) or ())
        if not problem_id or not capabilities:
            raise TypeError(
                "provider must expose problem_id and request_capabilities"
            )
        return cls(
            semantic_problem,
            EvaluationGateway(active_registry),
            problem_id=problem_id,
            capabilities=capabilities,
            provider=provider,
        )

    def evaluate(
        self,
        model: Any,
        state: UnknownState,
        context: Mapping[str, Any],
    ) -> Feedback:
        _ = model
        mode = str(context.get("evaluation.mode", "train") or "train").lower()
        request = EvaluationRequest(
            problem_id=self.problem_id,
            states=(state,),
            mode=mode,
            capabilities=self.capabilities,
            payload={
                "step": int(context.get("step", 0) or 0),
                "run_name": str(context.get("run_name", "")),
                "trajectory_id": str(
                    context.get("trajectory_id", context.get("run_name", ""))
                ),
            },
            metadata={"semantic_problem": self.name},
        )
        resource = ResourceContext.from_mapping(
            context.get("resource.context", context.get("resource_context", {}))
        )
        result = self.gateway.evaluate(request, resource)
        if len(result.feedback) != 1:
            raise RuntimeError("single-state provider evaluation returned wrong cardinality")
        feedback = result.feedback[0]
        binding = result.binding
        info = dict(feedback.info or {})
        if binding is not None:
            info["evaluation_binding"] = {
                "binding_id": binding.binding_id,
                "provider_id": binding.provider_id,
                "device": binding.device,
                "compute_backend": binding.compute_backend,
                "degraded": bool(binding.degraded),
                "grant_namespace": binding.resource_context.namespace,
            }
        if result.result_states and isinstance(result.result_states[0], StateRef):
            info["evaluation_state_ref"] = result.result_states[0]
        return Feedback(
            objectives=np.array(feedback.objectives, dtype=float, copy=True),
            constraints=np.array(feedback.constraints, dtype=float, copy=True),
            gradients=(
                None
                if feedback.gradients is None
                else np.array(feedback.gradients, dtype=float, copy=True)
            ),
            gradient_ref=feedback.gradient_ref,
            loss=None if feedback.loss is None else float(feedback.loss),
            metrics=dict(feedback.metrics or {}),
            residuals=(
                None
                if feedback.residuals is None
                else np.array(feedback.residuals, dtype=float, copy=True)
            ),
            signals=dict(feedback.signals or {}),
            info=info,
        )

    def prepare_model_for_evaluation(
        self,
        model: Any,
        state: UnknownState,
        context: Mapping[str, Any],
    ) -> Any:
        return self.semantic_problem.prepare_model_for_evaluation(
            model,
            state,
            context,
        )

    def describe(self) -> Mapping[str, Any]:
        describe = getattr(self.semantic_problem, "describe", None)
        semantic = dict(describe()) if callable(describe) else {"name": self.name}
        provider_spec = getattr(self.provider, "spec", None)
        provider_state = getattr(self.provider, "get_state", None)
        return {
            **semantic,
            "evaluation_problem_id": self.problem_id,
            "evaluation_capabilities": self.capabilities,
            "evaluation_gateway": "blackbase",
            "evaluation_provider": (
                None
                if provider_spec is None
                else provider_spec.as_dict()
            ),
            "evaluation_provider_state": (
                None
                if not callable(provider_state)
                else dict(provider_state())
            ),
        }

    def get_num_objectives(self) -> int:
        getter = getattr(self.semantic_problem, "get_num_objectives", None)
        if callable(getter):
            return int(getter())
        return 1

    def build_model_artifact(
        self,
        model: Any,
        context: Mapping[str, Any] | None = None,
    ) -> Any:
        """Preserve the wrapped Problem's model publication semantics.

        Provider-backed execution changes where loss/autograd runs; it must
        not erase the semantic Problem's typed Artifact builder.
        """

        materialize = getattr(self.semantic_problem, "build_model_artifact", None)
        if not callable(materialize):
            return model
        return materialize(model, dict(context or {}))


__all__ = ["FunctionalGradientLearningProblem", "ProviderBackedLearningProblem"]
