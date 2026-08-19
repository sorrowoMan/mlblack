"""NSGABlack outer search to a complete ML semantic child Case."""

from __future__ import annotations

from typing import Any, Callable, Mapping

import numpy as np

from blackbase.project import CaseRunRequest
from mlblack.core.types import TrainerResult, UnknownState


ResultProjector = Callable[
    [TrainerResult, UnknownState, Mapping[str, Any]],
    tuple[np.ndarray, float],
]
ComponentOverrideBuilder = Callable[
    [UnknownState, Mapping[str, Any]],
    Mapping[str, Any],
]


class NsgablackLearningCaseEvaluator:
    """Invoke a complete ML semantic Case through the shared substrate.

    This bridge never constructs a private Trainer, derives a ResourceContext,
    or owns another lifecycle. The parent CaseRuntime owns lineage,
    cancellation, budget delegation, and the bounded child resource grant.
    """

    def __init__(
        self,
        case_name: str,
        *,
        stage_name: str = "inner_learning",
        max_steps: int = 1,
        result_projector: ResultProjector | None = None,
        resource_request: Mapping[str, Any] | None = None,
        budget_request: Mapping[str, int] | None = None,
        component_overrides: Mapping[str, Any] | ComponentOverrideBuilder | None = None,
        result_key: str = "",
    ) -> None:
        if not str(case_name or "").strip():
            raise ValueError("case_name must be non-empty")
        self.case_name = str(case_name)
        self.stage_name = str(stage_name or "inner_learning")
        self.max_steps = max(1, int(max_steps))
        self.result_projector = result_projector or project_learning_result
        self.resource_request = dict(
            resource_request
            or {"workers": 1, "threads": 1, "memory_mb": 512}
        )
        self.budget_request = {
            str(key): int(value)
            for key, value in dict(budget_request or {}).items()
        }
        self.component_overrides = component_overrides
        self.result_key = str(result_key or "")

    def can_handle(self, *, solver: Any, x: np.ndarray) -> bool:
        del x
        return callable(getattr(getattr(solver, "case_runtime", None), "invoke", None))

    def evaluate(
        self,
        *,
        solver: Any,
        x: np.ndarray,
        individual_id: int,
        context: Mapping[str, Any] | None = None,
    ) -> tuple[np.ndarray, float]:
        runtime = getattr(solver, "case_runtime", None)
        invoke = getattr(runtime, "invoke", None)
        if not callable(invoke):
            raise RuntimeError(
                "complete ML Case evaluation requires an injected case_runtime"
            )
        ctx = dict(context or {})
        candidate = UnknownState(
            values=np.asarray(x, dtype=float).reshape(-1),
            metadata={
                "outer_individual_id": int(individual_id),
                "outer_generation": int(getattr(solver, "generation", 0)),
            },
        )
        overrides = self.component_overrides
        if callable(overrides):
            overrides = overrides(candidate, ctx)
        parent_request = getattr(runtime, "request", None)
        project_name = str(getattr(parent_request, "project_name", "") or "")
        if not project_name:
            raise RuntimeError("case_runtime request omitted its parent project_name")
        result = invoke(
            CaseRunRequest(
                project_name=project_name,
                stage_name=self.stage_name,
                case_name=self.case_name,
                case_kind="trainer",
                resource_request=self.resource_request,
                budget_request=self.budget_request,
                component_overrides=dict(overrides or {}),
                inputs={
                    "candidate": candidate.to_protocol_payload(),
                    "max_steps": self.max_steps,
                    "outer": {
                        "individual_id": int(individual_id),
                        "generation": int(getattr(solver, "generation", 0)),
                    },
                },
                metadata={"bridge": "nsgablack->mlblack.complete_learning_case"},
            )
        )
        if not result.ok:
            raise RuntimeError(
                f"inner ML Case '{self.case_name}' failed: {result.error}"
            )
        payload: Any = result.output
        if self.result_key:
            payload = result.output.get(self.result_key)
        elif str(result.output.get("protocol_type", "")) != "blackbase.trainer_result":
            payload = result.output.get("trainer_result", result.output.get("result"))
        if not isinstance(payload, Mapping):
            raise TypeError(
                f"inner ML Case '{self.case_name}' did not return a TrainerResult codec payload"
            )
        learning_result = TrainerResult.from_dict(payload)
        return self.result_projector(learning_result, candidate, ctx)


def project_learning_result(
    result: TrainerResult,
    candidate: UnknownState,
    context: Mapping[str, Any],
) -> tuple[np.ndarray, float]:
    """Project a semantic learning result into outer objectives/violation."""

    del candidate, context
    feedback = result.best_feedback
    objectives = result.best_objectives
    if objectives is None and feedback is not None:
        objectives = getattr(feedback, "objectives", None)
    if objectives is None:
        raise ValueError("TrainerResult has no best objectives to project")
    objective_array = np.asarray(objectives, dtype=float).reshape(-1)
    if objective_array.size == 0:
        raise ValueError("TrainerResult best objectives must not be empty")
    constraints = getattr(feedback, "constraints", ()) if feedback is not None else ()
    constraint_array = np.asarray(constraints, dtype=float).reshape(-1)
    violation = float(np.sum(np.maximum(0.0, constraint_array)))
    return objective_array, violation


__all__ = [
    "ComponentOverrideBuilder",
    "NsgablackLearningCaseEvaluator",
    "ResultProjector",
    "project_learning_result",
]
