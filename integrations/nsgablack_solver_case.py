"""Formal mlblack Trainer to complete nsgablack Solver Case bridge."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np

from blackbase.project import (
    CaseFailure,
    CaseRunRequest,
    CaseRunResult,
    ExecutionControl,
)
from blackbase.resources import DataRef
from blackbase.types import Feedback, PopulationSnapshot, SolverResult, UnknownState


SolverComponentOverrideBuilder = Callable[[Any, Mapping[str, Any]], Mapping[str, Any]]
ParetoSelector = Callable[[PopulationSnapshot, Any, Mapping[str, Any]], int]
DEFAULT_PROJECTABLE_SOLVE_STATUSES = frozenset({"optimal", "feasible"})


@dataclass(frozen=True)
class SolverCaseProjection:
    """ML-facing view created by an explicit optimization-result policy."""

    solver_result: SolverResult
    state: UnknownState | None = None
    feedback: Feedback | None = None
    pareto_front: PopulationSnapshot | None = None
    pareto_front_ref: DataRef | None = None
    artifact_refs: Mapping[str, DataRef] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "artifact_refs", dict(self.artifact_refs or {}))
        object.__setattr__(self, "metadata", dict(self.metadata or {}))


@dataclass(frozen=True)
class SolverCaseInvocationResult:
    """Complete child Case envelope plus optional decoded/projection views."""

    case_result: CaseRunResult
    solver_result: SolverResult | None = None
    projection: SolverCaseProjection | None = None

    @property
    def ok(self) -> bool:
        return self.case_result.ok and self.solver_result is not None

    @property
    def failure(self) -> CaseFailure | None:
        return self.case_result.failure

    def require_solver_result(self) -> SolverResult:
        if self.solver_result is None:
            raise SolverCaseInvocationError(self)
        return self.solver_result

    def require_projection(self) -> SolverCaseProjection:
        if self.projection is None:
            if not self.case_result.ok:
                raise SolverCaseInvocationError(self)
            if self.solver_result is None:
                raise SolverCaseInvocationError(
                    self,
                    message="Solver Case completed without a decodable SolverResult",
                )
            raise SolverCaseInvocationError(
                self,
                message=(
                    "No projection was requested for the completed Solver Case; "
                    "configure an explicit result_projector"
                ),
            )
        return self.projection


class SolverCaseInvocationError(RuntimeError):
    """Optional convenience error that retains the complete invocation result."""

    def __init__(
        self,
        result: SolverCaseInvocationResult,
        *,
        message: str = "",
    ) -> None:
        self.result = result
        self.case_result = result.case_result
        self.failure = result.failure
        detail = message or result.case_result.error or "Solver Case returned no SolverResult"
        super().__init__(detail)


class SolverCaseResultProjector(Protocol):
    """Policy for mapping optimization semantics into an outer Trainer view."""

    def project(
        self,
        result: SolverResult,
        *,
        trainer: Any,
        context: Mapping[str, Any],
    ) -> SolverCaseProjection: ...


class SolverFeedbackMapper(Protocol):
    """Explicit domain mapping from Solver semantics into ML Feedback."""

    def map_feedback(
        self,
        result: SolverResult,
        *,
        state: UnknownState | None,
        pareto_index: int | None,
        trainer: Any,
        context: Mapping[str, Any],
    ) -> Feedback | None: ...


class OptimizationFeedbackMapper:
    """Opt in to treating inner objectives/violations as outer ML Feedback.

    This mapper is intentionally separate from solution selection because an
    optimization objective is not automatically a training loss.
    """

    def map_feedback(
        self,
        result: SolverResult,
        *,
        state: UnknownState | None,
        pareto_index: int | None,
        trainer: Any,
        context: Mapping[str, Any],
    ) -> Feedback | None:
        del state, trainer, context
        if pareto_index is None:
            objectives = result.best_objectives
            constraints = (
                np.zeros(0, dtype=float)
                if result.best_constraint_violation is None
                else np.asarray([result.best_constraint_violation], dtype=float)
            )
            policy = "best_solution"
        else:
            front = result.pareto_front
            if front is None:
                raise ValueError("Pareto Feedback mapping requires an inline Pareto front")
            objectives = np.asarray(front.objectives, dtype=float)[pareto_index]
            if front.constraints is None:
                constraints = np.zeros(0, dtype=float)
            else:
                constraints = np.asarray(front.constraints, dtype=float)[pareto_index]
            policy = "pareto_selector"
        if objectives is None:
            return None
        return Feedback(
            objectives=np.asarray(objectives, dtype=float).reshape(-1),
            constraints=np.asarray(constraints, dtype=float).reshape(-1),
            metrics=_result_metrics(result),
            info={"projection_policy": policy},
        )


class BestSolutionProjector:
    """Project an authoritative best solution selected by the inner Solver."""

    def __init__(
        self,
        *,
        feedback_mapper: SolverFeedbackMapper | Callable[..., Feedback | None] | None = None,
        accepted_statuses: Sequence[str] = tuple(DEFAULT_PROJECTABLE_SOLVE_STATUSES),
    ) -> None:
        self.feedback_mapper = feedback_mapper
        self.accepted_statuses = _normalize_accepted_statuses(accepted_statuses)

    def project(
        self,
        result: SolverResult,
        *,
        trainer: Any,
        context: Mapping[str, Any],
    ) -> SolverCaseProjection:
        _validate_projectable(result, self.accepted_statuses)
        if result.best_solution is None and result.best_solution_ref is None:
            raise ValueError("SolverResult has no authoritative best solution")
        state = _coerce_unknown_state(result.best_solution)
        feedback = _map_feedback(
            self.feedback_mapper,
            result,
            state=state,
            pareto_index=None,
            trainer=trainer,
            context=context,
        )
        return SolverCaseProjection(
            solver_result=result,
            state=state,
            feedback=feedback,
            pareto_front=result.pareto_front,
            pareto_front_ref=result.pareto_front_ref,
            artifact_refs=result.artifact_refs,
            metadata={
                "policy": "best_solution",
                "solution_deferred": state is None and result.best_solution_ref is not None,
                "feedback_mapped": feedback is not None,
            },
        )


class ParetoSolutionProjector:
    """Select one Pareto member through an explicit caller-owned policy."""

    def __init__(
        self,
        selector: ParetoSelector,
        *,
        feedback_mapper: SolverFeedbackMapper | Callable[..., Feedback | None] | None = None,
        accepted_statuses: Sequence[str] = tuple(DEFAULT_PROJECTABLE_SOLVE_STATUSES),
    ) -> None:
        if not callable(selector):
            raise TypeError("selector must be callable")
        self.selector = selector
        self.feedback_mapper = feedback_mapper
        self.accepted_statuses = _normalize_accepted_statuses(accepted_statuses)

    def project(
        self,
        result: SolverResult,
        *,
        trainer: Any,
        context: Mapping[str, Any],
    ) -> SolverCaseProjection:
        _validate_projectable(result, self.accepted_statuses)
        front = result.pareto_front
        if front is None:
            if result.pareto_front_ref is not None:
                raise RuntimeError(
                    "Pareto front is artifact-backed; resolve it through a formal artifact "
                    "provider before applying ParetoSolutionProjector"
                )
            raise ValueError("SolverResult has no Pareto front to select from")
        index = int(self.selector(front, trainer, dict(context)))
        if index < 0 or index >= len(front.candidates):
            raise IndexError(
                f"Pareto selector returned index {index} for front size {len(front.candidates)}"
            )
        state = front.candidates[index]
        feedback = _map_feedback(
            self.feedback_mapper,
            result,
            state=state,
            pareto_index=index,
            trainer=trainer,
            context=context,
        )
        return SolverCaseProjection(
            solver_result=result,
            state=state,
            feedback=feedback,
            pareto_front=front,
            pareto_front_ref=result.pareto_front_ref,
            artifact_refs=result.artifact_refs,
            metadata={
                "policy": "pareto_selector",
                "pareto_index": index,
                "feedback_mapped": feedback is not None,
            },
        )


class ParetoFrontProjector:
    """Keep the complete frontier without silently selecting one solution."""

    def __init__(
        self,
        *,
        accepted_statuses: Sequence[str] = tuple(DEFAULT_PROJECTABLE_SOLVE_STATUSES),
    ) -> None:
        self.accepted_statuses = _normalize_accepted_statuses(accepted_statuses)

    def project(
        self,
        result: SolverResult,
        *,
        trainer: Any,
        context: Mapping[str, Any],
    ) -> SolverCaseProjection:
        del trainer, context
        _validate_projectable(result, self.accepted_statuses)
        if result.pareto_front is None and result.pareto_front_ref is None:
            raise ValueError("SolverResult has no Pareto front")
        return SolverCaseProjection(
            solver_result=result,
            pareto_front=result.pareto_front,
            pareto_front_ref=result.pareto_front_ref,
            artifact_refs=result.artifact_refs,
            metadata={"policy": "pareto_front"},
        )


class NsgablackSolverCaseInvoker:
    """Invoke a complete Solver Case through an outer Trainer's Case runtime."""

    def __init__(
        self,
        case_name: str,
        *,
        stage_name: str = "inner_optimization",
        result_projector: SolverCaseResultProjector
        | Callable[..., SolverCaseProjection]
        | None = None,
        resource_request: Mapping[str, Any] | None = None,
        budget_request: Mapping[str, int] | None = None,
        component_overrides: Mapping[str, Any] | SolverComponentOverrideBuilder | None = None,
        result_key: str = "",
        timeout_seconds: float | None = None,
        control: ExecutionControl | Mapping[str, Any] | None = None,
    ) -> None:
        if not str(case_name or "").strip():
            raise ValueError("case_name must be non-empty")
        self.case_name = str(case_name)
        self.stage_name = str(stage_name or "inner_optimization")
        self.result_projector = result_projector
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
        if timeout_seconds is not None and control is not None:
            raise ValueError("timeout_seconds and control are mutually exclusive")
        if timeout_seconds is not None and float(timeout_seconds) < 0:
            raise ValueError("timeout_seconds must be non-negative")
        self.timeout_seconds = (
            None if timeout_seconds is None else float(timeout_seconds)
        )
        self.control = _coerce_child_control(control)

    def invoke(
        self,
        *,
        trainer: Any,
        inputs: Mapping[str, Any] | None = None,
        input_artifacts: Mapping[str, DataRef | Mapping[str, Any]] | None = None,
        context: Mapping[str, Any] | None = None,
        timeout_seconds: float | None = None,
        control: ExecutionControl | Mapping[str, Any] | None = None,
    ) -> SolverCaseInvocationResult:
        runtime = getattr(trainer, "case_runtime", None)
        invoke = getattr(runtime, "invoke", None)
        if not callable(invoke):
            raise RuntimeError(
                "complete Solver Case invocation requires an injected case_runtime"
            )
        ctx = dict(context or {})
        overrides = self.component_overrides
        if callable(overrides):
            overrides = overrides(trainer, ctx)
        parent_request = getattr(runtime, "request", None)
        project_name = str(getattr(parent_request, "project_name", "") or "")
        if not project_name:
            raise RuntimeError("case_runtime request omitted its parent project_name")
        child_control = self._invocation_control(
            timeout_seconds=timeout_seconds,
            control=control,
        )
        case_result = invoke(
            CaseRunRequest(
                project_name=project_name,
                stage_name=self.stage_name,
                case_name=self.case_name,
                case_kind="solver",
                resource_request=self.resource_request,
                budget_request=self.budget_request,
                component_overrides=dict(overrides or {}),
                input_artifacts=dict(input_artifacts or {}),
                inputs=dict(inputs or {}),
                control=child_control,
                metadata={"bridge": "mlblack->nsgablack.complete_case"},
            )
        )
        if not case_result.ok:
            return SolverCaseInvocationResult(case_result=case_result)

        payload: Any = case_result.output
        if self.result_key:
            payload = case_result.output.get(self.result_key)
        elif str(case_result.output.get("protocol_type", "")) != "blackbase.solver_result":
            payload = case_result.output.get("solver_result", case_result.output.get("result"))
        if isinstance(payload, SolverResult):
            solver_result = payload
        elif isinstance(payload, Mapping):
            solver_result = SolverResult.from_dict(payload)
        else:
            raise TypeError(
                f"inner Solver Case '{self.case_name}' did not return a SolverResult codec payload"
            )
        merged_refs = {
            **dict(solver_result.artifact_refs),
            **dict(case_result.artifact_refs),
        }
        if merged_refs != dict(solver_result.artifact_refs):
            solver_result = replace(solver_result, artifact_refs=merged_refs)
        projection = None
        if self.result_projector is not None:
            projection = self._project(solver_result, trainer=trainer, context=ctx)
        return SolverCaseInvocationResult(
            case_result=case_result,
            solver_result=solver_result,
            projection=projection,
        )

    solve = invoke

    def _invocation_control(
        self,
        *,
        timeout_seconds: float | None,
        control: ExecutionControl | Mapping[str, Any] | None,
    ) -> ExecutionControl:
        if timeout_seconds is not None and control is not None:
            raise ValueError("timeout_seconds and control are mutually exclusive")
        if timeout_seconds is not None:
            if float(timeout_seconds) < 0:
                raise ValueError("timeout_seconds must be non-negative")
            return ExecutionControl.with_timeout(
                float(timeout_seconds),
                namespace="mlblack.nsgablack_solver_case",
                metadata={"bridge": "mlblack->nsgablack.complete_case"},
            )
        if control is not None:
            normalized = _coerce_child_control(control)
            assert normalized is not None
            return normalized
        if self.control is not None:
            return self.control
        if self.timeout_seconds is not None:
            return ExecutionControl.with_timeout(
                self.timeout_seconds,
                namespace="mlblack.nsgablack_solver_case",
                metadata={"bridge": "mlblack->nsgablack.complete_case"},
            )
        return ExecutionControl()

    def _project(
        self,
        result: SolverResult,
        *,
        trainer: Any,
        context: Mapping[str, Any],
    ) -> SolverCaseProjection:
        project = getattr(self.result_projector, "project", None)
        if callable(project):
            projection = project(result, trainer=trainer, context=context)
        elif callable(self.result_projector):
            projection = self.result_projector(
                result,
                trainer=trainer,
                context=context,
            )
        else:
            raise TypeError("result_projector must be callable or implement project()")
        if not isinstance(projection, SolverCaseProjection):
            raise TypeError("result_projector must return SolverCaseProjection")
        return projection


def _normalize_accepted_statuses(values: Sequence[str]) -> frozenset[str]:
    statuses = frozenset(str(value).strip().lower() for value in values)
    if not statuses:
        raise ValueError("accepted_statuses must be non-empty")
    return statuses


def _coerce_child_control(
    value: ExecutionControl | Mapping[str, Any] | None,
) -> ExecutionControl | None:
    if value is None:
        return None
    control = value if isinstance(value, ExecutionControl) else ExecutionControl.from_dict(value)
    if control.ancestor_cancellations:
        raise ValueError(
            "child control cannot supply ancestor_cancellations; the parent Case owns lineage"
        )
    return control


def _validate_projectable(result: SolverResult, accepted_statuses: frozenset[str]) -> None:
    if result.solve_status not in accepted_statuses:
        raise ValueError(
            f"SolverResult solve_status='{result.solve_status}' is not projectable; "
            f"accepted={sorted(accepted_statuses)}"
        )


def _map_feedback(
    mapper: SolverFeedbackMapper | Callable[..., Feedback | None] | None,
    result: SolverResult,
    *,
    state: UnknownState | None,
    pareto_index: int | None,
    trainer: Any,
    context: Mapping[str, Any],
) -> Feedback | None:
    if mapper is None:
        return None
    method = getattr(mapper, "map_feedback", None)
    if callable(method):
        feedback = method(
            result,
            state=state,
            pareto_index=pareto_index,
            trainer=trainer,
            context=context,
        )
    elif callable(mapper):
        feedback = mapper(
            result,
            state=state,
            pareto_index=pareto_index,
            trainer=trainer,
            context=context,
        )
    else:
        raise TypeError("feedback_mapper must be callable or implement map_feedback()")
    if feedback is not None and not isinstance(feedback, Feedback):
        raise TypeError("feedback_mapper must return Feedback or None")
    return feedback


def _coerce_unknown_state(value: Any) -> UnknownState | None:
    if value is None:
        return None
    if isinstance(value, UnknownState):
        return value
    if isinstance(value, Mapping):
        if "values" not in value:
            raise TypeError(
                "best solution mapping must expose 'values' or use a custom projector"
            )
        return UnknownState.from_protocol_payload(value)
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "best solution is not numeric; use a custom SolverCaseResultProjector"
        ) from exc
    return UnknownState(array.reshape(-1), metadata={"source": "inner_solver"})


def _result_metrics(result: SolverResult) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "solve_status": result.solve_status,
        "termination_reason": result.termination_reason,
        "feasibility": result.feasibility,
        "approximate": result.quality.approximate,
    }
    for key in ("generation", "steps", "evaluation_count", "elapsed_sec"):
        if key in result.report:
            metrics[key] = result.report[key]
    for key in ("absolute_gap", "relative_gap", "bound"):
        value = getattr(result.quality, key)
        if value is not None:
            metrics[key] = value
    metrics.update(dict(result.quality.metrics))
    return metrics


__all__ = [
    "BestSolutionProjector",
    "DEFAULT_PROJECTABLE_SOLVE_STATUSES",
    "NsgablackSolverCaseInvoker",
    "OptimizationFeedbackMapper",
    "ParetoFrontProjector",
    "ParetoSelector",
    "ParetoSolutionProjector",
    "SolverCaseInvocationError",
    "SolverCaseInvocationResult",
    "SolverCaseProjection",
    "SolverCaseResultProjector",
    "SolverComponentOverrideBuilder",
    "SolverFeedbackMapper",
]
