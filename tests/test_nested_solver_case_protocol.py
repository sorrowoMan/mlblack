from __future__ import annotations

from pathlib import Path
import time

import pytest

from blackbase.project import (
    CaseFailure,
    CaseRunRequest,
    CaseRunResult,
    ExecutionControl,
    execute_project,
)
from blackbase.project.scaffold import add_case, create_project
from blackbase.types import PopulationSnapshot, SolverResult, UnknownState

import numpy as np

from mlblack.integrations.nsgablack_solver_case import (
    BestSolutionProjector,
    NsgablackSolverCaseInvoker,
    OptimizationFeedbackMapper,
    ParetoSolutionProjector,
    SolverCaseInvocationError,
)


def test_complete_solver_case_bridge_projects_into_outer_trainer(tmp_path) -> None:
    project_root = create_project(tmp_path / "solver_case_project", framework="blackbase")
    child_root = add_case("inner_solver", "solver", project_root=project_root)
    parent_root = add_case("outer_trainer", "trainer", project_root=project_root)
    (child_root / "build_solver.py").write_text(
        """
import numpy as np
from nsgablack.core import BlackBoxProblem, SolverBase


class Problem(BlackBoxProblem):
    def __init__(self):
        super().__init__(dimension=2, objectives=("cost", "risk"))

    def evaluate(self, candidate, context=None):
        del context
        return np.asarray(candidate, dtype=float)


class InnerSolver(SolverBase):
    def __init__(self, resource_context):
        super().__init__(Problem(), resource_context=resource_context)

    def run(self):
        self.population = np.asarray([[1.0, 2.0], [2.0, 1.0]])
        self.objectives = np.asarray([[0.2, 0.8], [0.5, 0.4]])
        self.constraint_violations = np.asarray([0.0, 0.1])
        self.best_x = self.population[0]
        self.best_objectives = self.objectives[0]
        self.best_constraint_violation = 0.0
        self.pareto_solutions = {
            "individuals": self.population,
            "objectives": self.objectives,
        }
        self.pareto_objectives = self.objectives
        self.generation = 3
        self.evaluation_count = 8
        return {
            "status": "ok",
            "solve_status": "feasible",
            "termination_reason": "iteration_limit",
            "feasibility": "feasible",
            "quality": {"approximate": True},
            "generation": 3,
            "evaluation_count": 8,
        }


def build_solver(config=None, *, resource_context=None, component_overrides=None):
    del config, component_overrides
    return InnerSolver(resource_context)
""".lstrip(),
        encoding="utf-8",
    )
    (parent_root / "build_solver.py").write_text(
        """
from blackbase.types import TrainerResult
from mlblack.core import BlankTrainer
from mlblack.integrations import (
    BestSolutionProjector,
    NsgablackSolverCaseInvoker,
    OptimizationFeedbackMapper,
)


class OuterTrainer(BlankTrainer):
    def fit(self):
        invocation = NsgablackSolverCaseInvoker(
            "inner_solver",
            result_projector=BestSolutionProjector(
                feedback_mapper=OptimizationFeedbackMapper(),
            ),
            resource_request={
                "workers": 1,
                "threads": 1,
                "memory_mb": 256,
                "backend": "local",
                "compute_backend": "auto",
                "device": "cpu",
            },
        ).invoke(trainer=self, inputs={"source": "outer_trainer"})
        projection = invocation.require_projection()
        return TrainerResult(
            best_state=projection.state,
            best_objectives=projection.feedback.objectives,
            best_feedback=projection.feedback,
            report={
                "inner_policy": projection.metadata["policy"],
                "pareto_size": len(projection.pareto_front.candidates),
            },
        )


def build_solver(config=None, *, resource_context=None, component_overrides=None):
    del config, component_overrides
    return OuterTrainer(resource_context=resource_context)
""".lstrip(),
        encoding="utf-8",
    )
    (project_root / "project_config.py").write_text(
        """
PROJECT_NAME = "solver_case_project"
L0 = {
    "namespace": "solver_case_project",
    "offer": {
        "threads": 1,
        "gpus": 0,
        "backend": "local",
        "metadata": {"memory_mb": 1024},
    },
    "policy": {
        "mode": "strict",
        "max_workers": 1,
        "max_threads": 1,
        "max_memory_mb": 1024,
    },
    "default_request": {
        "workers": 1,
        "threads": 1,
        "memory_mb": 1024,
        "backend": "local",
    },
}
STAGES = [{"name": "outer", "cases": ["outer_trainer"]}]
GROUPS = {"default": {"stages": ["outer"]}}
""".lstrip(),
        encoding="utf-8",
    )

    project_result = execute_project(project_root, record=False)

    assert project_result.ok
    parent_result = project_result.case_results[0]
    assert parent_result.output["protocol_type"] == "blackbase.trainer_result"
    assert parent_result.output["best_state"]["values"] == [1.0, 2.0]
    assert parent_result.output["best_objectives"] == [0.2, 0.8]
    assert parent_result.output["report"]["inner_policy"] == "best_solution"
    assert parent_result.output["report"]["pareto_size"] == 2
    child_payload = parent_result.metadata["runtime_audit"]["child_invocations"]["results"][0]
    child_result = CaseRunResult.from_dict(child_payload)
    assert child_result.ok
    assert child_result.identity.parent_case_run_id == parent_result.identity.case_run_id
    assert child_result.output["protocol_type"] == "blackbase.solver_result"
    assert child_result.output["best_objectives"] == [0.2, 0.8]
    assert child_result.output["solve_status"] == "feasible"
    assert child_result.output["termination_reason"] == "iteration_limit"
    assert child_result.output["quality"]["approximate"] is True
    assert child_result.output["pareto_front"]["protocol_type"] == "blackbase.population_snapshot"


def test_pareto_projection_requires_and_obeys_explicit_selector() -> None:
    front = PopulationSnapshot(
        candidates=(UnknownState([1.0]), UnknownState([2.0])),
        objectives=np.asarray([[0.1, 0.9], [0.4, 0.3]]),
        constraints=np.asarray([0.0, 0.2]),
    )
    result = SolverResult(
        pareto_front=front,
        solve_status="feasible",
        feasibility="feasible",
    )
    projector = ParetoSolutionProjector(
        lambda value, trainer, context: 1,
        feedback_mapper=OptimizationFeedbackMapper(),
    )

    projection = projector.project(result, trainer=object(), context={})

    assert projection.metadata["pareto_index"] == 1
    assert projection.state.as_array().tolist() == [2.0]
    assert projection.feedback.objectives.tolist() == [0.4, 0.3]
    assert projection.feedback.constraints.tolist() == [0.2]


def test_selection_projector_does_not_create_feedback_without_explicit_mapper() -> None:
    result = SolverResult(
        best_solution=UnknownState([1.0]),
        best_objectives=[0.25],
        solve_status="feasible",
        feasibility="feasible",
    )

    projection = BestSolutionProjector().project(result, trainer=object(), context={})

    assert projection.state.as_array().tolist() == [1.0]
    assert projection.feedback is None
    assert projection.metadata["feedback_mapped"] is False


def test_builtin_projectors_reject_non_successful_solve_status_by_default() -> None:
    result = SolverResult(
        best_solution=UnknownState([1.0]),
        solve_status="stopped",
        termination_reason="user_stop",
        feasibility="feasible",
    )

    with pytest.raises(ValueError, match="not projectable"):
        BestSolutionProjector().project(result, trainer=object(), context={})

    projection = BestSolutionProjector(accepted_statuses=("stopped",)).project(
        result,
        trainer=object(),
        context={},
    )
    assert projection.state.as_array().tolist() == [1.0]


def _runtime_with_result(case_result: CaseRunResult):
    class _Runtime:
        request = CaseRunRequest(
            project_name="project",
            stage_name="outer",
            case_name="trainer",
            case_kind="trainer",
        )

        def invoke(self, request):
            del request
            return case_result

    class _Trainer:
        case_runtime = _Runtime()

    return _Trainer()


def test_invoker_defaults_to_solver_result_without_projection() -> None:
    request = CaseRunRequest(
        project_name="project",
        stage_name="inner",
        case_name="solver",
    )
    result = SolverResult(
        solve_status="feasible",
        feasibility="feasible",
        best_solution=UnknownState([1.0]),
    )
    child = CaseRunResult(request=request, status="succeeded", output=result.as_dict())

    invocation = NsgablackSolverCaseInvoker("solver").invoke(
        trainer=_runtime_with_result(child)
    )

    assert invocation.ok
    assert invocation.case_result is child
    assert invocation.solver_result.solve_status == "feasible"
    assert invocation.projection is None

    with pytest.raises(SolverCaseInvocationError, match="No projection was requested"):
        invocation.require_projection()


def test_invoker_preserves_structured_child_case_failure() -> None:
    request = CaseRunRequest(
        project_name="project",
        stage_name="inner",
        case_name="solver",
    )
    failure = CaseFailure(
        kind="BackendUnavailable",
        message="solver backend is offline",
        phase="run",
        retryable=True,
        details={"backend": "domain-solver"},
    )
    child = CaseRunResult(
        request=request,
        status="failed",
        exit_code=1,
        failure=failure,
    )

    invocation = NsgablackSolverCaseInvoker("solver").invoke(
        trainer=_runtime_with_result(child)
    )

    assert not invocation.ok
    assert invocation.case_result is child
    assert invocation.failure is failure
    assert invocation.failure.retryable
    assert invocation.failure.details == {"backend": "domain-solver"}
    assert invocation.solver_result is None
    assert invocation.projection is None

    with pytest.raises(SolverCaseInvocationError, match="BackendUnavailable") as exc_info:
        invocation.require_projection()
    assert getattr(exc_info.value, "failure", None) is failure


def test_invoker_builds_timeout_at_each_invocation_and_forwards_control() -> None:
    request = CaseRunRequest(
        project_name="project",
        stage_name="inner",
        case_name="solver",
    )
    result = SolverResult(
        solve_status="feasible",
        feasibility="feasible",
        best_solution=UnknownState([1.0]),
    )
    child = CaseRunResult(request=request, status="succeeded", output=result.as_dict())

    class _Runtime:
        request = CaseRunRequest(
            project_name="project",
            stage_name="outer",
            case_name="trainer",
            case_kind="trainer",
        )

        def __init__(self) -> None:
            self.requests = []

        def invoke(self, child_request):
            self.requests.append(child_request)
            return child

    class _Trainer:
        case_runtime = _Runtime()

    trainer = _Trainer()
    invoker = NsgablackSolverCaseInvoker("solver", timeout_seconds=30.0)

    first_started = time.time()
    invoker.invoke(trainer=trainer)
    first_deadline = trainer.case_runtime.requests[-1].control.deadline_at
    second_started = time.time()
    invoker.invoke(trainer=trainer, timeout_seconds=5.0)
    second_deadline = trainer.case_runtime.requests[-1].control.deadline_at

    assert 29.0 <= first_deadline - first_started <= 31.0
    assert 4.0 <= second_deadline - second_started <= 6.0
    assert second_deadline < first_deadline

    explicit = ExecutionControl.with_timeout(10.0, metadata={"policy": "explicit"})
    invoker.invoke(trainer=trainer, control=explicit)
    assert trainer.case_runtime.requests[-1].control is explicit


def test_invoker_rejects_ambiguous_or_prebuilt_ancestor_control() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        NsgablackSolverCaseInvoker(
            "solver",
            timeout_seconds=1.0,
            control=ExecutionControl(),
        )

    parent = ExecutionControl()
    child_with_ancestors = parent.derive_child(ExecutionControl())
    with pytest.raises(ValueError, match="ancestor_cancellations"):
        NsgablackSolverCaseInvoker("solver", control=child_with_ancestors)


def test_oversized_pareto_front_uses_resolvable_project_artifact(tmp_path) -> None:
    project_root = create_project(tmp_path / "pareto_artifact_project", framework="blackbase")
    child_root = add_case("inner_solver", "solver", project_root=project_root)
    parent_root = add_case("outer_trainer", "trainer", project_root=project_root)
    (child_root / "build_solver.py").write_text(
        """
import numpy as np
from nsgablack.core import BlackBoxProblem, SolverBase


class Problem(BlackBoxProblem):
    def __init__(self):
        super().__init__(dimension=2, objectives=("cost", "risk"))

    def evaluate(self, candidate, context=None):
        del context
        return np.asarray(candidate, dtype=float)


class InnerSolver(SolverBase):
    case_result_inline_max_bytes = 1

    def __init__(self, resource_context):
        super().__init__(Problem(), resource_context=resource_context)

    def run(self):
        self.population = np.asarray([[1.0, 2.0], [2.0, 1.0]])
        self.objectives = np.asarray([[0.2, 0.8], [0.5, 0.4]])
        self.constraint_violations = np.asarray([0.0, 0.0])
        self.best_x = np.arange(128, dtype=float)
        self.best_objectives = np.asarray([0.1, 0.2])
        self.best_constraint_violation = 0.0
        self.pareto_solutions = {
            "individuals": self.population,
            "objectives": self.objectives,
        }
        self.pareto_objectives = self.objectives
        return {
            "status": "ok",
            "solve_status": "feasible",
            "termination_reason": "iteration_limit",
            "feasibility": "feasible",
        }


def build_solver(config=None, *, resource_context=None, component_overrides=None):
    del config, component_overrides
    return InnerSolver(resource_context)
""".lstrip(),
        encoding="utf-8",
    )
    (parent_root / "build_solver.py").write_text(
        """
from blackbase.types import TrainerResult
from mlblack.core import BlankTrainer
from mlblack.integrations import NsgablackSolverCaseInvoker


class OuterTrainer(BlankTrainer):
    def fit(self):
        invocation = NsgablackSolverCaseInvoker(
            "inner_solver",
            resource_request={
                "workers": 1,
                "threads": 1,
                "memory_mb": 256,
                "backend": "local",
                "compute_backend": "auto",
                "device": "cpu",
            },
        ).invoke(trainer=self)
        solver_result = invocation.require_solver_result()
        return TrainerResult(
            report={"pareto_front_ref": solver_result.pareto_front_ref},
        )


def build_solver(config=None, *, resource_context=None, component_overrides=None):
    del config, component_overrides
    return OuterTrainer(resource_context=resource_context)
""".lstrip(),
        encoding="utf-8",
    )
    (project_root / "project_config.py").write_text(
        """
PROJECT_NAME = "pareto_artifact_project"
L0 = {
    "namespace": "pareto_artifact_project",
    "offer": {
        "threads": 1,
        "gpus": 0,
        "backend": "local",
        "metadata": {"memory_mb": 1024},
    },
    "policy": {
        "mode": "strict",
        "max_workers": 1,
        "max_threads": 1,
        "max_memory_mb": 1024,
    },
    "default_request": {
        "workers": 1,
        "threads": 1,
        "memory_mb": 1024,
        "backend": "local",
    },
    "artifacts": {"path": ".blackbase/artifacts"},
}
STAGES = [{"name": "outer", "cases": ["outer_trainer"]}]
GROUPS = {"default": {"stages": ["outer"]}}
""".lstrip(),
        encoding="utf-8",
    )

    project_result = execute_project(project_root, record=False)

    assert project_result.ok
    parent_result = project_result.case_results[0]
    child_payload = parent_result.metadata["runtime_audit"]["child_invocations"]["results"][0]
    child_result = CaseRunResult.from_dict(child_payload)
    assert child_result.ok
    assert child_result.output["best_solution"] is None
    best_ref_payload = child_result.output["best_solution_ref"]
    assert best_ref_payload["backend"] == "filesystem"
    assert best_ref_payload["kind"] == "solution"
    assert best_ref_payload["checksum"].startswith("sha256:")
    assert Path(best_ref_payload["uri"]).is_file()
    assert child_result.output["pareto_front"] is None
    ref_payload = child_result.output["pareto_front_ref"]
    assert ref_payload["backend"] == "filesystem"
    assert ref_payload["kind"] == "pareto_front"
    assert ref_payload["checksum"].startswith("sha256:")
    assert Path(ref_payload["uri"]).is_file()
    assert child_result.artifact_refs["best_solution"].uri == best_ref_payload["uri"]
    assert child_result.artifact_refs["pareto_front"].uri == ref_payload["uri"]
