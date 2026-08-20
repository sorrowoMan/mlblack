from __future__ import annotations

from mlblack.core import Feedback
from mlblack.integrations import build_diagnostic_solver
from mlblack.integrations import LearningSolver
from nsgablack.adapters import FixedCandidateAdapter


def test_diagnostic_uses_learning_solver_and_runs_once() -> None:
    calls: list[dict] = []

    def run_diagnostic(context):
        calls.append(dict(context))
        return {"status": "ok", "objective": 2.5, "metrics": {"rows": 12}}

    solver = build_diagnostic_solver(
        run_diagnostic,
        name="schema_check",
        resource_context={"threads": 2, "namespace": "tests.schema"},
    )
    result = solver.fit()

    assert isinstance(solver, LearningSolver)
    assert isinstance(solver.adapter, FixedCandidateAdapter)
    assert len(calls) == 1
    assert result.best_feedback.objectives.tolist() == [2.5]
    assert result.best_feedback.metrics["rows"] == 12
    assert result.report["optimization_control_plane"] == "nsgablack.ComposableSolver"


def test_diagnostic_accepts_feedback_runner() -> None:
    solver = build_diagnostic_solver(
        lambda context: Feedback(objectives=[1.0], metrics={"namespace": context["resource.namespace"]}),
        name="feedback_check",
        resource_context={"namespace": "tests.feedback"},
    )
    result = solver.run(max_steps=1)
    assert result.best_feedback.metrics["namespace"] == "tests.feedback"
