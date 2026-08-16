from __future__ import annotations

from mlblack.core import DiagnosticTrainer, build_diagnostic_trainer


def test_diagnostic_trainer_runs_once_and_uses_standard_state_surfaces() -> None:
    calls: list[dict] = []

    def run_diagnostic(context):
        calls.append(dict(context))
        return {"status": "ok", "objective": 2.5, "metrics": {"rows": 12}}

    trainer = build_diagnostic_trainer(
        run_diagnostic,
        name="schema_check",
        resource_context={"threads": 2, "namespace": "tests.schema"},
    )

    result = trainer.fit(max_steps=50)

    assert isinstance(trainer, DiagnosticTrainer)
    assert len(calls) == 1
    assert result.best_feedback is not None
    assert result.best_feedback.objectives.tolist() == [2.5]
    assert result.best_feedback.metrics["rows"] == 12
    assert result.report["trainer_kind"] == "diagnostic"
    assert result.report["resources"]["threads"] == 2
    assert trainer.context_store.get("last_population_snapshot")


def test_diagnostic_trainer_accepts_feedback_runner() -> None:
    from mlblack.core import Feedback

    trainer = build_diagnostic_trainer(
        lambda context: Feedback(objectives=[1.0], metrics={"namespace": context["resource.namespace"]}),
        name="feedback_check",
        resource_context={"namespace": "tests.feedback"},
    )

    result = trainer.run()

    assert result.best_feedback.metrics["namespace"] == "tests.feedback"
