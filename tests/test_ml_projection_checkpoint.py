"""Formal checkpoint recovery keeps an incumbent outside the final event."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from blackbase import CandidateBatch
from blackbase.resources import DataRef
from mlblack.core import Feedback, LearningProblem, ModelRepresentation, UnknownState
from mlblack.integrations import build_learning_solver
from nsgablack.adapters import FixedCandidateAdapter
from nsgablack.plugins import CheckpointResumeConfig, CheckpointResumePlugin, Plugin


class _Representation(ModelRepresentation):
    dimension = 1

    def __init__(self):
        self.init_calls = 0

    def init(self, context):
        self.init_calls += 1
        return UnknownState([float(context["step"])], {"model": "scalar"})

    def decode(self, state, context=None):
        return state.as_array().copy()


class _Problem(LearningProblem):
    def __init__(self):
        self.prepare_calls = 0
        self.evaluate_calls = 0
        self.restore_calls = 0

    def prepare_model_for_evaluation(self, model, state, context):
        self.prepare_calls += 1
        return model

    def evaluate(self, model, state, context=None):
        self.evaluate_calls += 1
        value = float(model[0])
        return Feedback(
            objectives=[value * value],
            metrics={"loss": value * value, "evaluated_value": value},
            info={"saved_model": value},
        )

    def restore_model_for_evaluation(self, model, state, feedback, context):
        self.restore_calls += 1
        return np.asarray([feedback.info["saved_model"]], dtype=float)


def _solver():
    representation, problem = _Representation(), _Problem()
    solver = build_learning_solver(
        problem=problem,
        representation=representation,
        adapter=FixedCandidateAdapter(),
        run_name="ml_projection_checkpoint",
    )
    return solver, representation, problem


class _ObserveRestoredProjection(Plugin):
    def __init__(self):
        super().__init__(name="observe_restored_projection")
        self.observed = None

    def on_solver_init(self, solver):
        self.observed = {
            "feedback": solver.best_feedback.as_dict(),
            "binding": copy.deepcopy(solver._best_feedback_binding),
            "history": copy.deepcopy(solver.history),
            "refs": dict(solver.result_artifact_refs),
            "best_model_ref": solver.best_model_ref,
            "model_handle": solver.best_model,
        }


def test_formal_checkpoint_restores_earlier_best_feedback_without_execution(
    tmp_path, monkeypatch,
):
    monkeypatch.setenv("NSGABLACK_CHECKPOINT_HMAC_KEY", "ml-projection-test-key")
    source, source_representation, source_problem = _solver()
    writer = CheckpointResumePlugin(config=CheckpointResumeConfig(
        checkpoint_dir=str(tmp_path), save_every=0, save_on_finish=False,
        strict=True,
    ))
    source.add_plugin(writer)
    source.fit(max_steps=2)
    assert source_representation.init_calls == 2
    assert source_problem.evaluate_calls == source_problem.prepare_calls == 2
    assert source.best_feedback.metrics["evaluated_value"] == 0.0
    assert source.last_evaluated_feedback[0].metrics["evaluated_value"] == 1.0
    final_batch = CandidateBatch.from_dict(source.get_last_evaluation_event().candidate_payload)
    assert source.get_incumbent().candidate_token not in final_batch.candidate_tokens
    best_ref = DataRef(uri="memory://ml-checkpoint/best", kind="model", backend="memory")
    report_ref = DataRef(uri="memory://ml-checkpoint/report", kind="report", backend="memory")
    source.set_best_model_ref(best_ref)
    source.result_artifact_refs["report"] = report_ref
    expected_history = copy.deepcopy(source.history)
    expected_feedback = source.best_feedback.as_dict()
    expected_binding = copy.deepcopy(source._best_feedback_binding)
    checkpoint = writer.save_checkpoint(reason="earlier-best")
    assert checkpoint is not None and checkpoint.is_file()

    target, representation, problem = _solver()
    reader = CheckpointResumePlugin(config=CheckpointResumeConfig(
        checkpoint_dir=str(tmp_path), save_every=0, save_on_finish=False,
        auto_resume=True, resume_from=str(checkpoint), strict=True,
    ))
    observer = _ObserveRestoredProjection()
    target.add_plugin(reader)
    target.add_plugin(observer)
    result = target.fit(max_steps=0)

    assert reader.last_loaded_path == str(checkpoint)
    assert "ml_projection" in reader.get_report()["resume_audit"]["restored_components"]
    assert observer.observed["feedback"] == expected_feedback
    assert observer.observed["binding"] == expected_binding
    assert observer.observed["history"] == expected_history
    assert observer.observed["refs"] == {"best_model": best_ref, "report": report_ref}
    assert observer.observed["best_model_ref"] == best_ref
    assert observer.observed["model_handle"] is None
    assert representation.init_calls == problem.prepare_calls == problem.evaluate_calls == 0
    assert problem.restore_calls == 1
    assert target.best_feedback.as_dict() == expected_feedback
    assert target._best_feedback_binding == expected_binding
    assert target.history == expected_history
    np.testing.assert_array_equal(result.best_state.as_array(), [0.0])
    np.testing.assert_array_equal(target.best_model, [0.0])


@pytest.mark.parametrize("field,value", [
    ("candidate_token", "another-candidate"),
    ("evaluation_id", "another-evaluation"),
])
def test_ml_projection_component_rejects_wrong_incumbent_binding(field, value):
    solver, _, _ = _solver()
    solver.fit(max_steps=1)
    component = solver.checkpoint_components()["ml_projection"]
    state = copy.deepcopy(component.get_state())
    before = copy.deepcopy(component.get_state())
    state["best_feedback_binding"][field] = value

    with pytest.raises(ValueError, match="not bound to the authoritative incumbent"):
        component.set_state(state)

    assert component.get_state() == before


def test_facade_and_formal_component_share_ml_projection_fields():
    solver, _, _ = _solver()
    solver.fit(max_steps=1)
    facade = solver.get_state()
    component = dict(solver.checkpoint_components()["ml_projection"].get_state())
    component.pop("schema")

    assert component == {name: facade[name] for name in component}
    assert "history" not in component
