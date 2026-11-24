"""Model projection restores recorded semantics without repeating execution."""

from __future__ import annotations

import numpy as np
import pytest

from blackbase import CandidateBatch
from mlblack.core import Feedback, LearningProblem, UnknownState
from mlblack.core.provider_problem import FunctionalGradientLearningProblem, ProviderBackedLearningProblem
from mlblack.integrations import build_learning_solver
from mlblack.representations.linear_point import LinearPointConfig, LinearPointRepresentation
from nsgablack.adapters import FixedCandidateAdapter, GradientOptimizerAdapter
from nsgablack.core.evaluation_feedback import OptimizationFeedbackBatch
from nsgablack.core.state.incumbent import CandidateProvenance, IncumbentState
from nsgablack.plugins import CheckpointResumeConfig, CheckpointResumePlugin


class _IdentityProblem(LearningProblem):
    def evaluate(self, *args, **kwargs):
        raise AssertionError("projection must not evaluate a model")


class _PreparingProblem(_IdentityProblem):
    def __init__(self):
        self.prepare_calls = 0

    def prepare_model_for_evaluation(self, model, state, context):
        self.prepare_calls += 1
        raise AssertionError("projection must not repeat preparation")


class _RestorableProblem(_PreparingProblem):
    def __init__(self):
        super().__init__()
        self.restore_calls = 0

    def restore_model_for_evaluation(self, model, state, feedback, context):
        self.restore_calls += 1
        return {"saved_result": feedback.info["saved_result"]}

    def compute_functional_gradient(self, *args, **kwargs):
        raise AssertionError("projection must not compute a new gradient")


def _solver_with_recorded_feedback(problem, feedback, *, representation=None, state=None):
    representation = representation or LinearPointRepresentation(LinearPointConfig(n_features=1))
    state = state or UnknownState([1.0, 2.0], metadata={"model": "linear"})
    solver = build_learning_solver(
        problem=problem,
        representation=representation,
        adapter=FixedCandidateAdapter(),
        run_name="projection_restore",
    )
    token = "candidate:recorded-model"
    batch = CandidateBatch(
        semantic_states=(state,),
        numeric_matrix=np.asarray([state.as_array()]),
        candidate_tokens=(token,),
    )
    solver.record_evaluation_event(
        batch,
        OptimizationFeedbackBatch.from_feedback((feedback,)),
        (CandidateProvenance(candidate_token=token),),
    )
    solver.set_incumbent(IncumbentState(
        candidate=state.as_array(),
        objectives=np.asarray(feedback.objectives),
        constraint_violation=0.0,
        score=float(feedback.objectives[0]),
        policy_id=str(solver.incumbent_scalarizer_id),
        policy_context=dict(solver.incumbent_scalarizer_context),
        candidate_token=token,
    ))
    return solver


def test_parameter_model_projection_never_calls_preparation(monkeypatch):
    def forbidden_prepare(self, model, state, context):
        raise AssertionError("restoration called the evaluation preparation hook")

    monkeypatch.setattr(LearningProblem, "prepare_model_for_evaluation", forbidden_prepare)
    solver = _solver_with_recorded_feedback(_IdentityProblem(), Feedback(objectives=[1.0]))

    solver._synchronize_ml_projection(record_history=False)

    np.testing.assert_allclose(solver.best_model.predict(np.asarray([[3.0]])), [7.0])
    assert solver.history == []


def test_prepared_model_without_restore_rule_is_rejected_without_execution():
    problem = _PreparingProblem()
    solver = _solver_with_recorded_feedback(problem, Feedback(objectives=[1.0]))

    with pytest.raises(RuntimeError, match="restore_model_for_evaluation must explicitly restore"):
        solver._synchronize_ml_projection(record_history=False)

    assert problem.prepare_calls == 0
    assert solver.best_model is None


def test_explicit_restore_rule_consumes_saved_feedback_without_execution():
    problem = _RestorableProblem()
    solver = _solver_with_recorded_feedback(
        problem, Feedback(objectives=[1.0], info={"saved_result": 42.0}),
    )

    solver._synchronize_ml_projection(record_history=False)

    assert solver.best_model == {"saved_result": 42.0}
    assert problem.restore_calls == 1
    assert problem.prepare_calls == 0


@pytest.mark.parametrize("wrapper", ["functional_gradient", "provider"])
def test_problem_wrappers_forward_the_explicit_restore_rule(wrapper):
    problem = _RestorableProblem()
    if wrapper == "functional_gradient":
        wrapped = FunctionalGradientLearningProblem(problem)
    else:
        wrapped = ProviderBackedLearningProblem(
            problem, object(), problem_id="restore-only", capabilities=("evaluate",),
        )
    feedback = Feedback(objectives=[1.0], info={"saved_result": 42.0})

    restored = wrapped.restore_model_for_evaluation(None, UnknownState([1.0]), feedback, {})

    assert restored == {"saved_result": 42.0}
    assert problem.restore_calls == 1
    assert problem.prepare_calls == 0


def test_nested_case_projection_reuses_recorded_child_result(monkeypatch):
    from nsgablack.examples.cases.super_topology_orchestration.cases.nested_trainer.pipeline.codec import NestedModelCodec
    from nsgablack.examples.cases.super_topology_orchestration.cases.nested_trainer.problem.learning_problem import NestedLearningProblem

    point = [0.6, 0.2]
    problem = NestedLearningProblem(point)
    codec = NestedModelCodec(point)
    audit = {
        "selected_calibration": 0.15,
        "calibration_target": problem.calibration_target,
        "inner_identity": {"case_name": "recorded_child"},
        "inner_pool": {"record_count": 3},
    }
    feedback = Feedback(objectives=[1.0, 2.0], info={"nested_audit": audit})
    solver = _solver_with_recorded_feedback(
        problem, feedback, representation=codec, state=codec.state(point),
    )

    def forbidden_child_run(*args, **kwargs):
        raise AssertionError("restoration started another nested Case")

    monkeypatch.setattr(problem.provider, "solve_calibration", forbidden_child_run)
    solver._synchronize_ml_projection(record_history=False)

    assert solver.best_model["calibration"] == 0.15
    assert solver.best_model["nested_audit"] == audit
    solver.best_model["nested_audit"]["inner_pool"]["record_count"] = 99
    assert feedback.info["nested_audit"]["inner_pool"]["record_count"] == 3


def test_nested_restore_rejects_evidence_from_another_outer_problem():
    from nsgablack.examples.cases.super_topology_orchestration.cases.nested_trainer.problem.learning_problem import NestedLearningProblem

    problem = NestedLearningProblem([0.6, 0.2])
    feedback = Feedback(objectives=[1.0, 2.0], info={"nested_audit": {
        "selected_calibration": 0.15, "calibration_target": 0.9,
    }})

    with pytest.raises(ValueError, match="another outer problem"):
        problem.restore_model_for_evaluation({}, UnknownState([0.6, 0.2]), feedback, {})


def test_formal_checkpoint_restores_deferred_dimension_without_initialization(tmp_path, monkeypatch):
    from mlblack.core import ModelRepresentation

    class DeferredParameters(ModelRepresentation):
        def __init__(self):
            self.init_calls = 0

        def init(self, context):
            self.init_calls += 1
            return UnknownState([2.0, 4.0], metadata={"model": "deferred_parameters"})

        def decode(self, state, context=None):
            return state.as_array().copy()

    class ParameterProblem(LearningProblem):
        def __init__(self):
            self.calls = []

        def evaluate(self, model, state, context=None):
            self.calls.append(state.as_array().copy())
            return Feedback(objectives=[float(model @ model)], gradients=2.0 * model)

    def build():
        representation, problem = DeferredParameters(), ParameterProblem()
        solver = build_learning_solver(
            problem=problem, representation=representation,
            adapter=GradientOptimizerAdapter(optimizer="sgd", learning_rate=0.1),
            run_name="deferred_dimension_checkpoint",
        )
        return solver, representation, problem

    monkeypatch.setenv("NSGABLACK_CHECKPOINT_HMAC_KEY", "deferred-dimension-test-key")
    source, _, _ = build()
    writer = CheckpointResumePlugin(config=CheckpointResumeConfig(
        checkpoint_dir=str(tmp_path), save_every=0, save_on_finish=False, strict=True,
    ))
    source.add_plugin(writer)
    source.fit(max_steps=2)
    expected_successor = source.adapter.current_x.copy()
    checkpoint = writer.save_checkpoint(reason="deferred-dimension")
    assert checkpoint is not None
    target, representation, problem = build()
    assert target.dimension == 0
    reader = CheckpointResumePlugin(config=CheckpointResumeConfig(
        checkpoint_dir=str(tmp_path), save_every=0, save_on_finish=False,
        auto_resume=True, resume_from=str(checkpoint), strict=True,
    ))
    target.add_plugin(reader)

    # Auto-resume is discovered during setup; the shared run receives the
    # final logical step budget, including the two restored commits.
    target.fit(max_steps=source.run_progress_steps + 1)

    assert target.dimension == target.problem.dimension == 2
    assert representation.init_calls == 0
    assert len(problem.calls) == 1
    np.testing.assert_array_equal(problem.calls[0], expected_successor)


@pytest.mark.parametrize("strict,level", [(True, "error"), (False, "warn")])
def test_doctor_rejects_an_incompatible_restore_port_without_constructing(strict, level):
    from pathlib import Path
    from mlblack.project.doctor import _check_model_restore_contract

    class BadRestore(_IdentityProblem):
        def __init__(self):
            raise AssertionError("Doctor must not construct a Problem")

        def restore_model_for_evaluation(self, model, state, context):
            raise AssertionError("Doctor must not execute a restore hook")

    diagnostics = []
    _check_model_restore_contract(BadRestore, diagnostics, strict=strict, path=Path(__file__))

    assert [(item.level, item.code) for item in diagnostics] == [
        (level, "model-restore-contract-invalid")
    ]
