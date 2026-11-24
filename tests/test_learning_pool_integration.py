"""The ML facade preserves the shared pool's startup and restore semantics."""
from __future__ import annotations

import copy

import numpy as np
import pytest

from blackbase.orchestration import ExecutionPlan
from mlblack.core import Feedback, LearningProblem, ModelRepresentation, UnknownState
from mlblack.integrations import build_learning_solver
from nsgablack.adapters import (
    GradientOptimizerAdapter, MultiStrategyConfig, NSGA2Adapter, PoolTaskSpec,
    SearchPlanAdapter, StrategySpec, subpools,
)


class _Representation(ModelRepresentation):
    dimension = 1

    def __init__(self):
        self.control = None
        self.init_calls = 0
        self.init_pools = []
        self.forbid_init = False

    def init(self, context):
        if self.forbid_init:
            pytest.fail("restoring a run must not initialize a fresh ML candidate")
        self.init_calls += 1
        self.init_pools.append(
            None if self.control is None else self.control.solution_pool
        )
        return UnknownState([4.0], {"model": "scalar", "feature_order": ["x"]})

    def decode(self, state, context=None):
        return state.as_array().copy()


class _Problem(LearningProblem):
    def __init__(self):
        self.calls = []

    def evaluate(self, model, state, context=None):
        assert state.metadata["model"] == "scalar"
        assert tuple(state.metadata["feature_order"]) == ("x",)
        np.testing.assert_array_equal(model, state.as_array())
        x = float(state.as_array()[0])
        self.calls.append(x)
        return Feedback(objectives=[x * x], gradients=[2.0 * x])


def _solver(representation=None):
    representation = representation or _Representation()
    problem = _Problem()
    adapter = GradientOptimizerAdapter(optimizer="sgd", learning_rate=0.1)
    control = build_learning_solver(
        problem=problem,
        representation=representation,
        adapter=adapter,
        run_name="learning_pool_integration",
    )
    representation.control = control
    return control, representation, problem, adapter


def test_ml_initialization_is_deferred_until_the_shared_pool_exists():
    control, representation, problem, adapter = _solver()

    assert representation.init_calls == 0
    assert adapter.algorithm_unit is None
    assert problem.calls == []

    control.fit(max_steps=1)

    assert representation.init_calls == 1
    assert representation.init_pools == [control.solution_pool]
    assert adapter.algorithm_unit is not None
    assert problem.calls == [4.0]
    refs = adapter.get_pool_refinement_results(control)
    assert len(refs) == 1
    assert control.read_solutions(refs).semantic_states[0].metadata["model"] == "scalar"


def test_ml_facade_state_restores_exact_pool_results_without_a_new_evaluation():
    source, _, source_problem, source_adapter = _solver()
    source.fit(max_steps=2)
    refs = source_adapter.get_pool_refinement_results(source)
    assert len(refs) == 1
    expected = source.read_solutions(refs)
    raw_objectives, raw_violations = source.solution_pool.read_raw_feedback(refs)
    state = copy.deepcopy(source.get_state())
    expected_steps = source.run_progress_steps
    expected_attempts = source.run_progress_attempts
    expected_generation = source.generation
    expected_evaluations = source.evaluation_count
    expected_clock = source.solution_pool.timestamp
    expected_history = copy.deepcopy(source.history)

    target, representation, problem, adapter = _solver()
    representation.forbid_init = True
    target.set_state(state)
    # Restore-only fit exercises the public post-setup restore slot and finalization.
    result = target.fit(max_steps=0)

    assert problem.calls == []
    assert target.run_progress_steps == expected_steps
    assert target.run_progress_attempts == expected_attempts
    assert target.generation == expected_generation
    assert target.evaluation_count == expected_evaluations
    assert target.solution_pool.timestamp == expected_clock
    assert target.run_session.run_id == source.run_session.run_id
    assert target.solution_pool.pool_id == source.solution_pool.pool_id
    assert adapter.get_pool_refinement_results(target) == refs
    assert set(refs).issubset(set(adapter.get_solution_pool_roots(target)))
    actual = target.read_solutions(refs)
    assert actual.candidate_tokens == expected.candidate_tokens
    np.testing.assert_array_equal(actual.numeric_matrix, expected.numeric_matrix)
    assert actual.semantic_states[0].metadata == expected.semantic_states[0].metadata
    actual_objectives, actual_violations = target.solution_pool.read_raw_feedback(refs)
    np.testing.assert_array_equal(actual_objectives, raw_objectives)
    np.testing.assert_array_equal(actual_violations, raw_violations)
    assert target.history == expected_history
    assert representation.init_calls == 0
    np.testing.assert_array_equal(result.best_state.as_array(), source.best_state.as_array())
    assert result.best_state.metadata == source.best_state.metadata
    assert len(source_problem.calls) == expected_evaluations


def test_ml_facade_resume_keeps_old_versions_and_evaluates_the_pending_successor():
    source, _, _, source_adapter = _solver()
    source.fit(max_steps=2)
    previous = source_adapter.get_pool_refinement_results(source)
    old_batch = source.read_solutions(previous)
    old_objectives, old_violations = source.solution_pool.read_raw_feedback(previous)
    successor = source_adapter.current_x.copy()
    state = copy.deepcopy(source.get_state())

    target, representation, problem, adapter = _solver()
    representation.forbid_init = True
    target.set_state(state)
    target.fit(max_steps=1)

    assert problem.calls == pytest.approx(successor.tolist())
    assert target.run_progress_steps == source.run_progress_steps + 1
    assert target.run_progress_attempts == source.run_progress_attempts + 1
    assert target.generation == source.generation + 1
    assert target.evaluation_count == source.evaluation_count + 1
    assert adapter.step_index == source_adapter.step_index + 1
    restored_old = target.read_solutions(previous)
    assert restored_old.candidate_tokens == old_batch.candidate_tokens
    np.testing.assert_array_equal(restored_old.numeric_matrix, old_batch.numeric_matrix)
    restored_objectives, restored_violations = target.solution_pool.read_raw_feedback(previous)
    np.testing.assert_array_equal(restored_objectives, old_objectives)
    np.testing.assert_array_equal(restored_violations, old_violations)
    final, = adapter.get_pool_refinement_results(target)
    assert final != previous[0]
    assert final.candidate_token != previous[0].candidate_token
    np.testing.assert_array_equal(target.read_solutions((final,)).numeric_matrix[0], successor)
    assert target.read_solutions((final,)).semantic_states[0].metadata["model"] == "scalar"
    assert representation.init_calls == 0
    raw, violations = target.solution_pool.read_raw_feedback((final,))
    np.testing.assert_allclose(raw, [[float(successor[0]) ** 2]])
    np.testing.assert_array_equal(violations, [0.0])


@pytest.mark.parametrize("version", [7, 6])
def test_ml_state_without_pool_value_closure_is_rejected_before_runtime(version):
    source, _, _, _ = _solver()
    source.fit(max_steps=1)
    state = copy.deepcopy(source.get_state())
    state["state_version"] = version
    state.pop("solution_pool")
    target, representation, problem, adapter = _solver()

    with pytest.raises(ValueError, match="solution_pool.*closure"):
        target.set_state(state)

    assert representation.init_calls == 0
    assert problem.calls == []
    assert adapter.algorithm_unit is None
    assert not target.run_session.pending_restore


def test_first_runtime_candidate_must_match_the_declared_model_dimension():
    class WrongLayout(_Representation):
        dimension = 2

    control, representation, problem, _ = _solver(WrongLayout())
    assert representation.init_calls == 0
    with pytest.raises(ValueError, match="dimension.*configured model layout"):
        control.fit(max_steps=1)
    assert representation.init_calls == 1
    assert problem.calls == []
    assert control.evaluation_count == 0
    assert control.run_progress_steps == 0


class _PopulationRepresentation(_Representation):
    def init(self, context):
        state = super().init(context)
        return state.with_values([2.0 + self.init_calls])


def _population_gradient_solver():
    representation = _PopulationRepresentation()
    problem = _Problem()
    ga = NSGA2Adapter(name="model_population", population_size=2, offspring_size=2)
    gradient = GradientOptimizerAdapter(
        name="model_refinement", optimizer="sgd", learning_rate=0.1
    )
    plan = SearchPlanAdapter(
        ExecutionPlan.serial(
            ExecutionPlan.leaf("ga", steps=2), ExecutionPlan.leaf("gradient")
        ),
        strategies=[
            StrategySpec(ga, "ga"),
            StrategySpec(gradient, "gradient", pool_task=PoolTaskSpec(
                subpools.all(ga), count="all", steps=2, shuffle=False
            )),
        ],
        config=MultiStrategyConfig(total_batch_size=2, adapt_weights=False),
    )
    control = build_learning_solver(
        problem=problem, representation=representation, adapter=plan,
        run_name="population_then_gradient",
    )
    representation.control = control
    control.set_random_seed(37)
    return control, representation, problem, ga, plan


@pytest.mark.parametrize("restore_before_refinement", [
    pytest.param(False, id="continuous"),
    pytest.param(True, id="restore_after_ga"),
])
def test_ml_population_view_hands_every_member_to_gradient_refinement(restore_before_refinement):
    control, representation, problem, ga, plan = _population_gradient_solver()
    if restore_before_refinement:
        source = control
        source.fit(max_steps=2)
        original_members = ga.get_population_solution_refs(source)
        original_population = source.read_solutions(original_members)
        original_raw, original_violations = source.solution_pool.read_raw_feedback(original_members)
        state = copy.deepcopy(source.get_state())
        source_calls = list(problem.calls)
        control, representation, problem, ga, plan = _population_gradient_solver()
        representation.forbid_init = True
        control.set_state(state)
        control.fit(max_steps=4)
        assert representation.init_calls == 0
        assert ga.get_population_solution_refs(control) == original_members
        np.testing.assert_array_equal(ga.population, original_population.numeric_matrix)
        restored_raw, restored_violations = control.solution_pool.read_raw_feedback(original_members)
        np.testing.assert_array_equal(restored_raw, original_raw)
        np.testing.assert_array_equal(restored_violations, original_violations)
        assert control.run_session.run_id == source.run_session.run_id
        assert source.semantic_problem.calls == source_calls
        assert control.evaluation_count == source.evaluation_count + 4
        assert control.run_progress_attempts == source.run_progress_attempts + 4
    else:
        control.fit(max_steps=6)
        assert representation.init_calls == 2

    assert plan.cursor.completed
    assert control.run_progress_steps == 6
    assert all(pool is control.solution_pool for pool in representation.init_pools)
    members = ga.get_population_solution_refs(control)
    final = plan.get_pool_task_results(control, "gradient")
    assert len(members) == len(final) == 2
    population = control.read_solutions(members)
    refined = control.read_solutions(final)
    np.testing.assert_allclose(refined.numeric_matrix, population.numeric_matrix * 0.8)
    np.testing.assert_array_equal(ga.population, population.numeric_matrix)
    raw, violations = control.solution_pool.read_raw_feedback(members)
    np.testing.assert_allclose(raw, population.numeric_matrix ** 2)
    np.testing.assert_array_equal(violations, [0.0, 0.0])
    refined_raw, refined_violations = control.solution_pool.read_raw_feedback(final)
    np.testing.assert_allclose(refined_raw, refined.numeric_matrix ** 2)
    np.testing.assert_array_equal(refined_violations, [0.0, 0.0])
    assert set(population.candidate_tokens).isdisjoint(refined.candidate_tokens)
    assert all(state.metadata["model"] == "scalar" for state in refined.semantic_states)
    # Each final candidate descends from the baseline evaluated for that source,
    # and the baseline itself retains the exact GA handoff reference.
    all_refs = control.solution_pool.select(stage="accepted")
    by_token = {ref.candidate_token: ref for ref in all_refs}
    for member, result in zip(members, final):
        result_origin, = control.solution_pool.read_provenance((result,))
        baseline = by_token[result_origin["parent_token"]]
        baseline_origin, = control.solution_pool.read_provenance((baseline,))
        assert baseline_origin["parent_token"] == member.candidate_token
        assert baseline_origin["metadata"]["solution_source_ref"] == member.as_dict()
        np.testing.assert_array_equal(
            control.read_solutions((baseline,)).numeric_matrix,
            control.read_solutions((member,)).numeric_matrix,
        )
