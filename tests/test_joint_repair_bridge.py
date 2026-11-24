import numpy as np
import pytest

from blackbase.types import CandidateBatch
from mlblack.core import Feedback, LearningProblem, ModelRepresentation, UnknownState
from mlblack.integrations import build_learning_solver
from nsgablack.adapters import GaussianSearchAdapter
from nsgablack.representation import JointRepairPolicy
from nsgablack.utils.runtime_contracts import inspect_runtime_contracts


class Representation(ModelRepresentation):
    def init(self, context):
        return UnknownState([0.8], {"model": "point"})
    def decode(self, state, context=None):
        return state.values.copy()
    def repair(self, state, context=None):
        return state.with_values(np.clip(state.values, 0, 1))


class Problem(LearningProblem):
    def evaluate(self, model, state, context=None):
        assert state.metadata["model"] == "point"
        assert state.metadata["unit_repaired"]
        return Feedback(objectives=[float(state.values[0])])


class Repair:
    def repair(self, unit, evidence, context):
        result = CandidateBatch.from_candidates(
            [state.with_values([0.4]).with_meta(unit_repaired=True) for state in evidence.after.semantic_states],
            candidate_tokens=evidence.after.candidate_tokens)
        unit.current = result.numeric_matrix[0].copy()
        return result
    def validate(self, unit, evidence, context):
        return bool(np.array_equal(unit.current, evidence.after.numeric_matrix[0]))


def solver(validator=None):
    adapter = GaussianSearchAdapter(population_size=1)
    adapter.set_unit_repair(Repair())
    return build_learning_solver(problem=Problem(), representation=Representation(), adapter=adapter,
                                 problem_validator=validator, joint_repair_policy=JointRepairPolicy())


def test_ml_bridge_validates_final_semantics_in_the_shared_step():
    seen = []
    def validate(state, context):
        seen.append(state)
        return bool(state.metadata["unit_repaired"] and state.values[0] == 0.4)
    control = solver(validate)
    assert not [code for code, _ in inspect_runtime_contracts(control) if code == "joint-repair-contract-invalid"]
    outcome = control.step()
    assert outcome.committed and outcome.evaluations == 1
    assert seen[0].metadata["model"] == "point"
    assert outcome.metadata["joint_repair"]["admission"]
    assert control.feedback[0].objectives[0] == 0.4


def test_ml_bridge_requires_explicit_validation_without_guessing_representation_methods():
    control = solver()
    assert any(code == "joint-repair-contract-invalid" for code, _ in inspect_runtime_contracts(control))
    with pytest.raises(TypeError, match="ProblemRepair"):
        control.step()


def test_ml_validator_false_rejects_without_evaluating_model():
    control = solver(lambda state, context: False)
    outcome = control.step()
    assert outcome.status == "rejected" and outcome.stop_requested
    assert control.evaluation_count == 0
    assert outcome.metadata["joint_repair"]["failures"][0]["reason"] == "problem_invalid"


@pytest.mark.parametrize("final_value", [0.8, 0.4])
def test_gradient_unit_tracks_numeric_and_metadata_repair_before_evaluation(final_value):
    from nsgablack.adapters import GradientOptimizerAdapter

    adapter = GradientOptimizerAdapter(optimizer="sgd", learning_rate=0.1)

    class ProjectedRepresentation(Representation):
        def repair(self, state, context=None):
            return state.with_values([final_value]).with_meta(repaired=True)

    class GradientProblem(LearningProblem):
        def evaluate(self, model, state, context=None):
            assert state.values[0] == final_value
            assert adapter.current_x[0] == final_value
            assert adapter.algorithm_unit.candidate_metadata["repaired"]
            return Feedback(objectives=[final_value ** 2], gradients=[2 * final_value])

    control = build_learning_solver(problem=GradientProblem(), representation=ProjectedRepresentation(),
                                    adapter=adapter)
    outcome = control.step()
    assert outcome.committed and outcome.evaluations == 1
    assert adapter.step_index == 1
    assert control.feedback[0].objectives[0] == final_value ** 2
