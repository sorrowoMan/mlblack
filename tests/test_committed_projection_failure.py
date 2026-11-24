import pytest

from blackbase.plugin import PluginBase
from mlblack.core import Feedback, LearningProblem, ModelRepresentation, UnknownState
from mlblack.integrations import build_learning_solver
from nsgablack.adapters import FixedCandidateAdapter
from nsgablack.core.semantic_extension import CommittedStepError


def test_projection_failure_retains_committed_authority_progress_and_lifecycle():
    class Representation(ModelRepresentation):
        def init(self, context):
            return UnknownState([0.0])

        def decode(self, state, context=None):
            return state.as_array()

    class Problem(LearningProblem):
        def evaluate(self, model, state, context=None):
            return Feedback(objectives=[0.0])

    class Observer(PluginBase):
        def __init__(self):
            super().__init__("observer")
            self.outcomes = []
            self.commits = []

        def on_step_attempt_end(self, attempt, logical_step, outcome):
            self.outcomes.append(outcome)

        def on_generation_committed(self, generation, outcome):
            self.commits.append(outcome)

    solver = build_learning_solver(problem=Problem(), representation=Representation(),
                                   adapter=FixedCandidateAdapter(), run_name="projection_failure")
    observer = Observer()
    solver.add_plugin(observer)

    def fail_projection(*, record_history):
        raise RuntimeError("projection unavailable")

    solver._synchronize_ml_projection = fail_projection
    with pytest.raises(CommittedStepError, match="projection unavailable"):
        solver.fit(max_steps=1)
    assert solver.run_progress_steps == solver.run_progress_attempts == 1
    assert solver.get_population_authority_identity() is not None
    assert observer.outcomes[-1]["committed"] is True
    assert observer.commits[-1]["committed"] is True
    assert solver.get_last_evaluation_evidence_record().terminal_verified is True
