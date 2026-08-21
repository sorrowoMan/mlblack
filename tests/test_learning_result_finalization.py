from __future__ import annotations

import numpy as np

from blackbase.resources import DataRef
from mlblack.core import Feedback, LearningProblem, ModelRepresentation, UnknownState
from mlblack.integrations import build_learning_solver
from nsgablack.adapters import FixedCandidateAdapter
from nsgablack.core.state.incumbent import IncumbentState


class _Problem(LearningProblem):
    def evaluate(self, model, state, context=None):
        del model, state, context
        return Feedback(objectives=[0.0], metrics={"loss": 0.0})


class _NumericEquivalentRepresentation(ModelRepresentation):
    def __init__(self, size: int = 1) -> None:
        self.size = int(size)

    def init(self, context):
        del context
        return UnknownState(np.zeros(self.size), metadata={"architecture": "initial"})

    def decode(self, state, context=None):
        del context
        return state.as_array()

    def equivalent(self, left, right):
        return np.array_equal(left.as_array(), right.as_array())


class _CaseRuntime:
    def __init__(self) -> None:
        self.artifact_refs: dict[str, DataRef] = {}

    def checkpoint(self) -> None:
        return None

    def publish_artifact(self, name, value, **kwargs):
        assert value is not None
        ref = DataRef(
            uri=f"memory://learning-result/{name}",
            kind=str(kwargs.get("kind", "artifact")),
            backend="memory",
        )
        self.artifact_refs[str(name)] = ref
        return ref


def _solver(*, size: int = 1, inline_limit: int = 64 * 1024):
    return build_learning_solver(
        problem=_Problem(),
        representation=_NumericEquivalentRepresentation(size),
        adapter=FixedCandidateAdapter(),
        run_name="result_finalization",
        result_inline_limit_bytes=inline_limit,
    )


def test_incumbent_token_has_priority_over_numeric_equivalence() -> None:
    solver = _solver()
    first = UnknownState([1.0], metadata={"architecture": "first"})
    second = UnknownState([1.0], metadata={"architecture": "second"})
    first_feedback = Feedback(objectives=[10.0], metrics={"source": "first"})
    second_feedback = Feedback(objectives=[1.0], metrics={"source": "second"})
    solver._record_ml_evaluation(
        first,
        first_feedback,
        "first-model",
        candidate_token="candidate:first",
    )
    solver._record_ml_evaluation(
        second,
        second_feedback,
        "second-model",
        candidate_token="candidate:second",
    )
    solver.set_incumbent(
        IncumbentState(
            candidate=[1.0],
            objectives=[1.0],
            constraint_violation=0.0,
            score=1.0,
            policy_id=str(solver.incumbent_scalarizer_id),
            policy_context=dict(solver.incumbent_scalarizer_context),
            candidate_token="candidate:second",
            metadata={"candidate.semantic_metadata": {"architecture": "second"}},
        )
    )

    solver._synchronize_ml_projection(record_history=False)

    assert solver.best_model == "second-model"
    assert solver.best_feedback is second_feedback
    assert solver.best_state.metadata["architecture"] == "second"


def test_large_trainer_result_fields_are_published_before_case_serialization() -> None:
    solver = _solver(size=512, inline_limit=1024)
    runtime = _CaseRuntime()
    solver.set_case_runtime(runtime)

    result = solver.fit(max_steps=20)
    payload = result.as_dict()

    assert result.best_model_ref is not None
    assert result.best_state_ref is not None
    assert result.history_ref is not None
    assert result.report_ref is not None
    assert payload["best_model"] is None
    assert payload["best_state"] is None
    assert payload["history"] == []
    assert payload["report"] == {}
    assert payload["best_state_ref"]["uri"].endswith("/trainer_best_state")
    assert payload["history_ref"]["uri"].endswith("/trainer_history")
    assert payload["report_ref"]["uri"].endswith("/trainer_report")
    assert set(runtime.artifact_refs).issuperset(
        {"best_model", "trainer_best_state", "trainer_history", "trainer_report"}
    )
