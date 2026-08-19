from __future__ import annotations

import time

import numpy as np
import pytest

from blackbase.resources import CancellationRef, CancellationToken, CaseDeadlineExceeded
from mlblack.core import Feedback, LearningProblem, ModelRepresentation, UnknownState
from mlblack.integrations import build_learning_solver
from nsgablack.adapters import FixedCandidateAdapter


class _PointRepresentation(ModelRepresentation):
    def init(self, context):
        del context
        return UnknownState([0.5])

    def decode(self, state, context=None):
        del context
        return state.as_array().copy()


class _SlowProblem(LearningProblem):
    def __init__(self) -> None:
        self.executed = 0

    def evaluate(self, model, state, context=None):
        del model, context
        self.executed += 1
        time.sleep(0.01)
        value = float(state.as_array()[0])
        return Feedback(objectives=[value * value])


class _TokenRuntime:
    def __init__(self, token: CancellationToken) -> None:
        self.token = token

    def checkpoint(self) -> None:
        self.token.checkpoint()


class _CountingRuntime:
    def __init__(self) -> None:
        self.count = 0

    def checkpoint(self) -> None:
        self.count += 1


def _solver(problem: LearningProblem):
    return build_learning_solver(
        problem=problem,
        representation=_PointRepresentation(),
        adapter=FixedCandidateAdapter(),
        run_name="control",
    )


def test_learning_solver_deadline_interrupts_running_nsg_loop() -> None:
    problem = _SlowProblem()
    solver = _solver(problem)
    ref = CancellationRef(backend="memory", deadline_at=time.time() + 0.04)
    solver.set_case_runtime(_TokenRuntime(CancellationToken(ref)))

    with pytest.raises(CaseDeadlineExceeded):
        solver.fit(max_steps=100)

    assert 0 < problem.executed < 100


def test_learning_solver_checks_shared_control_around_evaluation() -> None:
    solver = _solver(_SlowProblem())
    runtime = _CountingRuntime()
    solver.set_case_runtime(runtime)

    feedback = solver.evaluate_individual(UnknownState(np.asarray([0.5])))

    assert feedback.ok
    assert runtime.count >= 2
