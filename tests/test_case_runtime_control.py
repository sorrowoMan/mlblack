from __future__ import annotations

import time

import numpy as np
import pytest

from blackbase.resources import CancellationRef, CancellationToken, CaseDeadlineExceeded
from mlblack.core import BlankTrainer, Feedback, LearningProblem, ModelRepresentation, UnknownState


class _PointRepresentation(ModelRepresentation):
    def init(self, context):
        del context
        return UnknownState([0.5])

    def decode(self, state, context=None):
        del context
        return state.as_array().copy()


class _PointProblem(LearningProblem):
    def evaluate(self, model, state, context=None):
        del model, context
        value = float(state.as_array()[0])
        return Feedback(objectives=[value * value])


class _SlowTrainer(BlankTrainer):
    def __init__(self) -> None:
        super().__init__(run_name="slow")
        self.executed = 0

    def step(self, context=None):
        del context
        self.executed += 1
        time.sleep(0.01)
        return {"step": self.executed}


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


def test_trainer_deadline_interrupts_a_running_step_loop() -> None:
    trainer = _SlowTrainer()
    ref = CancellationRef(backend="memory", deadline_at=time.time() + 0.04)
    trainer.set_case_runtime(_TokenRuntime(CancellationToken(ref)))

    with pytest.raises(CaseDeadlineExceeded):
        trainer.fit(max_steps=100)

    assert 0 < trainer.executed < 100


def test_trainer_checks_control_around_evaluation_and_snapshot_commit() -> None:
    trainer = BlankTrainer(
        problem=_PointProblem(),
        representation=_PointRepresentation(),
        run_name="control",
    )
    runtime = _CountingRuntime()
    trainer.set_case_runtime(runtime)
    state = UnknownState(np.asarray([0.5]))

    feedback = trainer.evaluate_individual(state)
    evaluation_checkpoints = runtime.count
    assert feedback.ok
    assert evaluation_checkpoints >= 2

    trainer.write_population_snapshot((state,), (feedback,))
    assert runtime.count > evaluation_checkpoints
