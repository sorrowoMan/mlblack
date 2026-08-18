from __future__ import annotations

import numpy as np
import pytest

from mlblack.adapters import GradientDescentAdapter, GradientDescentConfig
from mlblack.core import (
    BlankTrainer,
    Feedback,
    LearningProblem,
    ModelRepresentation,
    OptimizerAdapter,
    Trainer,
    TrainerResult,
    UnknownState,
)


class _PointRepresentation(ModelRepresentation):
    def init(self, context):
        del context
        return UnknownState(values=np.asarray([1.0]), metadata={"source": "init"})

    def decode(self, state, context=None):
        del context
        return state.as_array().copy()


class _GradientProblem(LearningProblem):
    def evaluate(self, model, state, context=None):
        del model, context
        value = float(state.as_array()[0])
        return Feedback(objectives=np.asarray([value * value]), gradients=np.asarray([1.0]))


def test_trainer_snapshot_commits_adapter_authoritative_state_without_stale_feedback() -> None:
    trainer = Trainer(
        problem=_GradientProblem(),
        representation=_PointRepresentation(),
        adapter=GradientDescentAdapter(GradientDescentConfig(learning_rate=0.1)),
        run_name="authoritative-state",
    )

    trainer.fit(max_steps=1)

    assert np.allclose(trainer.population[0].as_array(), [0.9])
    assert trainer.feedback == tuple()
    assert np.allclose(trainer.last_evaluated_population[0].as_array(), [1.0])
    snapshot_key = trainer.context_store.get("last_population_snapshot")
    snapshot = trainer.read_snapshot(snapshot_key)
    assert np.allclose(snapshot["candidates"][0].as_array(), [0.9])
    assert snapshot["objectives"] is None
    assert np.allclose(snapshot["evaluated_candidates"][0].as_array(), [1.0])
    assert np.allclose(snapshot["evaluated_objectives"], [[1.0]])
    assert snapshot["metadata"]["feedback_aligned"] is False


class _MetadataChangingAdapter(OptimizerAdapter):
    name = "metadata-changing"

    def __init__(self) -> None:
        self.current = None

    def propose(self, control, context):
        del control, context
        return (UnknownState(values=[1.0], metadata={"decoder_branch": "a"}),)

    def update(self, control, candidates, feedback, context):
        del control, feedback, context
        self.current = UnknownState(
            values=candidates[0].as_array(),
            metadata={"decoder_branch": "b"},
        )

    def get_population(self):
        return None if self.current is None else (self.current,)


def test_feedback_alignment_rejects_same_values_with_different_semantic_metadata() -> None:
    trainer = Trainer(
        problem=_GradientProblem(),
        representation=_PointRepresentation(),
        adapter=_MetadataChangingAdapter(),
        run_name="metadata-identity",
    )

    trainer.fit(max_steps=1)

    assert np.array_equal(trainer.population[0].as_array(), [1.0])
    assert trainer.population[0].metadata["decoder_branch"] == "b"
    assert trainer.feedback == tuple()
    snapshot = trainer.read_snapshot(trainer.context_store.get("last_population_snapshot"))
    assert snapshot["metadata"]["feedback_aligned"] is False


def test_resource_context_setter_rebuilds_compute_backend_session() -> None:
    trainer = Trainer(compute_backend={"name": "numpy", "device": "cpu"})
    previous = trainer.compute_backend_session

    trainer.set_resource_context(
        {
            "threads": 2,
            "device": "cuda:7",
            "compute_backend": "numpy",
            "namespace": "project.ml",
        }
    )

    assert trainer.compute_backend_session is not previous
    assert trainer.compute_backend_session.requested_name == "numpy"
    assert trainer.compute_backend_session.device == "cuda:7"
    assert trainer.build_context()["backend.device"] == "cuda:7"


class _StepOnlyTrainer:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = bool(fail)
        self.resource_context = None
        self.context_store = None
        self.snapshot_store = None
        self.setup_called = False
        self.teardown_called = False

    def set_resource_context(self, context):
        self.resource_context = context

    def set_context_store(self, store):
        self.context_store = store

    def set_snapshot_store(self, store):
        self.snapshot_store = store

    def setup(self):
        self.setup_called = True

    def step(self):
        if self.fail:
            raise RuntimeError("stage failed")

    def teardown(self):
        self.teardown_called = True


class _FitOverrideTrainer(BlankTrainer):
    def __init__(self) -> None:
        super().__init__()
        self.received_max_steps = None

    def fit(self, max_steps=100):
        self.received_max_steps = int(max_steps)
        return TrainerResult(metadata={"fit_override": True})


def test_blank_trainer_run_dynamically_dispatches_to_subclass_fit() -> None:
    trainer = _FitOverrideTrainer()

    result = trainer.run(max_steps=7)

    assert trainer.received_max_steps == 7
    assert result.metadata["fit_override"] is True
