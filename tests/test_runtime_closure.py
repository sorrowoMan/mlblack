from __future__ import annotations

import numpy as np
import pytest
from blackbase.plugin import Plugin

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
from mlblack.core.trainer_stage import CompletionPolicy, SerialTrainer, StageSpec


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


def test_serial_trainer_injects_parent_runtime_into_child_stage() -> None:
    child = _StepOnlyTrainer()
    trainer = SerialTrainer(
        [StageSpec("child", lambda: child, completion=CompletionPolicy(max_steps=1))],
        resource_context={
            "threads": 2,
            "namespace": "project.outer",
            "lease": {"lease_id": "project-lease"},
        },
    )

    trainer.fit()

    assert child.resource_context.nested is True
    assert child.resource_context.lease["lease_id"] == "project-lease"
    assert child.resource_context.namespace.endswith("stage.0.child")
    assert child.context_store is trainer.context_store
    assert child.snapshot_store is trainer.snapshot_store
    assert child.setup_called is True
    assert child.teardown_called is True


def test_serial_trainer_step_only_stage_tears_down_on_failure() -> None:
    child = _StepOnlyTrainer(fail=True)
    trainer = SerialTrainer(
        [StageSpec("failing", lambda: child, completion=CompletionPolicy(max_steps=1))]
    )

    with pytest.raises(RuntimeError, match="stage failed"):
        trainer.fit()

    assert child.teardown_called is True


class _ResultTrainer:
    def __init__(self, label: str, score: float) -> None:
        self.label = label
        self.score = float(score)
        self.resource_context = None
        self.context_store = None
        self.snapshot_store = None

    def set_resource_context(self, context):
        self.resource_context = context

    def set_context_store(self, store):
        self.context_store = store

    def set_snapshot_store(self, store):
        self.snapshot_store = store

    def fit(self, max_steps=None):
        del max_steps
        state = UnknownState(
            values=[self.score], metadata={"output_stage": self.label}
        )
        feedback = Feedback(objectives=[self.score])
        return TrainerResult(
            best_state=state,
            best_model={"stage": self.label},
            best_objectives=[self.score],
            best_feedback=feedback,
            history=({"stage": self.label},),
            metadata={"child": self.label},
        )


def test_serial_trainer_returns_last_stage_result_by_default() -> None:
    first = _ResultTrainer("first", 3.0)
    last = _ResultTrainer("last", 1.0)
    trainer = SerialTrainer(
        [
            StageSpec("first", lambda: first),
            StageSpec("last", lambda: last),
        ]
    )

    result = trainer.fit()

    assert result.best_state.metadata["output_stage"] == "last"
    assert result.best_model == {"stage": "last"}
    assert np.array_equal(result.best_objectives, [1.0])
    assert np.array_equal(result.best_feedback.objectives, [1.0])
    assert trainer.best_state is result.best_state
    assert trainer.best_model is result.best_model
    assert trainer.best_feedback is result.best_feedback
    assert trainer.best_score == pytest.approx(1.0)
    assert result.metadata["output_stage"] == "last"
    assert result.report["result_policy"] == "selected_stage"
    assert [entry["stage_name"] for entry in result.history] == ["first", "last"]


def test_serial_trainer_run_executes_the_composite_fit_lifecycle() -> None:
    created = []

    def build_child():
        child = _ResultTrainer("via-run", 2.0)
        created.append(child)
        return child

    trainer = SerialTrainer([StageSpec("via-run", build_child)])

    result = trainer.run(max_steps=999)

    assert len(created) == 1
    assert result.best_state.metadata["output_stage"] == "via-run"
    assert result.best_model == {"stage": "via-run"}
    assert np.array_equal(result.best_objectives, [2.0])
    assert [entry["stage_name"] for entry in trainer.stage_results] == ["via-run"]


def test_serial_trainer_can_select_an_explicit_output_stage() -> None:
    trainer = SerialTrainer(
        [
            StageSpec("first", lambda: _ResultTrainer("first", 3.0)),
            StageSpec("last", lambda: _ResultTrainer("last", 1.0)),
        ],
        output_stage="first",
    )

    result = trainer.fit()

    assert result.best_state.metadata["output_stage"] == "first"
    assert np.array_equal(result.best_objectives, [3.0])
    assert result.report["output_stage"] == "first"


def test_serial_trainer_can_aggregate_child_results() -> None:
    def aggregate(results):
        objectives = np.asarray(
            [float(result.best_objectives[0]) for result in results], dtype=float
        )
        state = UnknownState(values=[objectives.mean()], metadata={"aggregate": True})
        feedback = Feedback(objectives=[objectives.mean()])
        return TrainerResult(
            best_state=state,
            best_model={"aggregate": True},
            best_objectives=feedback.objectives,
            best_feedback=feedback,
            metadata={"children": len(results)},
        )

    trainer = SerialTrainer(
        [
            StageSpec("first", lambda: _ResultTrainer("first", 3.0)),
            StageSpec("last", lambda: _ResultTrainer("last", 1.0)),
        ],
        result_aggregator=aggregate,
    )

    result = trainer.fit()

    assert result.best_state.metadata["aggregate"] is True
    assert np.array_equal(result.best_objectives, [2.0])
    assert result.metadata["children"] == 2
    assert result.metadata["result_policy"] == "aggregator"
    assert result.report["result_policy"] == "aggregator"
    assert trainer.history == trainer.stage_results
    assert trainer.step_index == 1


class _LifecyclePlugin(Plugin):
    def __init__(self, events) -> None:
        super().__init__("serial-lifecycle")
        self.events = events

    def on_solver_init(self, solver):
        del solver
        self.events.append("init")

    def on_solver_finish(self, result):
        del result
        self.events.append("finish")

    def on_error(self, error, context=None):
        del error, context
        self.events.append("error")


class _LifecycleSerialTrainer(SerialTrainer):
    def __init__(self, *args, events, **kwargs) -> None:
        self.events = events
        super().__init__(*args, **kwargs)

    def setup(self) -> None:
        self.events.append("setup")

    def teardown(self) -> None:
        self.events.append("teardown")


def test_serial_trainer_runs_parent_lifecycle_once_on_success() -> None:
    events = []
    trainer = _LifecycleSerialTrainer(
        [StageSpec("only", lambda: _ResultTrainer("only", 1.0))],
        events=events,
    )
    trainer.add_plugin(_LifecyclePlugin(events))

    trainer.fit()

    assert events == ["setup", "init", "teardown", "finish"]


def test_serial_trainer_runs_parent_error_and_teardown_once_on_failure() -> None:
    events = []
    trainer = _LifecycleSerialTrainer(
        [
            StageSpec(
                "failing",
                lambda: _StepOnlyTrainer(fail=True),
                completion=CompletionPolicy(max_steps=1),
            )
        ],
        events=events,
    )
    trainer.add_plugin(_LifecyclePlugin(events))

    with pytest.raises(RuntimeError, match="stage failed"):
        trainer.fit()

    assert events == ["setup", "init", "error", "teardown"]
