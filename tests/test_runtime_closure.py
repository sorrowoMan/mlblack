from __future__ import annotations

import importlib.util

import numpy as np

import mlblack
from mlblack.core import Feedback, LearningProblem, ModelRepresentation, UnknownState
from mlblack.integrations import LearningSolver, build_gradient_trainer, build_learning_solver
from nsgablack.adapters import FixedCandidateAdapter, GradientOptimizerAdapter


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
        return Feedback(objectives=[value * value], gradients=[1.0])


def test_gradient_training_has_one_nsg_control_plane() -> None:
    solver = build_gradient_trainer(
        problem=_GradientProblem(),
        representation=_PointRepresentation(),
        method="gradient.sgd",
        compute_backend="problem",
        learning_rate=0.1,
        run_name="single-control-plane",
    )
    result = solver.fit(max_steps=1)

    assert isinstance(solver, LearningSolver)
    assert isinstance(solver.adapter, GradientOptimizerAdapter)
    assert solver.control_plane == "nsgablack.ComposableSolver"
    assert np.allclose(solver.adapter.current_x, [0.9])
    assert np.allclose(solver.last_evaluated_population[0].as_array(), [1.0])
    assert result.report["optimization_control_plane"] == "nsgablack.ComposableSolver"


def test_resource_context_setter_rebuilds_only_ml_compute_session() -> None:
    solver = build_learning_solver(
        problem=_GradientProblem(),
        representation=_PointRepresentation(),
        adapter=FixedCandidateAdapter(),
        compute_backend={"name": "numpy", "device": "cpu"},
    )
    previous = solver.compute_backend_session
    solver.set_resource_context({
        "threads": 2,
        "device": "cuda:7",
        "compute_backend": "numpy",
        "namespace": "project.ml",
    })
    assert solver.compute_backend_session is not previous
    assert solver.compute_backend_session.requested_name == "numpy"
    assert solver.build_context()["backend.device"] == "cuda:7"


def test_legacy_control_plane_modules_and_classes_are_deleted() -> None:
    assert importlib.util.find_spec("mlblack.core.trainer") is None
    assert importlib.util.find_spec("mlblack.core.adapter") is None
    assert importlib.util.find_spec("mlblack.integrations.nsgablack_trainer_evaluator") is None
    for name in ("Trainer", "BlankTrainer", "ComposableTrainer", "OptimizerAdapter"):
        assert not hasattr(mlblack, name)
