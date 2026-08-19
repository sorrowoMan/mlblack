from __future__ import annotations

import numpy as np

from blackbase.types import Feedback, UnknownState
from mlblack.core import LearningProblem, ModelRepresentation
from mlblack.bias import ObjectiveWeightBias
from mlblack.integrations import LearningSolver, build_optimization_adapter
from nsgablack.adapters import AlgorithmAdapter
from mlblack.models import FittedEstimatorModel
from mlblack.pipeline.data_views import NumericDataView
from mlblack.presets import (
    build_orthogonal_linear_interval_trainer,
    build_orthogonal_linear_point_trainer,
    build_orthogonal_logistic_classification_trainer,
    build_tree_estimator_search_trainer,
)


def _regression_data() -> NumericDataView:
    X = np.linspace(-1.0, 1.0, 12, dtype=float).reshape(-1, 1)
    y = 1.0 + (2.5 * X[:, 0])
    return NumericDataView(X_train=X, y_train=y, feature_names=("x0",))


def test_problem_owned_gradient_uses_nsgablack_stable_method() -> None:
    trainer = build_orthogonal_linear_point_trainer(
        _regression_data(),
        learning_rate=0.05,
    )

    result = trainer.fit(max_steps=2)

    assert type(trainer.adapter).__name__ == "GradientOptimizerAdapter"
    assert type(trainer.adapter).__module__.startswith("nsgablack.")
    assert trainer.optimizer_method == "gradient.sgd"
    assert trainer.evaluation_mode == "problem"
    assert not hasattr(trainer, "evaluation_provider")
    assert result.report["adapter"]["state"]["step_index"] == 2


def test_black_box_ml_presets_share_nsgablack_gaussian_search() -> None:
    interval = build_orthogonal_linear_interval_trainer(
        _regression_data(),
        population_size=3,
    )
    classification = build_orthogonal_logistic_classification_trainer(
        NumericDataView(
            X_train=np.asarray([[0.0], [0.2], [0.8], [1.0]], dtype=float),
            y_train=np.asarray([0, 0, 1, 1], dtype=int),
        ),
        population_size=3,
    )
    estimator = build_tree_estimator_search_trainer(
        _regression_data(),
        population_size=3,
    )

    for trainer in (interval, classification, estimator):
        assert trainer.control_plane == "nsgablack.ComposableSolver"
        assert type(trainer.adapter).__name__ == "GaussianSearchAdapter"
        assert type(trainer.adapter).__module__.startswith("nsgablack.")
    assert estimator.adapter.config.initialization == "center"


def test_estimator_search_returns_the_fitted_model_that_was_evaluated() -> None:
    trainer = build_tree_estimator_search_trainer(
        _regression_data(),
        population_size=2,
        params={"n_estimators": 3},
    )

    result = trainer.fit(max_steps=1)

    assert isinstance(result.best_model, FittedEstimatorModel)
    assert hasattr(result.best_model.estimator, "estimators_")
    assert trainer.semantic_problem.build_model_artifact(result.best_model) is result.best_model


def test_ml_vocabulary_resolves_without_exposing_adapter_repository() -> None:
    gradient = build_optimization_adapter("adam", learning_rate=0.01)
    search = build_optimization_adapter("random_search", population_size=2)

    assert type(gradient).__name__ == "GradientOptimizerAdapter"
    assert gradient.method_ids == (
        "gradient.sgd",
        "gradient.adam",
        "gradient.adamw",
    )
    assert type(search).__name__ == "GaussianSearchAdapter"
    assert search.method_ids == ("search.random_gaussian",)


def test_ml_bias_and_completion_policy_project_into_nsg_control() -> None:
    class _OneStep:
        def is_complete(self, *, step, elapsed, ctx):
            del elapsed, ctx
            return int(step) >= 1

    control = build_orthogonal_linear_point_trainer(
        _regression_data(),
        learning_rate=0.05,
    )
    control.add_bias(ObjectiveWeightBias((0.5, 1.0)))
    control.set_completion_policy(_OneStep())

    result = control.fit(max_steps=5)

    assert len(result.history) == 1
    assert result.best_feedback is not None
    assert result.best_feedback.metrics["bias.objective_weight_applied"] is True
    assert result.report["biases"][0]["name"] == "objective_weight_bias"
    assert result.report["optimization_runtime"]["steps_executed"] == 1
    assert result.report["optimization_runtime"]["steps"] == 1
    assert result.report["optimization_runtime"]["termination_reason"] == "completion_policy"


def test_unknown_state_metadata_survives_the_numeric_control_plane() -> None:
    class _Representation(ModelRepresentation):
        def init(self, context):
            del context
            return UnknownState([0.0], metadata={"model_family": "a"})

        def decode(self, state, context=None):
            del context
            return str(state.metadata["model_family"])

    class _Problem(LearningProblem):
        def evaluate(self, model, state, context):
            del context
            assert model == state.metadata["model_family"]
            score = 0.0 if model == "b" else 1.0
            return Feedback(objectives=[score], metrics={"model_family": model})

    class _TwoSemanticCandidates(AlgorithmAdapter):
        def propose(self, control, context):
            del control, context
            return (
                UnknownState([0.0], metadata={"model_family": "a"}),
                UnknownState([0.0], metadata={"model_family": "b"}),
            )

        def update(self, control, candidates, feedback, context):
            del control, candidates, feedback, context

    control = LearningSolver(
        problem=_Problem(),
        representation=_Representation(),
        adapter=_TwoSemanticCandidates(name="two_semantic_candidates"),
    )

    result = control.fit(max_steps=1)

    assert np.array_equal(
        control.last_evaluated_population[0].as_array(),
        control.last_evaluated_population[1].as_array(),
    )
    assert control.last_evaluated_population[0].metadata["model_family"] == "a"
    assert control.last_evaluated_population[1].metadata["model_family"] == "b"
    assert result.best_state is not None
    assert result.best_state.metadata["model_family"] == "b"
    assert result.best_feedback is not None
    assert result.best_feedback.metrics["model_family"] == "b"
    assert result.best_model == "b"
