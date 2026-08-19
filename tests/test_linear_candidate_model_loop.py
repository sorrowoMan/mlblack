from __future__ import annotations

import numpy as np

from mlblack.core import ArtifactBuilder, UnknownState
from mlblack.core.artifact_provider import select_model_serializer
from mlblack.models import LinearPointModel
from mlblack.pipeline.data_views import NumericDataView
from mlblack.presets import build_linear_point_trainer


def _linear_data() -> NumericDataView:
    X = np.linspace(-1.0, 1.0, 41, dtype=float).reshape(-1, 1)
    y = 1.5 + (2.0 * X[:, 0])
    return NumericDataView(
        X_train=X,
        y_train=y,
        feature_names=("x0",),
    )


def test_linear_candidate_codec_round_trip_is_explicit_and_detached() -> None:
    trainer = build_linear_point_trainer(_linear_data())
    candidate = UnknownState([1.5, 2.0], metadata={"source": "test"})

    model = trainer.decode_candidate(candidate)
    restored = trainer.encode_candidate(model)
    layout = trainer.representation_pipeline.describe()["codec"]["parameter_layout"]

    assert isinstance(model, LinearPointModel)
    assert np.allclose(restored.as_array(), candidate.as_array())
    assert layout["schema"] == "mlblack.linear_parameter_layout.v1"
    assert layout["total_size"] == 2
    assert [slot["name"] for slot in layout["slots"]] == ["intercept", "weights"]
    assert not model.weights.flags.writeable


def test_linear_problem_returns_objective_and_gradient_in_candidate_coordinates() -> None:
    trainer = build_linear_point_trainer(_linear_data(), l2=0.0)
    optimum = UnknownState([1.5, 2.0])

    feedback = trainer.evaluate_individual(optimum)

    assert feedback.objectives.shape == (2,)
    assert feedback.gradients is not None
    assert feedback.gradients.shape == optimum.as_array().shape
    assert feedback.loss == 0.0
    assert np.allclose(feedback.gradients, [0.0, 0.0], atol=1e-12)


def test_linear_candidate_optimization_produces_reusable_model_artifact() -> None:
    trainer = build_linear_point_trainer(
        _linear_data(),
        method="gradient.adam",
        learning_rate=0.08,
        random_seed=7,
    )
    initial = trainer.init_candidate()
    initial_loss = float(trainer.evaluate_individual(initial).loss)

    result = trainer.fit(max_steps=160)
    bundle = ArtifactBuilder().build(trainer, result)

    assert result.best_state is not None
    assert result.best_model is not None
    assert result.best_feedback is not None
    assert float(result.best_feedback.loss) < initial_loss * 1e-3
    assert np.allclose(result.best_state.as_array(), [1.5, 2.0], atol=2e-2)
    assert type(trainer.adapter).__module__.startswith("nsgablack.")
    assert trainer.optimizer_method == "gradient.adam"

    artifact = bundle.model_artifact
    assert artifact is not None
    assert isinstance(artifact.model, LinearPointModel)
    assert select_model_serializer(artifact.model) == "auto"
    restored_model = LinearPointModel.from_dict(artifact.model.as_dict())
    assert np.allclose(
        restored_model.predict(_linear_data().X_train),
        artifact.model.predict(_linear_data().X_train),
    )
