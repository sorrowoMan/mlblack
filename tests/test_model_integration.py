from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from mlblack.core import ArtifactBuilder
from mlblack.models import LinearPointModel, PredictionIOContract, PredictionInputSpec, PredictionIntegrationComponent
from mlblack.pipeline import ModelConditionedTargetComponent, build_model_conditioned_target
from mlblack.pipeline.data_views import NumericDataView
from mlblack.presets import build_orthogonal_linear_point_trainer


def test_model_conditioned_target_builds_residual_data() -> None:
    X = np.linspace(-1.0, 1.0, 8).reshape(-1, 1)
    y = 1.0 + (2.0 * X[:, 0]) + (0.75 * X[:, 0] ** 2)
    data = NumericDataView(X_train=X, y_train=y, feature_names=("x",))
    main_model = LinearPointModel(intercept=1.0, weights=np.asarray([2.0]), feature_names=("x",))

    residual_data = build_model_conditioned_target(
        data,
        main_model,
        mode="residual",
        reference_name="main",
    )

    assert residual_data.target_name == "target.residual"
    assert np.allclose(residual_data.y_train, 0.75 * X[:, 0] ** 2)
    assert residual_data.metadata["pipeline.model_conditioned_target"]["reference_name"] == "main"


def test_next_stage_trainer_can_fit_residual_and_integrate_models() -> None:
    X = np.linspace(-1.0, 1.0, 80).reshape(-1, 1)
    y = 1.0 + (2.0 * X[:, 0]) + (0.75 * X[:, 0] ** 2)
    data = NumericDataView(X_train=X, y_train=y, feature_names=("x",))
    main_model = LinearPointModel(intercept=1.0, weights=np.asarray([2.0]), feature_names=("x",))

    residual_data = ModelConditionedTargetComponent().build(data, reference_model=main_model)
    residual_trainer = build_orthogonal_linear_point_trainer(
        residual_data,
        learning_rate=0.2,
        energy_threshold=None,
        run_name="residual_stage_inner_trainer",
    )
    result = residual_trainer.fit(max_steps=120)
    assert result.best_model is not None

    integrated_model = PredictionIntegrationComponent.additive(
        component_order=("main", "residual"),
        metadata={"case": "main_plus_residual"},
    ).compose(
        {"main": main_model, "residual": result.best_model},
        metadata={"orchestration_owner": "nsgablack"},
    )

    base_mse = float(np.mean((main_model.predict(X) - y) ** 2))
    integrated_mse = float(np.mean((integrated_model.predict(X) - y) ** 2))
    assert integrated_mse < base_mse * 0.05

    bundle = ArtifactBuilder().build(
        SimpleNamespace(run_name="integrated_residual_case", context={}),
        SimpleNamespace(
            best_model=integrated_model,
            report={
                "run_name": "integrated_residual_case",
                "problem": {"name": "integrated_regression", "head": "point"},
                "representation": {"name": "prediction_integration"},
                "adapter": {"name": "nsgablack_serial_stage"},
                "best_metrics": {"integrated.mse": integrated_mse, "main.mse": base_mse},
            },
        ),
    )
    assert bundle.model_artifact is not None
    assert bundle.model_artifact.describe()["artifact_type"] == "integrated_model"


def test_integrated_prediction_model_routes_named_component_inputs() -> None:
    class _TabularModel:
        def predict(self, X):
            arr = np.asarray(X, dtype=float)
            return arr[:, 0]

    class _ImageModel:
        def predict(self, X):
            arr = np.asarray(X, dtype=float)
            return np.mean(arr, axis=(1, 2, 3))

    io_contract = PredictionIOContract.by_component(
        {
            "tabular": PredictionInputSpec(key="tabular", ndim=2, n_features=1),
            "image": PredictionInputSpec(key="image", ndim=4),
        }
    )
    model = PredictionIntegrationComponent.additive(
        component_order=("tabular", "image"),
        io_contract=io_contract,
    ).compose({"tabular": _TabularModel(), "image": _ImageModel()})

    tabular = np.asarray([[1.0], [2.0], [3.0]])
    image = np.ones((3, 1, 2, 2), dtype=float) * 0.5

    assert np.allclose(model.predict({"tabular": tabular, "image": image}), [1.5, 2.5, 3.5])
    assert model.describe()["io_contract"]["component_inputs"]["image"]["ndim"] == 4


def test_integrated_prediction_model_fails_on_input_or_output_contract_violation() -> None:
    class _PointModel:
        def predict(self, X):
            return np.asarray(X, dtype=float)[:, 0]

    class _BadMatrixModel:
        def predict(self, X):
            rows = np.asarray(X, dtype=float).shape[0]
            return np.ones((rows, 2), dtype=float)

    io_contract = PredictionIOContract.by_component(
        {"tabular": PredictionInputSpec(key="tabular", ndim=2, n_features=1)}
    )
    missing_input_model = PredictionIntegrationComponent.additive(
        component_order=("tabular",),
        io_contract=io_contract,
    ).compose({"tabular": _PointModel()})

    with pytest.raises(KeyError, match="missing input"):
        missing_input_model.predict({"wrong": np.ones((2, 1), dtype=float)})

    bad_output_model = PredictionIntegrationComponent.additive(
        component_order=("tabular", "bad"),
        io_contract=PredictionIOContract.by_component(
            {
                "tabular": PredictionInputSpec(key="tabular", ndim=2, n_features=1),
                "bad": PredictionInputSpec(key="tabular", ndim=2, n_features=1),
            }
        ),
    ).compose({"tabular": _PointModel(), "bad": _BadMatrixModel()})

    with pytest.raises(ValueError, match="point vector"):
        bad_output_model.predict({"tabular": np.ones((2, 1), dtype=float)})
