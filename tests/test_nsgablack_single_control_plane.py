from __future__ import annotations

import numpy as np
import pytest

from blackbase.plugin import PluginBase
from blackbase.resources import DataRef
from mlblack.core import ComputeBackendSession, ComputeBackendSpec
from mlblack.integrations import LearningSolver
from mlblack.pipeline.data_views import NumericDataView
from mlblack.presets import build_mlp_regression_trainer
from mlblack.representations.codecs import NeuralGraphCodec, NeuralGraphSpec
from nsgablack.core.composable_solver import ComposableSolver


class _LifecycleProbe(PluginBase):
    def __init__(self) -> None:
        super().__init__("single_control_probe")
        self.events: list[str] = []

    def on_solver_init(self, solver) -> None:
        assert isinstance(solver, ComposableSolver)
        self.events.append("init")

    def on_generation_start(self, generation: int) -> None:
        self.events.append(f"start:{generation}")

    def on_step(self, solver, generation: int) -> None:
        assert isinstance(solver, ComposableSolver)
        self.events.append(f"step:{generation}")

    def on_generation_end(self, generation: int) -> None:
        self.events.append(f"end:{generation}")

    def on_solver_finish(self, result) -> None:
        self.events.append("finish")


def _data() -> NumericDataView:
    X = np.linspace(-1.0, 1.0, 8, dtype=float).reshape(-1, 1)
    y = 1.5 + (2.0 * X[:, 0])
    return NumericDataView(X_train=X, y_train=y, feature_names=("x0",))


def test_standard_mlp_training_has_one_nsg_control_plane() -> None:
    pytest.importorskip("torch")
    control = build_mlp_regression_trainer(
        _data(),
        hidden_layers=(4,),
        optimizer="sgd",
        learning_rate=0.05,
        batch_size=4,
        shuffle=False,
    )
    probe = _LifecycleProbe()
    control.add_plugin(probe)

    result = control.fit(max_steps=2)

    assert isinstance(control, LearningSolver)
    assert isinstance(control, ComposableSolver)
    assert control.control_plane == "nsgablack.ComposableSolver"
    assert type(control.model_representation).__name__ == "NeuralGraphRepresentation"
    assert control.model_representation.graph_spec.metadata["route"] == "mlp"
    assert type(control.semantic_problem.semantic_problem).__name__ == (
        "TabularNeuralRegressionProblem"
    )
    assert len(result.history) == 2
    assert result.report["control_plane"] == control.control_plane
    assert probe.events == [
        "init",
        "start:0",
        "step:0",
        "end:0",
        "start:1",
        "step:1",
        "end:1",
        "finish",
    ]


def test_mlp_codec_and_torch_lowering_share_layout_and_mapping() -> None:
    pytest.importorskip("torch")
    spec = NeuralGraphSpec.mlp(
        input_dim=2,
        hidden_layers=(3,),
        output_dim=1,
        activation="tanh",
    )
    codec = NeuralGraphCodec(spec, random_seed=7)
    local_layout = codec.parameter_layout()
    values = np.linspace(-0.2, 0.3, local_layout.total_size, dtype=float)
    X = np.asarray([[0.0, 1.0], [1.0, -1.0], [0.5, 0.25]], dtype=float)

    local_model = codec.decode(values)
    context = {
        "backend.session": ComputeBackendSession(
            ComputeBackendSpec(name="torch", device="cpu")
        )
    }
    torch_layout = codec.parameter_layout(context)
    torch_model = codec.decode(values, context)

    assert torch_layout.shapes == local_layout.shapes
    assert torch_layout.names == local_layout.names
    assert torch_layout.metadata["backend"] == "torch"
    assert np.allclose(torch_model.predict(X), local_model.predict(X), atol=1e-6)


def test_single_control_plane_publishes_model_before_result_serialization() -> None:
    pytest.importorskip("torch")

    class _CaseRuntime:
        def __init__(self) -> None:
            self.artifact_refs: dict[str, DataRef] = {}

        def checkpoint(self) -> None:
            return None

        def publish_artifact(self, name, value, **kwargs):
            assert value is not None
            ref = DataRef(
                uri=f"memory://single-control/{name}",
                kind=str(kwargs.get("kind", "model")),
                backend="memory",
            )
            self.artifact_refs[str(name)] = ref
            return ref

    control = build_mlp_regression_trainer(
        _data(),
        hidden_layers=(2,),
        optimizer="adam",
        batch_size=4,
        shuffle=False,
    )
    control.set_case_runtime(_CaseRuntime())

    result = control.fit(max_steps=1)
    payload = result.as_dict()

    assert result.best_model_ref is not None
    assert payload["best_model"] is None
    assert payload["best_model_ref"]["uri"] == "memory://single-control/best_model"
    assert payload["best_state"]["values"]
