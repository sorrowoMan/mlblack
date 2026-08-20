from __future__ import annotations

import numpy as np
import pytest

from blackbase.evaluation import (
    StateMaterializationRequest,
    StateTransitionRequest,
    StateVersionConflict,
)
from blackbase.state_ref import StateRef
from mlblack.assembly import build_trainer
from mlblack.backends.torch_neural import (
    TorchEvaluationProvider,
    TorchEvaluationProviderConfig,
)
from mlblack.integrations import build_gradient_trainer
from mlblack.pipeline.data_views import NumericDataView
from mlblack.pipeline.datasets import (
    DatasetStreamConfig,
    NumericBatchSchedule,
)
from mlblack.presets import (
    build_mlp_regression_trainer,
    build_tabular_tabnet_classification_trainer,
    build_tabular_tabnet_regression_trainer,
    build_temporal_lstm_forecast_trainer,
)
from mlblack.problems import TabularNeuralRegressionProblem
from mlblack.representations import NeuralGraphRepresentation


def _data() -> NumericDataView:
    X = np.linspace(-1.0, 1.0, 8, dtype=float).reshape(-1, 1)
    y = 1.5 + 2.0 * X[:, 0]
    return NumericDataView(X_train=X, y_train=y, feature_names=("x0",))


def test_numeric_batch_schedule_restores_the_exact_next_batch() -> None:
    data = _data()
    config = DatasetStreamConfig(batch_size=3, shuffle=True, seed=17)
    schedule = NumericBatchSchedule(data, config)

    schedule.next_train()
    schedule.next_train()
    checkpoint = dict(schedule.get_state())
    expected = schedule.next_train()

    restored = NumericBatchSchedule(data, config)
    restored.set_state(checkpoint)
    actual = restored.next_train()

    assert actual.indices == expected.indices
    assert actual.metadata == expected.metadata
    assert np.allclose(actual.X, expected.X)
    assert np.allclose(actual.y, expected.y)


def test_shared_checkpoint_components_restore_provider_data_schedule() -> None:
    from nsgablack.plugins import CheckpointResumePlugin

    data = _data()
    config = DatasetStreamConfig(batch_size=3, shuffle=True, seed=31)

    def build(run_name: str):
        schedule = NumericBatchSchedule(data, config)
        trainer = build_gradient_trainer(
            problem=TabularNeuralRegressionProblem(data),
            representation=_representation(31),
            method="gradient.adam",
            data_schedule=schedule,
            provider_config=TorchEvaluationProviderConfig(publish_state_refs=True),
            run_name=run_name,
        )
        return trainer, schedule

    source, source_schedule = build("shared_checkpoint_source")
    source_schedule.next_train()
    source_schedule.next_train()
    writer = CheckpointResumePlugin()
    writer.attach(source)
    payload = writer._build_payload(solver=source, reason="provider-schedule")
    expected = source_schedule.next_train()

    target, target_schedule = build("shared_checkpoint_target")
    reader = CheckpointResumePlugin()
    reader.attach(target)
    restored = reader._apply_component_states(
        target,
        payload["stateful_components"],
    )
    actual = target_schedule.next_train()

    assert "evaluation_provider" in restored
    assert "data_schedule" not in restored
    assert actual.indices == expected.indices


def test_shared_checkpoint_rejects_provider_configuration_mismatch() -> None:
    from nsgablack.plugins import CheckpointResumeConfig, CheckpointResumePlugin

    data = _data()

    def build(batch_size: int):
        schedule = NumericBatchSchedule(
            data,
            DatasetStreamConfig(batch_size=batch_size, shuffle=True, seed=31),
        )
        return build_gradient_trainer(
            problem=TabularNeuralRegressionProblem(data),
            representation=_representation(31),
            method="gradient.adam",
            data_schedule=schedule,
            provider_config=TorchEvaluationProviderConfig(publish_state_refs=True),
        )

    source = build(3)
    writer = CheckpointResumePlugin()
    writer.attach(source)
    payload = writer._build_payload(solver=source, reason="provider-identity")

    target = build(4)
    reader = CheckpointResumePlugin(
        config=CheckpointResumeConfig(strict=True)
    )
    reader.attach(target)
    with pytest.raises(ValueError, match="configuration mismatch"):
        reader._apply_component_states(target, payload["stateful_components"])


def _representation(seed: int) -> NeuralGraphRepresentation:
    return NeuralGraphRepresentation.mlp(
        input_dim=1,
        hidden_layers=(2,),
        output_dim=1,
        random_seed=seed,
        representation_name="test_mlp",
    )


def test_provider_default_identity_includes_representation_semantics() -> None:
    problem = TabularNeuralRegressionProblem(_data())
    first = TorchEvaluationProvider(problem, _representation(3))
    second = TorchEvaluationProvider(
        problem,
        NeuralGraphRepresentation.mlp(
            input_dim=1,
            hidden_layers=(3,),
            output_dim=1,
            random_seed=3,
            representation_name="test_mlp",
        ),
    )

    assert first.problem_id == second.problem_id
    assert first.representation_id != second.representation_id
    assert first.spec.provider_id != second.spec.provider_id


def test_mlp_preset_uses_nsg_adapter_and_torch_provider() -> None:
    pytest.importorskip("torch")
    trainer = build_mlp_regression_trainer(
        _data(),
        hidden_layers=(4,),
        optimizer="sgd",
        learning_rate=0.05,
        batch_size=4,
        shuffle=False,
        device="cpu",
        run_name="unified_gradient_test",
    )

    result = trainer.fit(max_steps=2)

    assert type(trainer.adapter).__name__ == "GradientOptimizerAdapter"
    assert type(trainer.adapter).__module__.startswith("nsgablack.")
    assert trainer.optimizer_method == "gradient.sgd"
    assert trainer.evaluation_provider.spec.compute_backend == "torch"
    assert trainer.evaluation_provider.spec.problem_ids == (
        trainer.problem.problem_id,
    )
    assert trainer.evaluation_provider.get_state()["evaluation_count"] == 2
    assert trainer.evaluation_provider.get_state()["data_schedule"]["epoch_index"] == 1
    assert trainer.adapter.get_state()["step_index"] == 2
    assert trainer.adapter.get_state()["provider_transition"]["count"] == 2
    assert trainer.last_evaluated_feedback[0].gradients is not None
    assert trainer.last_evaluated_feedback[0].signals["batch_indices"] == (4, 5, 6, 7)
    assert (
        trainer.last_evaluated_feedback[0].info["evaluation_binding"]["device"]
        == "cpu"
    )
    assert result.report["adapter"]["name"] == "gradient_optimizer"
    assert result.report["problem"]["evaluation_gateway"] == "blackbase"
    assert (
        result.report["problem"]["evaluation_provider"]["provider_id"]
        == trainer.evaluation_provider.spec.provider_id
    )


def test_assembly_resolves_ml_optimizer_vocabulary_to_stable_method() -> None:
    trainer = build_trainer(
        {
            "preset": "mlp_regression",
            "params": {
                "hidden_layers": (2,),
                "optimizer": "adam",
                "batch_size": 4,
            },
        },
        data=_data(),
    )

    assert trainer.optimizer_method == "gradient.adam"
    assert type(trainer.adapter).__name__ == "GradientOptimizerAdapter"
    assert type(trainer.evaluation_provider).__name__ == "TorchEvaluationProvider"


def test_unified_builder_runs_device_only_gradients_with_checkpoint_slot_shadow() -> None:
    pytest.importorskip("torch")
    representation = _representation(7)
    trainer = build_gradient_trainer(
        problem=TabularNeuralRegressionProblem(_data()),
        representation=representation,
        method="gradient.adam",
        provider_config=TorchEvaluationProviderConfig(
            publish_state_refs=True,
            inline_gradients=False,
        ),
    )

    trainer.fit(max_steps=2)
    feedback = trainer.last_evaluated_feedback[0]
    adapter_state = trainer.adapter.get_state()

    assert feedback.gradients is None
    assert isinstance(feedback.gradient_ref, StateRef)
    assert adapter_state["provider_transition"]["count"] == 2
    assert adapter_state["provider_transition"]["needs_slot_seed"] is True
    assert adapter_state["first_moment"] is not None
    assert adapter_state["second_moment"] is not None
    assert trainer.evaluation_provider.get_state()["live_state_count"] == 0


def test_torch_provider_executes_version_fenced_gradient_transition() -> None:
    pytest.importorskip("torch")
    data = _data()
    representation = _representation(11)
    trainer = build_gradient_trainer(
        problem=TabularNeuralRegressionProblem(data),
        representation=representation,
        method="gradient.sgd",
        learning_rate=0.05,
        max_gradient_norm=None,
        provider_config=TorchEvaluationProviderConfig(
            random_seed=11,
            publish_state_refs=True,
        ),
        run_name="provider_state_transition_test",
    )

    trainer.fit(max_steps=1)
    feedback = trainer.last_evaluated_feedback[0]
    state_ref = feedback.info["evaluation_state_ref"]
    gradient_ref = feedback.gradient_ref
    adapter_runtime = trainer.adapter.get_state()["provider_transition"]
    request = StateTransitionRequest(
        state_ref=state_ref,
        method_id="gradient.sgd",
        operands={"gradient": gradient_ref},
        parameters={"learning_rate": 0.05},
        step_index=0,
    )
    adapter_state = trainer.adapter.get_population()[0]
    expected_values = (
        trainer.last_evaluated_population[0].as_array()
        - (0.05 * np.asarray(feedback.gradients, dtype=float))
    )

    assert adapter_runtime["state_ref"] is None
    assert adapter_runtime["count"] == 1
    assert np.allclose(expected_values, adapter_state.as_array(), atol=1e-6)
    assert trainer.evaluation_provider.get_state()["live_state_count"] == 0

    with pytest.raises(StateVersionConflict):
        trainer.problem.gateway.transition(request, trainer.resource_context)
    with pytest.raises(StateVersionConflict):
        trainer.problem.gateway.materialize(
            StateMaterializationRequest(state_ref=state_ref.next_version()),
            trainer.resource_context,
        )


def test_torch_provider_owns_and_versions_adam_slots() -> None:
    pytest.importorskip("torch")
    data = _data()
    representation = _representation(13)
    trainer = build_gradient_trainer(
        problem=TabularNeuralRegressionProblem(data),
        representation=representation,
        method="gradient.adam",
        provider_config=TorchEvaluationProviderConfig(
            random_seed=13,
            publish_state_refs=True,
        ),
        run_name="provider_adam_slots_test",
    )
    trainer.fit(max_steps=2)
    runtime = trainer.adapter.get_state()["provider_transition"]
    assert runtime["count"] == 2
    assert runtime["slot_refs"] == {}
    assert runtime["needs_slot_seed"] is True
    assert trainer.adapter.get_state()["first_moment"] is not None
    assert trainer.adapter.get_state()["second_moment"] is not None
    assert trainer.evaluation_provider.get_state()["live_state_count"] == 0
    assert trainer._last_state_release is not None
    assert trainer._last_state_release.released_count == 2


def test_provider_transition_checkpoint_resumes_from_materialized_shadow() -> None:
    pytest.importorskip("torch")
    data = _data()

    def build(run_name: str):
        representation = _representation(19)
        return build_gradient_trainer(
            problem=TabularNeuralRegressionProblem(data),
            representation=representation,
            method="gradient.adam",
            provider_config=TorchEvaluationProviderConfig(
                random_seed=19,
                publish_state_refs=True,
            ),
            run_name=run_name,
        )

    original = build("provider_checkpoint_original")
    original.fit(max_steps=1)
    checkpoint = original.get_state()
    expected_start = original.adapter.get_population()[0].as_array().copy()

    restored = build("provider_checkpoint_restored")
    restored.set_state(checkpoint)
    restored.fit(max_steps=1)
    runtime = restored.adapter.get_state()["provider_transition"]
    resumed = restored.adapter.get_population()[0].as_array()

    assert runtime["count"] == 2
    assert runtime["state_ref"] is None
    assert runtime["slot_refs"] == {}
    assert runtime["needs_slot_seed"] is True
    assert restored.adapter.get_state()["step_index"] == 2
    assert not np.allclose(resumed, expected_start)


def test_temporal_neural_preset_uses_the_same_provider_transition_path() -> None:
    pytest.importorskip("torch")
    X = np.asarray(
        [
            [0.0, 0.1, 0.2],
            [0.1, 0.2, 0.3],
            [0.2, 0.3, 0.4],
            [0.3, 0.4, 0.5],
        ],
        dtype=float,
    )
    y = np.asarray([[0.3], [0.4], [0.5], [0.6]], dtype=float)
    trainer = build_temporal_lstm_forecast_trainer(
        NumericDataView(X_train=X, y_train=y),
        input_dim=1,
        sequence_length=3,
        hidden_dim=4,
        output_dim=1,
        learning_rate=1e-2,
        random_seed=23,
    )

    result = trainer.fit(max_steps=1)

    assert result.report["adapter"]["name"] == "gradient_optimizer"
    assert result.report["adapter"]["state"]["provider_transition"]["count"] == 1
    assert result.report["problem"]["route"] == "temporal"
    assert result.best_feedback is not None
    assert "train.rmse" in result.best_feedback.metrics


@pytest.mark.parametrize(
    ("builder", "targets", "metric"),
    (
        (
            build_tabular_tabnet_classification_trainer,
            np.asarray([0, 1, 0, 1], dtype=int),
            "train.accuracy",
        ),
        (
            build_tabular_tabnet_regression_trainer,
            np.asarray([0.0, 1.0, 0.5, 1.5], dtype=float),
            "train.rmse",
        ),
    ),
)
def test_tabnet_presets_use_problem_owned_loss_and_unified_transition(
    builder,
    targets: np.ndarray,
    metric: str,
) -> None:
    pytest.importorskip("torch")
    X = np.asarray(
        [
            [0.0, 0.1],
            [0.8, 0.9],
            [0.2, 0.3],
            [1.0, 1.1],
        ],
        dtype=float,
    )
    trainer = builder(
        NumericDataView(X_train=X, y_train=targets),
        hidden_dim=4,
        n_steps=2,
        learning_rate=1e-2,
        random_seed=29,
    )

    result = trainer.fit(max_steps=1)

    assert result.report["adapter"]["name"] == "gradient_optimizer"
    assert result.report["adapter"]["state"]["provider_transition"]["count"] == 1
    assert result.report["problem"]["route"] == "tabular"
    assert result.best_feedback is not None
    assert metric in result.best_feedback.metrics
