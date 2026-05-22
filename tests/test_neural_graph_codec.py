from __future__ import annotations

import numpy as np
import pytest

from mlblack.core import ArtifactBuilder, ComputeBackendSession, ComputeBackendSpec, render_artifact_html
from mlblack.assembly import build_trainer
from mlblack.adapters import FunctionalBackpropAdapter, FunctionalBackpropConfig
from mlblack.pipeline.data import NumericDataView, PreferencePairDataView
from mlblack.pipeline import VocabularyTokenizer
from mlblack.integrations import (
    PretrainedCheckpointMapper,
    PretrainedCheckpointMappingConfig,
    PretrainedModelBridge,
    PretrainedModelBridgeConfig,
    PretrainedTokenizerBridge,
    PretrainedTokenizerBridgeConfig,
)
from mlblack.integrations.nsgablack_neural import TransformerSpecSearchConfig, TransformerSpecSearchProblem
from mlblack.presets import (
    build_tiny_transformer_classification_trainer,
    build_tiny_transformer_dpo_preference_trainer,
    build_tiny_transformer_lm_trainer,
)
from mlblack.models import NumpyMLPPointModel
from mlblack.backends import get_backend
from mlblack.representations.codecs import (
    NeuralBackboneSpec,
    NeuralGraphCodec,
    NeuralGraphSpec,
    NumpyMLPCodec,
    NumpyMLPCodecConfig,
)
from mlblack.representations import NeuralGraphRepresentation
from mlblack.problems import SupervisedRegressionProblem
from mlblack.core import Trainer


def _backend_context() -> dict[str, object]:
    return {"backend.session": ComputeBackendSession(ComputeBackendSpec(name="torch", device="cpu"))}


def _numpy_backend_context() -> dict[str, object]:
    return {"backend.session": ComputeBackendSession(ComputeBackendSpec(name="numpy", device="cpu"))}


def _jax_backend_context() -> dict[str, object]:
    return {"backend.session": ComputeBackendSession(ComputeBackendSpec(name="jax", device="cpu"))}


def test_neural_graph_codec_decodes_mlp_model() -> None:
    spec = NeuralGraphSpec.mlp(input_dim=2, hidden_layers=(3,), output_dim=1, activation="tanh")
    codec = NeuralGraphCodec(spec, init_scale=0.01, random_seed=7)

    layout = codec.parameter_layout()
    assert layout.total_size == 13
    assert layout.names == (
        "mlp.layers.0.weight",
        "mlp.layers.0.bias",
        "mlp.layers.1.weight",
        "mlp.layers.1.bias",
    )

    values = codec.init_values()
    assert values.shape == (layout.total_size,)

    model = codec.decode(values)
    assert isinstance(model, NumpyMLPPointModel)
    pred = model.predict(np.ones((4, 2), dtype=float))
    assert pred.shape == (4,)
    assert codec.describe()["route"] == "mlp"


def test_neural_graph_codec_can_lower_mlp_with_numpy_backend() -> None:
    spec = NeuralGraphSpec.mlp(input_dim=2, hidden_layers=(3,), output_dim=1, activation="tanh")
    codec = NeuralGraphCodec(spec, init_scale=0.01, random_seed=7)
    ctx = _numpy_backend_context()

    layout = codec.parameter_layout(ctx)
    assert layout.total_size == 13
    assert layout.metadata["backend"] == "numpy"

    values = codec.init_values(ctx)
    model = codec.decode(values, ctx)
    assert isinstance(model, NumpyMLPPointModel)
    assert model.metadata["backend"] == "numpy"
    assert model.predict(np.ones((2, 2), dtype=float)).shape == (2,)


def test_neural_graph_codec_can_lower_mlp_with_jax_backend() -> None:
    pytest.importorskip("jax")
    spec = NeuralGraphSpec.mlp(input_dim=2, hidden_layers=(3,), output_dim=1, activation="tanh")
    codec = NeuralGraphCodec(spec, init_scale=0.01, random_seed=7)
    ctx = _jax_backend_context()

    layout = codec.parameter_layout(ctx)
    assert layout.total_size == 13
    assert layout.metadata["backend"] == "jax"

    values = codec.init_values(ctx)
    model = codec.decode(values, ctx)
    assert model.metadata["backend"] == "jax"
    pred = model.predict(np.ones((2, 2), dtype=float))
    grad = get_backend("jax").autograd.mse_parameter_gradient(
        model,
        np.ones((2, 2), dtype=float),
        np.ones(2, dtype=float),
    )
    assert pred.shape == (2,)
    assert grad.shape == (layout.total_size,)
    assert not hasattr(model, "parameter_gradient")


def test_jax_backend_mlp_regression_trainer_smoke() -> None:
    pytest.importorskip("jax")
    X_train = np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=float)
    y_train = X_train[:, 0] - X_train[:, 1]
    data = NumericDataView(X_train=X_train, y_train=y_train)
    representation = NeuralGraphRepresentation(
        NeuralGraphSpec.mlp(input_dim=2, hidden_layers=(4,), output_dim=1, activation="tanh")
    )
    trainer = Trainer(
        problem=SupervisedRegressionProblem(data),
        representation=representation,
        adapter=FunctionalBackpropAdapter(FunctionalBackpropConfig(learning_rate=0.05, max_grad_norm=10.0)),
        run_name="jax_mlp_regression_smoke",
        compute_backend=ComputeBackendSpec(name="jax", device="cpu"),
    )

    result = trainer.fit(max_steps=2)
    assert len(result.history) == 2
    assert result.best_feedback is not None
    assert result.report["adapter"]["name"] == "functional_backprop"
    assert result.report["compute_backend"]["resolved_name"] == "jax"
    assert result.report["representation"]["parameter_layout"]["metadata"]["backend"] == "jax"


def test_functional_backprop_fails_fast_on_numpy_backend() -> None:
    X_train = np.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=float)
    y_train = X_train[:, 0]
    data = NumericDataView(X_train=X_train, y_train=y_train)
    trainer = Trainer(
        problem=SupervisedRegressionProblem(data),
        representation=NeuralGraphRepresentation(
            NeuralGraphSpec.mlp(input_dim=2, hidden_layers=(3,), output_dim=1, activation="tanh")
        ),
        adapter=FunctionalBackpropAdapter(FunctionalBackpropConfig(learning_rate=0.05)),
        run_name="numpy_functional_backprop_missing_capability",
        compute_backend=ComputeBackendSpec(name="numpy", device="cpu"),
    )

    with pytest.raises(ValueError, match="autograd.functional.grad"):
        trainer.fit(max_steps=1)


def test_tensorflow_backend_mlp_regression_trainer_smoke_if_installed() -> None:
    pytest.importorskip("tensorflow")
    X_train = np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=float)
    y_train = X_train[:, 0] - X_train[:, 1]
    data = NumericDataView(X_train=X_train, y_train=y_train)
    trainer = Trainer(
        problem=SupervisedRegressionProblem(data),
        representation=NeuralGraphRepresentation(
            NeuralGraphSpec.mlp(input_dim=2, hidden_layers=(4,), output_dim=1, activation="tanh")
        ),
        adapter=FunctionalBackpropAdapter(FunctionalBackpropConfig(learning_rate=0.05, max_grad_norm=10.0)),
        run_name="tensorflow_mlp_regression_smoke",
        compute_backend=ComputeBackendSpec(name="tensorflow", device="cpu"),
    )

    result = trainer.fit(max_steps=2)
    assert len(result.history) == 2
    assert result.report["adapter"]["name"] == "functional_backprop"
    assert result.report["compute_backend"]["resolved_name"] == "tensorflow"


def test_legacy_numpy_mlp_codec_import_still_works() -> None:
    codec = NumpyMLPCodec(
        NumpyMLPCodecConfig(
            input_dim=2,
            backbone=NeuralBackboneSpec(hidden_layers=(3,)),
            output_dim=1,
        )
    )
    values = codec.init_values()
    model = codec.decode(values)
    assert isinstance(model, NumpyMLPPointModel)
    assert model.predict(np.ones((2, 2), dtype=float)).shape == (2,)


def test_neural_graph_codec_decodes_tiny_transformer_classification() -> None:
    import torch

    spec = NeuralGraphSpec.tiny_transformer(
        vocab_size=32,
        max_length=8,
        hidden_dim=16,
        num_layers=2,
        num_heads=4,
        ffn_expansion_ratio=2.0,
        heads=({"kind": "classification", "name": "cls", "params": {"num_classes": 3}},),
    )
    codec = NeuralGraphCodec(spec, random_seed=11)
    ctx = _backend_context()
    assert codec.describe(ctx)["route"] == "tiny_transformer"
    assert codec.parameter_layout(ctx).total_size > 0
    values = codec.init_values(ctx)
    model = codec.decode(values, ctx)

    batch = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 0]], dtype=torch.long)
    mask = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]], dtype=torch.long)
    output = model(batch, attention_mask=mask, return_audit=True)
    assert output["logits"].shape == (2, 3)
    assert output["hidden_states"].shape == (2, 4, 16)
    assert len(output["audit"]["attention_maps"]) == 2


def test_neural_graph_codec_decodes_tiny_transformer_lm() -> None:
    import torch

    spec = NeuralGraphSpec.tiny_transformer(
        vocab_size=40,
        max_length=6,
        hidden_dim=12,
        num_layers=1,
        num_heads=3,
        ffn_expansion_ratio=2.0,
        heads=({"kind": "language_modeling", "name": "lm"},),
    )
    codec = NeuralGraphCodec(spec, random_seed=13)
    ctx = _backend_context()
    model = codec.decode(codec.init_values(ctx), ctx)
    batch = torch.tensor([[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]], dtype=torch.long)
    output = model(batch)
    assert output["logits"].shape == (2, 5, 40)
    assert output["head_outputs"]["lm"].shape == (2, 5, 40)
    generated, cache = model.generate(batch[:, :2], max_new_tokens=2, return_cache=True)
    assert generated.shape == (2, 4)
    assert cache["schema"] == "mlblack.tiny_transformer.kv_cache.v1"
    assert cache["num_layers"] == 1


def test_tiny_transformer_advanced_block_and_heads_smoke() -> None:
    import torch

    spec = NeuralGraphSpec.tiny_transformer(
        vocab_size=24,
        max_length=5,
        hidden_dim=8,
        num_layers=1,
        num_heads=2,
        ffn_expansion_ratio=2.0,
        ffn_kind="swiglu",
        norm="rms_norm",
        position_encoding="rope",
        lora={"enabled": True, "rank": 2, "alpha": 4.0, "targets": ("attention.q", "attention.v"), "freeze_base": True},
        heads=(
            {"kind": "embedding", "name": "embed", "params": {"output_dim": 6}},
            {"kind": "ranking", "name": "rank"},
            {"kind": "preference", "name": "pref"},
        ),
    )
    codec = NeuralGraphCodec(spec, random_seed=17)
    ctx = _backend_context()
    model = codec.decode(codec.init_values(ctx), ctx)
    batch = torch.tensor([[1, 2, 3, 4, 0], [5, 4, 3, 2, 1]], dtype=torch.long)
    output = model(batch, return_audit=True)
    names = tuple(name for name, _param in model.named_parameters())
    assert any("lora_a" in name for name in names)
    assert output["embeddings"].shape == (2, 6)
    assert output["ranking_scores"].shape == (2, 1)
    assert output["preference_scores"].shape == (2, 1)
    assert len(output["audit"]["attention_maps"]) == 1
    assert len(output["audit"]["ffn_activations"]) == 1


def test_tiny_transformer_qlora_surface_smoke() -> None:
    spec = NeuralGraphSpec.tiny_transformer(
        vocab_size=18,
        max_length=5,
        hidden_dim=8,
        num_layers=1,
        num_heads=2,
        qlora={"rank": 2, "bits": 4, "targets": ("attention.q", "attention.v")},
        heads=({"kind": "language_modeling", "name": "lm"},),
    )
    codec = NeuralGraphCodec(spec, random_seed=19)
    ctx = _backend_context()
    model = codec.decode(codec.init_values(ctx), ctx)
    qlora_layers = [module for module in model.modules() if hasattr(module, "quantize_base")]
    assert qlora_layers
    assert all(bool(module.quantize_base) for module in qlora_layers)
    assert any("lora_a" in name for name, _param in model.named_parameters())


def test_tiny_transformer_classification_trainer_smoke() -> None:
    X_train = np.asarray(
        [
            [1, 2, 3, 0],
            [2, 2, 1, 0],
            [7, 1, 1, 0],
            [8, 2, 1, 0],
            [1, 3, 5, 0],
            [9, 3, 1, 0],
        ],
        dtype=float,
    )
    y_train = (X_train[:, 0] >= 5).astype(float)
    X_valid = np.asarray([[1, 1, 2, 0], [8, 1, 1, 0]], dtype=float)
    y_valid = (X_valid[:, 0] >= 5).astype(float)
    data = NumericDataView(X_train=X_train, y_train=y_train, X_valid=X_valid, y_valid=y_valid)

    trainer = build_tiny_transformer_classification_trainer(
        data,
        vocab_size=16,
        max_length=4,
        hidden_dim=8,
        num_layers=1,
        num_heads=2,
        ffn_expansion_ratio=2.0,
        num_classes=2,
        learning_rate=1e-2,
        random_seed=5,
        run_name="tiny_transformer_classification_smoke",
    )
    result = trainer.fit(max_steps=2)
    assert len(result.history) == 2
    assert result.best_feedback is not None
    assert result.best_feedback.gradients is None
    assert result.report["adapter"]["name"] == "neural_graph_backprop"
    assert result.report["adapter"]["state"]["optimizer_state"]
    assert result.report["problem"]["route"] == "tiny_transformer"

    bundle = ArtifactBuilder().build(trainer, result)
    assert bundle.model_artifact is not None
    artifact = bundle.model_artifact.describe()
    assert artifact["artifact_type"] == "neural_graph"
    assert artifact["graph_spec_digest"]
    assert artifact["parameter_layout_digest"]
    assert artifact["audit_artifact"]["attention"]["num_layers"] == 1
    html = render_artifact_html(bundle, title="tiny artifact")
    assert "<html" in html
    assert "neural_graph" in html


def test_tiny_transformer_lm_trainer_smoke() -> None:
    X_train = np.asarray(
        [
            [1, 2, 3, 4, 5],
            [1, 2, 2, 3, 4],
            [5, 4, 3, 2, 1],
            [6, 6, 4, 4, 2],
        ],
        dtype=float,
    )
    X_valid = np.asarray([[1, 2, 3, 4, 5], [6, 4, 3, 2, 1]], dtype=float)
    data = NumericDataView(
        X_train=X_train,
        y_train=np.zeros(X_train.shape[0], dtype=float),
        X_valid=X_valid,
        y_valid=np.zeros(X_valid.shape[0], dtype=float),
    )

    trainer = build_tiny_transformer_lm_trainer(
        data,
        vocab_size=12,
        max_length=5,
        hidden_dim=8,
        num_layers=1,
        num_heads=2,
        ffn_kind="swiglu",
        norm="rms_norm",
        position_encoding="rope",
        learning_rate=1e-2,
        random_seed=9,
        run_name="tiny_transformer_lm_smoke",
    )
    result = trainer.fit(max_steps=2)
    assert len(result.history) == 2
    assert result.best_feedback is not None
    assert "train.perplexity" in result.best_feedback.metrics
    assert result.report["adapter"]["name"] == "neural_graph_backprop"
    assert result.report["adapter"]["state"]["optimizer_state"]


def test_tiny_transformer_lm_assembly_preset_smoke() -> None:
    X_train = np.asarray([[1, 2, 3, 4], [4, 3, 2, 1], [2, 2, 3, 3]], dtype=float)
    data = NumericDataView(X_train=X_train, y_train=np.zeros(X_train.shape[0], dtype=float))
    trainer = build_trainer(
        {
            "preset": "tiny_transformer_lm",
            "params": {
                "vocab_size": 8,
                "max_length": 4,
                "hidden_dim": 8,
                "num_layers": 1,
                "num_heads": 2,
                "learning_rate": 1e-2,
                "random_seed": 12,
            },
        },
        data,
    )
    result = trainer.fit(max_steps=1)
    assert result.best_feedback is not None
    assert result.report["problem"]["head"] == "language_modeling"


def test_tiny_transformer_dpo_preference_trainer_smoke() -> None:
    data = PreferencePairDataView(
        chosen_train=np.asarray([[1, 2, 3, 4], [1, 3, 4, 5], [2, 3, 5, 6]], dtype=float),
        rejected_train=np.asarray([[1, 2, 2, 2], [1, 3, 3, 3], [2, 3, 3, 3]], dtype=float),
    )
    trainer = build_tiny_transformer_dpo_preference_trainer(
        data,
        vocab_size=10,
        max_length=4,
        hidden_dim=8,
        num_layers=1,
        num_heads=2,
        qlora={"rank": 2, "bits": 4, "targets": ("attention.q", "attention.v")},
        learning_rate=1e-2,
        random_seed=21,
    )
    result = trainer.fit(max_steps=1)
    assert result.best_feedback is not None
    assert "train.preference_accuracy" in result.best_feedback.metrics
    assert result.report["problem"]["head"] == "preference_dpo"


def test_vocabulary_tokenizer_and_pretrained_bridge_describe_smoke() -> None:
    tokenizer = VocabularyTokenizer()
    ids = tokenizer.fit_transform(["Cat sat.", "Dog sat."], max_length=5)
    assert ids.shape == (2, 5)
    assert tokenizer.describe()["vocab_size"] >= 5

    pretrained_tokenizer = PretrainedTokenizerBridge(PretrainedTokenizerBridgeConfig(model_name_or_path="local-or-hf-id"))
    pretrained_model = PretrainedModelBridge(PretrainedModelBridgeConfig(model_name_or_path="local-or-hf-id", task="causal_lm"))
    assert pretrained_tokenizer.describe()["loaded"] is False
    assert pretrained_model.describe()["task"] == "causal_lm"


def test_pretrained_checkpoint_mapper_smoke() -> None:
    import torch

    spec = NeuralGraphSpec.tiny_transformer(
        vocab_size=12,
        max_length=4,
        hidden_dim=8,
        num_layers=1,
        num_heads=2,
        heads=({"kind": "language_modeling", "name": "lm"},),
    )
    codec = NeuralGraphCodec(spec, random_seed=23)
    ctx = _backend_context()
    model = codec.decode(codec.init_values(ctx), ctx)
    source = {f"src.{name}": torch.ones_like(param) for name, param in model.state_dict().items()}
    name_map = {name: f"src.{name}" for name in model.state_dict().keys()}
    mapper = PretrainedCheckpointMapper(PretrainedCheckpointMappingConfig(name_map=name_map))
    report = mapper.map_state_dict(source, model)
    assert report.mapped_fraction == 1.0
    flat = mapper.flat_values_from_model(model)
    assert flat.shape[0] == codec.parameter_layout(ctx).total_size


def test_pretrained_checkpoint_mapper_prefix_normalization() -> None:
    import torch

    target = torch.nn.Linear(2, 1)
    source = {f"module.{name}": torch.ones_like(param) for name, param in target.state_dict().items()}
    mapper = PretrainedCheckpointMapper(PretrainedCheckpointMappingConfig(source_prefixes=("module.",)))

    report = mapper.map_state_dict(source, target)
    assert report.mapped_fraction == 1.0
    assert all(source_name.startswith("module.") for _target_name, source_name in report.matched)
    assert report.metadata["source_prefixes"] == ("module.",)


def test_pretrained_model_bridge_thin_forward_and_generate_wrapper() -> None:
    class _FakeModel:
        def __call__(self, **inputs):
            return {"seen": tuple(sorted(inputs))}

        def generate(self, **inputs):
            return ("generated", tuple(sorted(inputs)))

    bridge = PretrainedModelBridge(PretrainedModelBridgeConfig(model_name_or_path="unused", task="causal_lm"))
    bridge._model = _FakeModel()

    assert bridge.forward(input_ids=[1, 2]) == {"seen": ("input_ids",)}
    assert bridge.generate(input_ids=[1, 2]) == ("generated", ("input_ids",))


def test_neural_graph_codec_requires_backend_session_for_non_mlp_routes() -> None:
    spec = NeuralGraphSpec.tiny_transformer(
        vocab_size=16,
        max_length=4,
        hidden_dim=8,
        num_layers=1,
        num_heads=2,
        heads=({"kind": "language_modeling", "name": "lm"},),
    )
    codec = NeuralGraphCodec(spec, random_seed=29)

    with pytest.raises(ValueError, match="backend context"):
        codec.parameter_layout()

    with pytest.raises(ValueError, match="backend context"):
        codec.init_values()


def test_nsgablack_neural_transformer_spec_outer_problem_smoke() -> None:
    X_train = np.asarray([[1, 2, 3, 0], [4, 3, 2, 0], [1, 1, 2, 0], [5, 4, 1, 0]], dtype=float)
    y_train = (X_train[:, 0] >= 4).astype(float)
    data = NumericDataView(X_train=X_train, y_train=y_train)
    problem = TransformerSpecSearchProblem(
        data,
        config=TransformerSpecSearchConfig(
            task="classification",
            vocab_size=8,
            max_length=4,
            num_classes=2,
            hidden_dim_choices=(8,),
            num_layer_choices=(1,),
            num_head_choices=(1, 2),
            inner_steps=1,
            objective_names=("loss", "complexity"),
        ),
    )
    record = problem.evaluate_detailed(np.zeros(problem.dimension, dtype=float))
    assert len(record.objectives) == 2
    assert record.graph_spec["metadata"]["route"] == "tiny_transformer"
    assert problem.evaluate(np.zeros(problem.dimension, dtype=float)).shape == (2,)
