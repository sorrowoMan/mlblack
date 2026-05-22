from __future__ import annotations

from typing import Any, Mapping, Sequence

from mlblack.adapters import (
    EstimatorSpecSearchAdapter,
    EstimatorSpecSearchConfig,
    NeuralGraphBackpropAdapter,
    NeuralGraphBackpropConfig,
    TorchBackpropAdapter,
    TorchBackpropConfig,
)
from mlblack.core import ComputeBackendSpec, Trainer
from mlblack.pipeline.data import GraphDataView, ImageContrastivePairDataView, ImageDataView, NumericDataView
from mlblack.problems import (
    SupervisedEstimatorFitRegressionProblem,
    SupervisedRegressionProblem,
    TinyCNNImageClassificationProblem,
    TinyCNNImageContrastiveProblem,
    TinyGNNGraphClassificationProblem,
    TinyTransformerClassificationProblem,
    TinyTransformerDPOPreferenceProblem,
    TinyTransformerLanguageModelProblem,
)
from mlblack.representations import (
    NeuralBackboneSpec,
    NeuralBatchingSpec,
    NeuralOptimizationSpec,
    NeuralEngineSpec,
    NeuralGraphRepresentation,
    NumpyMLPPointConfig,
    NumpyMLPPointRepresentation,
    build_neural_estimator_representation,
    make_sklearn_mlp_factory,
)


def build_numpy_mlp_torch_backprop_trainer(
    data: NumericDataView,
    *,
    hidden_layers: Sequence[int] = (64, 32),
    activation: str = "relu",
    dropout: float = 0.0,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.0,
    batch_size: int | None = None,
    shuffle: bool = True,
    optimizer: str = "adamw",
    device: str = "cpu",
    device_policy: str = "fallback_cpu",
    checkpoint_enabled: bool = True,
    resume_enabled: bool = True,
    run_name: str = "numpy_mlp_torch_backprop",
) -> Trainer:
    cfg = NumpyMLPPointConfig(
        input_dim=int(data.X_train.shape[1]),
        hidden_layers=tuple(int(v) for v in hidden_layers),
        activation=activation,
        dropout=float(dropout),
        optimizer=optimizer,
        learning_rate=float(learning_rate),
        weight_decay=float(weight_decay),
        batch_size=batch_size,
        device=device,
        checkpoint_enabled=checkpoint_enabled,
        resume_enabled=resume_enabled,
    )
    representation = NumpyMLPPointRepresentation.from_data(
        data.X_train,
        config=cfg,
    )
    problem = SupervisedRegressionProblem(data)
    adapter = TorchBackpropAdapter(
        TorchBackpropConfig(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            batch_size=batch_size,
            shuffle=shuffle,
            optimizer=optimizer,
            device=device,
            device_policy=device_policy,
        )
    )
    return Trainer(
        problem=problem,
        representation=representation,
        adapter=adapter,
        run_name=run_name,
        compute_backend=ComputeBackendSpec(name="torch", device=device, device_policy=device_policy),
    )


def build_sklearn_mlp_estimator_search_trainer(
    data: NumericDataView,
    *,
    params: Mapping[str, Any] | None = None,
    tunable_params: tuple[str, ...] = ("alpha", "learning_rate_init"),
    bounds: Mapping[str, tuple[float, float]] | None = None,
    hidden_layers: Sequence[int] = (64, 32),
    activation: str = "relu",
    early_stopping: bool = True,
    population_size: int = 6,
    mutation_scale: float = 0.1,
    run_name: str = "sklearn_mlp_estimator_search",
) -> Trainer:
    mechanisms = {
        "backbone": NeuralBackboneSpec(hidden_layers=tuple(int(v) for v in hidden_layers), activation=activation).as_dict(),
        "optimization": NeuralOptimizationSpec(optimizer="adam", learning_rate=1e-3).as_dict(),
        "batching": NeuralBatchingSpec(batch_size=None).as_dict(),
        "engine": NeuralEngineSpec(engine="sklearn", device="cpu", checkpoint_enabled=False).as_dict(),
    }
    representation = build_neural_estimator_representation(
        route="sklearn_mlp",
        params=params
        or {
            "hidden_layer_sizes": tuple(int(v) for v in hidden_layers),
            "activation": activation,
            "solver": "adam",
            "max_iter": 200,
            "random_state": 42,
            "early_stopping": bool(early_stopping),
        },
        tunable_params=tunable_params,
        bounds=bounds or {"alpha": (1e-6, 1e-2), "learning_rate_init": (1e-4, 1e-2)},
        factory=make_sklearn_mlp_factory(),
        mechanisms=mechanisms,
    )
    problem = SupervisedEstimatorFitRegressionProblem(data)
    adapter = EstimatorSpecSearchAdapter(
        EstimatorSpecSearchConfig(population_size=population_size, mutation_scale=mutation_scale),
    )
    return Trainer(problem=problem, representation=representation, adapter=adapter, run_name=run_name)


def build_tiny_transformer_classification_trainer(
    data: NumericDataView,
    *,
    vocab_size: int,
    max_length: int | None = None,
    hidden_dim: int = 16,
    num_layers: int = 1,
    num_heads: int = 4,
    ffn_expansion_ratio: float = 2.0,
    attention_kind: str = "causal_self_attention",
    ffn_kind: str = "mlp",
    activation: str = "gelu",
    norm: str = "layer_norm",
    norm_position: str = "pre",
    position_encoding: str = "learned",
    lora: Mapping[str, Any] | None = None,
    qlora: Mapping[str, Any] | None = None,
    num_classes: int = 2,
    compute_backend: str = "torch",
    optimizer: str = "adamw",
    learning_rate: float = 1e-2,
    weight_decay: float = 0.0,
    max_grad_norm: float | None = 10.0,
    device: str = "cpu",
    device_policy: str = "fallback_cpu",
    random_seed: int = 42,
    run_name: str = "tiny_transformer_classification",
) -> Trainer:
    """Build a tiny Transformer classifier smoke trainer.

    This preset uses the backend-dispatched neural graph backprop adapter.
    Problem.evaluate remains a no-backward evaluation path; the adapter owns
    backend loss backward, optimizer step, and parameter-state export.
    """

    seq_len = int(max_length or data.X_train.shape[1])
    representation = NeuralGraphRepresentation.tiny_transformer(
        vocab_size=int(vocab_size),
        max_length=seq_len,
        hidden_dim=int(hidden_dim),
        num_layers=int(num_layers),
        num_heads=int(num_heads),
        ffn_expansion_ratio=float(ffn_expansion_ratio),
        attention_kind=str(attention_kind),
        ffn_kind=str(ffn_kind),
        activation=str(activation),
        norm=str(norm),
        norm_position=str(norm_position),
        position_encoding=str(position_encoding),
        lora=lora,
        qlora=qlora,
        heads=({"kind": "classification", "name": "classification", "params": {"num_classes": int(num_classes)}},),
        random_seed=int(random_seed),
        representation_name="tiny_transformer_classification",
    )
    problem = TinyTransformerClassificationProblem(data, head_name="classification")
    adapter = _build_neural_graph_adapter(
        optimizer=optimizer,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        max_grad_norm=max_grad_norm,
        random_seed=random_seed,
    )
    return Trainer(
        problem=problem,
        representation=representation,
        adapter=adapter,
        run_name=run_name,
        compute_backend=_compute_backend_spec(compute_backend, device, device_policy),
    )


def build_tiny_transformer_lm_trainer(
    data: NumericDataView,
    *,
    vocab_size: int,
    max_length: int | None = None,
    hidden_dim: int = 16,
    num_layers: int = 1,
    num_heads: int = 4,
    ffn_expansion_ratio: float = 2.0,
    attention_kind: str = "causal_self_attention",
    ffn_kind: str = "mlp",
    activation: str = "gelu",
    norm: str = "layer_norm",
    norm_position: str = "pre",
    position_encoding: str = "learned",
    lora: Mapping[str, Any] | None = None,
    qlora: Mapping[str, Any] | None = None,
    compute_backend: str = "torch",
    optimizer: str = "adamw",
    learning_rate: float = 1e-2,
    weight_decay: float = 0.0,
    max_grad_norm: float | None = 10.0,
    device: str = "cpu",
    device_policy: str = "fallback_cpu",
    random_seed: int = 42,
    run_name: str = "tiny_transformer_lm",
) -> Trainer:
    """Build a tiny Transformer language-modeling smoke trainer."""

    seq_len = int(max_length or data.X_train.shape[1])
    representation = NeuralGraphRepresentation.tiny_transformer(
        vocab_size=int(vocab_size),
        max_length=max(1, seq_len - 1),
        hidden_dim=int(hidden_dim),
        num_layers=int(num_layers),
        num_heads=int(num_heads),
        ffn_expansion_ratio=float(ffn_expansion_ratio),
        attention_kind=str(attention_kind),
        ffn_kind=str(ffn_kind),
        activation=str(activation),
        norm=str(norm),
        norm_position=str(norm_position),
        position_encoding=str(position_encoding),
        lora=lora,
        qlora=qlora,
        heads=({"kind": "language_modeling", "name": "lm", "params": {"vocab_size": int(vocab_size)}},),
        random_seed=int(random_seed),
        representation_name="tiny_transformer_lm",
    )
    problem = TinyTransformerLanguageModelProblem(data, head_name="lm")
    adapter = _build_neural_graph_adapter(
        optimizer=optimizer,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        max_grad_norm=max_grad_norm,
        random_seed=random_seed,
    )
    return Trainer(
        problem=problem,
        representation=representation,
        adapter=adapter,
        run_name=run_name,
        compute_backend=_compute_backend_spec(compute_backend, device, device_policy),
    )


def build_tiny_transformer_dpo_preference_trainer(
    data: Any,
    *,
    vocab_size: int,
    max_length: int | None = None,
    hidden_dim: int = 16,
    num_layers: int = 1,
    num_heads: int = 4,
    ffn_expansion_ratio: float = 2.0,
    attention_kind: str = "causal_self_attention",
    ffn_kind: str = "mlp",
    activation: str = "gelu",
    norm: str = "layer_norm",
    norm_position: str = "pre",
    position_encoding: str = "learned",
    lora: Mapping[str, Any] | None = None,
    qlora: Mapping[str, Any] | None = None,
    beta: float = 0.1,
    compute_backend: str = "torch",
    optimizer: str = "adamw",
    learning_rate: float = 1e-2,
    weight_decay: float = 0.0,
    max_grad_norm: float | None = 10.0,
    device: str = "cpu",
    device_policy: str = "fallback_cpu",
    random_seed: int = 42,
    run_name: str = "tiny_transformer_dpo_preference",
) -> Trainer:
    """Build a tiny Transformer DPO/preference trainer."""

    seq_len = int(max_length or data.sequence_length)
    representation = NeuralGraphRepresentation.tiny_transformer(
        vocab_size=int(vocab_size),
        max_length=max(1, seq_len - 1),
        hidden_dim=int(hidden_dim),
        num_layers=int(num_layers),
        num_heads=int(num_heads),
        ffn_expansion_ratio=float(ffn_expansion_ratio),
        attention_kind=str(attention_kind),
        ffn_kind=str(ffn_kind),
        activation=str(activation),
        norm=str(norm),
        norm_position=str(norm_position),
        position_encoding=str(position_encoding),
        lora=lora,
        qlora=qlora,
        heads=({"kind": "language_modeling", "name": "lm", "params": {"vocab_size": int(vocab_size)}},),
        random_seed=int(random_seed),
        representation_name="tiny_transformer_dpo_preference",
    )
    problem = TinyTransformerDPOPreferenceProblem(data, head_name="lm", beta=float(beta))
    adapter = _build_neural_graph_adapter(
        optimizer=optimizer,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        max_grad_norm=max_grad_norm,
        random_seed=random_seed,
    )
    return Trainer(
        problem=problem,
        representation=representation,
        adapter=adapter,
        run_name=run_name,
        compute_backend=_compute_backend_spec(compute_backend, device, device_policy),
    )


def build_tiny_cnn_image_classification_trainer(
    data: ImageDataView,
    *,
    conv_channels: Sequence[int] = (8, 16),
    kernel_size: int = 3,
    activation: str = "relu",
    dropout: float = 0.0,
    num_classes: int = 2,
    compute_backend: str = "torch",
    optimizer: str = "adamw",
    learning_rate: float = 1e-2,
    weight_decay: float = 0.0,
    max_grad_norm: float | None = 10.0,
    device: str = "cpu",
    device_policy: str = "fallback_cpu",
    random_seed: int = 42,
    run_name: str = "tiny_cnn_image_classification",
) -> Trainer:
    representation = NeuralGraphRepresentation.tiny_cnn(
        channels=int(data.channels),
        height=int(data.height),
        width=int(data.width),
        conv_channels=tuple(int(v) for v in conv_channels),
        kernel_size=int(kernel_size),
        activation=str(activation),
        dropout=float(dropout),
        heads=({"kind": "classification", "name": "classification", "params": {"num_classes": int(num_classes)}},),
        random_seed=int(random_seed),
        representation_name="tiny_cnn_image_classification",
    )
    problem = TinyCNNImageClassificationProblem(data, head_name="classification")
    adapter = _build_neural_graph_adapter(
        optimizer=optimizer,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        max_grad_norm=max_grad_norm,
        random_seed=random_seed,
    )
    return Trainer(
        problem=problem,
        representation=representation,
        adapter=adapter,
        run_name=run_name,
        compute_backend=_compute_backend_spec(compute_backend, device, device_policy),
    )


def build_tiny_gnn_graph_classification_trainer(
    data: GraphDataView,
    *,
    hidden_dim: int = 16,
    num_layers: int = 2,
    activation: str = "relu",
    dropout: float = 0.0,
    pooling: str = "mean",
    num_classes: int = 2,
    compute_backend: str = "torch",
    optimizer: str = "adamw",
    learning_rate: float = 1e-2,
    weight_decay: float = 0.0,
    max_grad_norm: float | None = 10.0,
    device: str = "cpu",
    device_policy: str = "fallback_cpu",
    random_seed: int = 42,
    run_name: str = "tiny_gnn_graph_classification",
) -> Trainer:
    representation = NeuralGraphRepresentation.tiny_gnn(
        node_feature_dim=int(data.node_feature_dim),
        num_nodes=int(data.num_nodes),
        hidden_dim=int(hidden_dim),
        num_layers=int(num_layers),
        activation=str(activation),
        dropout=float(dropout),
        pooling=str(pooling),
        heads=({"kind": "classification", "name": "classification", "params": {"num_classes": int(num_classes)}},),
        random_seed=int(random_seed),
        representation_name="tiny_gnn_graph_classification",
    )
    problem = TinyGNNGraphClassificationProblem(data, head_name="classification")
    adapter = _build_neural_graph_adapter(
        optimizer=optimizer,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        max_grad_norm=max_grad_norm,
        random_seed=random_seed,
    )
    return Trainer(
        problem=problem,
        representation=representation,
        adapter=adapter,
        run_name=run_name,
        compute_backend=_compute_backend_spec(compute_backend, device, device_policy),
    )


def build_tiny_cnn_image_contrastive_trainer(
    data: ImageContrastivePairDataView,
    *,
    conv_channels: Sequence[int] = (8, 16),
    embedding_dim: int = 8,
    margin: float = 0.5,
    kernel_size: int = 3,
    activation: str = "relu",
    dropout: float = 0.0,
    compute_backend: str = "torch",
    optimizer: str = "adamw",
    learning_rate: float = 1e-2,
    weight_decay: float = 0.0,
    max_grad_norm: float | None = 10.0,
    device: str = "cpu",
    device_policy: str = "fallback_cpu",
    random_seed: int = 42,
    run_name: str = "tiny_cnn_image_contrastive",
) -> Trainer:
    representation = NeuralGraphRepresentation.tiny_cnn(
        channels=int(data.channels),
        height=int(data.height),
        width=int(data.width),
        conv_channels=tuple(int(v) for v in conv_channels),
        kernel_size=int(kernel_size),
        activation=str(activation),
        dropout=float(dropout),
        heads=({"kind": "retrieval", "name": "retrieval", "params": {"output_dim": int(embedding_dim)}},),
        random_seed=int(random_seed),
        representation_name="tiny_cnn_image_contrastive",
    )
    problem = TinyCNNImageContrastiveProblem(data, head_name="retrieval", margin=float(margin))
    adapter = _build_neural_graph_adapter(
        optimizer=optimizer,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        max_grad_norm=max_grad_norm,
        random_seed=random_seed,
    )
    return Trainer(
        problem=problem,
        representation=representation,
        adapter=adapter,
        run_name=run_name,
        compute_backend=_compute_backend_spec(compute_backend, device, device_policy),
    )


def _build_neural_graph_adapter(
    *,
    optimizer: str,
    learning_rate: float,
    weight_decay: float,
    max_grad_norm: float | None,
    random_seed: int,
) -> Any:
    return NeuralGraphBackpropAdapter(
        NeuralGraphBackpropConfig(
            optimizer=str(optimizer),
            learning_rate=float(learning_rate),
            weight_decay=float(weight_decay),
            max_grad_norm=max_grad_norm,
            random_seed=int(random_seed),
        )
    )


def _compute_backend_spec(compute_backend: Any, device: str, device_policy: str) -> ComputeBackendSpec:
    if isinstance(compute_backend, Mapping):
        payload = dict(compute_backend)
        payload.setdefault("device", device)
        payload.setdefault("device_policy", device_policy)
        return ComputeBackendSpec.from_value(payload)
    return ComputeBackendSpec(name=str(compute_backend), device=str(device), device_policy=str(device_policy))



