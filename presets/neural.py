from __future__ import annotations

from typing import Any, Mapping, Sequence

from mlblack.core import ComputeBackendSpec
from mlblack.pipeline.data_views import GraphDataView, ImageContrastivePairDataView, ImageDataView, NumericDataView
from mlblack.pipeline.datasets import DatasetStreamConfig, NumericBatchSchedule
from mlblack.problems import (
    SupervisedEstimatorFitRegressionProblem,
    TabularNeuralClassificationProblem,
    TabularNeuralRegressionProblem,
    TemporalNeuralForecastingProblem,
    TemporalNeuralProbabilisticForecastingProblem,
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
    NeuralGraphRepresentation,
    NeuralGraphRepresentationConfig,
    NeuralGraphSpec,
    NeuralOptimizationSpec,
    NeuralEngineSpec,
    build_neural_estimator_representation,
    make_sklearn_mlp_factory,
)


def build_mlp_regression_trainer(
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
    run_name: str = "mlp_regression",
    resource_context: Mapping[str, Any] | None = None,
) -> Any:
    # Model structure is declared once. Optimizer, batching, backend, and
    # checkpoint policy remain separate assembly concerns.
    representation = NeuralGraphRepresentation.mlp(
        input_dim=int(data.X_train.shape[1]),
        hidden_layers=tuple(int(v) for v in hidden_layers),
        output_dim=1,
        activation=activation,
        dropout=float(dropout),
        random_seed=42,
        representation_name="mlp_regression",
    )
    problem = TabularNeuralRegressionProblem(data)
    if (
        str(device or "cpu").strip().lower() not in {"", "cpu", "auto"}
        and resource_context is None
        and str(device_policy or "fallback_cpu").strip().lower() == "strict"
    ):
        raise ValueError(
            "strict accelerator use requires an injected Project L0 resource_context"
        )
    schedule = NumericBatchSchedule(
        data,
        DatasetStreamConfig(
            batch_size=(
                int(data.X_train.shape[0])
                if batch_size is None
                else int(batch_size)
            ),
            shuffle=bool(shuffle),
            drop_last=False,
            seed=42,
        ),
    )
    from mlblack.integrations.nsgablack_gradient import build_gradient_trainer

    return build_gradient_trainer(
        problem=problem,
        representation=representation,
        method=optimizer,
        compute_backend="torch",
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        data_schedule=schedule,
        resource_context=resource_context,
        run_name=run_name,
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
) -> Any:
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
    adapter = _build_optimization_adapter(
        "search.random_gaussian",
        population_size=population_size,
        mutation_scale=mutation_scale,
        initialization="center",
        include_center_candidate=True,
    )
    return _build_learning_solver(problem=problem, representation=representation, adapter=adapter, run_name=run_name)


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
    resource_context: Mapping[str, Any] | None = None,
) -> Any:
    """Build a tiny Transformer classifier smoke trainer.

    Torch uses the unified ML Provider -> nsgablack gradient Adapter path.
    Functional backends use the same nsgablack Adapter and expose gradients at
    the ML Problem/Provider boundary; no private ML optimization Adapter exists.
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
    return _build_neural_graph_trainer(
        problem=problem,
        representation=representation,
        compute_backend=compute_backend,
        optimizer=optimizer,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        max_grad_norm=max_grad_norm,
        device=device,
        device_policy=device_policy,
        random_seed=random_seed,
        run_name=run_name,
        resource_context=resource_context,
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
    resource_context: Mapping[str, Any] | None = None,
) -> Any:
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
    return _build_neural_graph_trainer(
        problem=problem,
        representation=representation,
        compute_backend=compute_backend,
        optimizer=optimizer,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        max_grad_norm=max_grad_norm,
        device=device,
        device_policy=device_policy,
        random_seed=random_seed,
        run_name=run_name,
        resource_context=resource_context,
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
    resource_context: Mapping[str, Any] | None = None,
) -> Any:
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
    return _build_neural_graph_trainer(
        problem=problem,
        representation=representation,
        compute_backend=compute_backend,
        optimizer=optimizer,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        max_grad_norm=max_grad_norm,
        device=device,
        device_policy=device_policy,
        random_seed=random_seed,
        run_name=run_name,
        resource_context=resource_context,
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
    resource_context: Mapping[str, Any] | None = None,
) -> Any:
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
    return _build_neural_graph_trainer(
        problem=problem,
        representation=representation,
        compute_backend=compute_backend,
        optimizer=optimizer,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        max_grad_norm=max_grad_norm,
        device=device,
        device_policy=device_policy,
        random_seed=random_seed,
        run_name=run_name,
        resource_context=resource_context,
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
    resource_context: Mapping[str, Any] | None = None,
) -> Any:
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
    return _build_neural_graph_trainer(
        problem=problem,
        representation=representation,
        compute_backend=compute_backend,
        optimizer=optimizer,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        max_grad_norm=max_grad_norm,
        device=device,
        device_policy=device_policy,
        random_seed=random_seed,
        run_name=run_name,
        resource_context=resource_context,
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
    resource_context: Mapping[str, Any] | None = None,
) -> Any:
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
    return _build_neural_graph_trainer(
        problem=problem,
        representation=representation,
        compute_backend=compute_backend,
        optimizer=optimizer,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        max_grad_norm=max_grad_norm,
        device=device,
        device_policy=device_policy,
        random_seed=random_seed,
        run_name=run_name,
        resource_context=resource_context,
    )


def build_temporal_lstm_forecast_trainer(
    data: NumericDataView,
    *,
    input_dim: int = 1,
    sequence_length: int = 12,
    hidden_dim: int = 32,
    num_layers: int = 1,
    output_dim: int = 1,
    dropout: float = 0.0,
    bidirectional: bool = False,
    compute_backend: str = "torch",
    optimizer: str = "adamw",
    learning_rate: float = 1e-2,
    weight_decay: float = 0.0,
    max_grad_norm: float | None = 10.0,
    device: str = "cpu",
    device_policy: str = "fallback_cpu",
    random_seed: int = 42,
    run_name: str = "temporal_lstm_forecast",
    resource_context: Mapping[str, Any] | None = None,
) -> Any:
    representation = NeuralGraphRepresentation(
        NeuralGraphRepresentationConfig(
            graph_spec=NeuralGraphSpec.temporal_lstm(
                input_dim=int(input_dim),
                sequence_length=int(sequence_length),
                hidden_dim=int(hidden_dim),
                num_layers=int(num_layers),
                output_dim=int(output_dim),
                dropout=float(dropout),
                bidirectional=bool(bidirectional),
            ),
            random_seed=int(random_seed),
            representation_name="temporal_lstm_forecast",
        )
    )
    problem = TemporalNeuralForecastingProblem(data, head_name="forecast")
    return _build_neural_graph_trainer(
        problem=problem,
        representation=representation,
        compute_backend=compute_backend,
        optimizer=optimizer,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        max_grad_norm=max_grad_norm,
        device=device,
        device_policy=device_policy,
        random_seed=random_seed,
        run_name=run_name,
        resource_context=resource_context,
    )


def build_temporal_tcn_forecast_trainer(
    data: NumericDataView,
    *,
    input_dim: int = 1,
    sequence_length: int = 12,
    channels: Sequence[int] = (32, 32),
    kernel_size: int = 3,
    dilation_base: int = 2,
    output_dim: int = 1,
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
    run_name: str = "temporal_tcn_forecast",
    resource_context: Mapping[str, Any] | None = None,
) -> Any:
    representation = NeuralGraphRepresentation(
        NeuralGraphRepresentationConfig(
            graph_spec=NeuralGraphSpec.temporal_tcn(
                input_dim=int(input_dim),
                sequence_length=int(sequence_length),
                channels=tuple(int(v) for v in channels),
                kernel_size=int(kernel_size),
                dilation_base=int(dilation_base),
                output_dim=int(output_dim),
                activation=str(activation),
                dropout=float(dropout),
            ),
            random_seed=int(random_seed),
            representation_name="temporal_tcn_forecast",
        )
    )
    problem = TemporalNeuralForecastingProblem(data, head_name="forecast")
    return _build_neural_graph_trainer(
        problem=problem,
        representation=representation,
        compute_backend=compute_backend,
        optimizer=optimizer,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        max_grad_norm=max_grad_norm,
        device=device,
        device_policy=device_policy,
        random_seed=random_seed,
        run_name=run_name,
        resource_context=resource_context,
    )


def build_temporal_transformer_forecast_trainer(
    data: NumericDataView,
    *,
    input_dim: int = 1,
    sequence_length: int = 12,
    hidden_dim: int = 64,
    num_layers: int = 2,
    num_heads: int = 4,
    ffn_expansion_ratio: float = 4.0,
    output_dim: int = 1,
    activation: str = "gelu",
    dropout: float = 0.0,
    compute_backend: str = "torch",
    optimizer: str = "adamw",
    learning_rate: float = 1e-2,
    weight_decay: float = 0.0,
    max_grad_norm: float | None = 10.0,
    device: str = "cpu",
    device_policy: str = "fallback_cpu",
    random_seed: int = 42,
    run_name: str = "temporal_transformer_forecast",
    resource_context: Mapping[str, Any] | None = None,
) -> Any:
    representation = NeuralGraphRepresentation(
        NeuralGraphRepresentationConfig(
            graph_spec=NeuralGraphSpec.temporal_transformer(
                input_dim=int(input_dim),
                sequence_length=int(sequence_length),
                hidden_dim=int(hidden_dim),
                num_layers=int(num_layers),
                num_heads=int(num_heads),
                ffn_expansion_ratio=float(ffn_expansion_ratio),
                output_dim=int(output_dim),
                activation=str(activation),
                dropout=float(dropout),
            ),
            random_seed=int(random_seed),
            representation_name="temporal_transformer_forecast",
        )
    )
    problem = TemporalNeuralForecastingProblem(data, head_name="forecast")
    return _build_neural_graph_trainer(
        problem=problem,
        representation=representation,
        compute_backend=compute_backend,
        optimizer=optimizer,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        max_grad_norm=max_grad_norm,
        device=device,
        device_policy=device_policy,
        random_seed=random_seed,
        run_name=run_name,
        resource_context=resource_context,
    )


def build_temporal_nbeats_forecast_trainer(
    data: NumericDataView,
    *,
    input_dim: int = 1,
    sequence_length: int = 12,
    hidden_dim: int = 64,
    theta_dim: int = 8,
    num_stacks: int = 2,
    num_blocks: int = 3,
    output_dim: int = 1,
    share_weights: bool = False,
    dropout: float = 0.0,
    compute_backend: str = "torch",
    optimizer: str = "adamw",
    learning_rate: float = 1e-2,
    weight_decay: float = 0.0,
    max_grad_norm: float | None = 10.0,
    device: str = "cpu",
    device_policy: str = "fallback_cpu",
    random_seed: int = 42,
    run_name: str = "temporal_nbeats_forecast",
    resource_context: Mapping[str, Any] | None = None,
) -> Any:
    representation = NeuralGraphRepresentation(
        NeuralGraphRepresentationConfig(
            graph_spec=NeuralGraphSpec.temporal_nbeats(
                input_dim=int(input_dim),
                sequence_length=int(sequence_length),
                hidden_dim=int(hidden_dim),
                theta_dim=int(theta_dim),
                num_stacks=int(num_stacks),
                num_blocks=int(num_blocks),
                output_dim=int(output_dim),
                share_weights=bool(share_weights),
                dropout=float(dropout),
            ),
            random_seed=int(random_seed),
            representation_name="temporal_nbeats_forecast",
        )
    )
    problem = TemporalNeuralForecastingProblem(data, head_name="forecast")
    return _build_neural_graph_trainer(
        problem=problem,
        representation=representation,
        compute_backend=compute_backend,
        optimizer=optimizer,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        max_grad_norm=max_grad_norm,
        device=device,
        device_policy=device_policy,
        random_seed=random_seed,
        run_name=run_name,
        resource_context=resource_context,
    )


def build_temporal_deepar_forecast_trainer(
    data: NumericDataView,
    *,
    input_dim: int = 1,
    sequence_length: int = 12,
    hidden_dim: int = 32,
    num_layers: int = 1,
    output_dim: int = 1,
    dropout: float = 0.0,
    bidirectional: bool = False,
    alpha: float = 0.1,
    compute_backend: str = "torch",
    optimizer: str = "adamw",
    learning_rate: float = 1e-2,
    weight_decay: float = 0.0,
    max_grad_norm: float | None = 10.0,
    device: str = "cpu",
    device_policy: str = "fallback_cpu",
    random_seed: int = 42,
    run_name: str = "temporal_deepar_forecast",
    resource_context: Mapping[str, Any] | None = None,
) -> Any:
    representation = NeuralGraphRepresentation(
        NeuralGraphRepresentationConfig(
            graph_spec=NeuralGraphSpec.temporal_deepar(
                input_dim=int(input_dim),
                sequence_length=int(sequence_length),
                hidden_dim=int(hidden_dim),
                num_layers=int(num_layers),
                output_dim=int(output_dim),
                dropout=float(dropout),
                bidirectional=bool(bidirectional),
            ),
            random_seed=int(random_seed),
            representation_name="temporal_deepar_forecast",
        )
    )
    problem = TemporalNeuralProbabilisticForecastingProblem(data, head_name="deepar", alpha=float(alpha))
    return _build_neural_graph_trainer(
        problem=problem,
        representation=representation,
        compute_backend=compute_backend,
        optimizer=optimizer,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        max_grad_norm=max_grad_norm,
        device=device,
        device_policy=device_policy,
        random_seed=random_seed,
        run_name=run_name,
        resource_context=resource_context,
    )


def build_temporal_patchtst_forecast_trainer(
    data: NumericDataView,
    *,
    input_dim: int = 1,
    sequence_length: int = 24,
    patch_len: int = 8,
    stride: int | None = None,
    hidden_dim: int = 64,
    num_layers: int = 2,
    num_heads: int = 4,
    ffn_dim: int | None = None,
    output_dim: int = 1,
    dropout: float = 0.0,
    compute_backend: str = "torch",
    optimizer: str = "adamw",
    learning_rate: float = 1e-2,
    weight_decay: float = 0.0,
    max_grad_norm: float | None = 10.0,
    device: str = "cpu",
    device_policy: str = "fallback_cpu",
    random_seed: int = 42,
    run_name: str = "temporal_patchtst_forecast",
    resource_context: Mapping[str, Any] | None = None,
) -> Any:
    representation = NeuralGraphRepresentation(
        NeuralGraphRepresentationConfig(
            graph_spec=NeuralGraphSpec.temporal_patchtst(
                input_dim=int(input_dim),
                sequence_length=int(sequence_length),
                patch_len=int(patch_len),
                stride=int(stride or patch_len),
                hidden_dim=int(hidden_dim),
                num_layers=int(num_layers),
                num_heads=int(num_heads),
                ffn_dim=int(ffn_dim or hidden_dim * 4),
                output_dim=int(output_dim),
                dropout=float(dropout),
            ),
            random_seed=int(random_seed),
            representation_name="temporal_patchtst_forecast",
        )
    )
    problem = TemporalNeuralForecastingProblem(data, head_name="forecast")
    return _build_neural_graph_trainer(
        problem=problem,
        representation=representation,
        compute_backend=compute_backend,
        optimizer=optimizer,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        max_grad_norm=max_grad_norm,
        device=device,
        device_policy=device_policy,
        random_seed=random_seed,
        run_name=run_name,
        resource_context=resource_context,
    )


def build_temporal_tft_forecast_trainer(
    data: NumericDataView,
    *,
    input_dim: int = 1,
    sequence_length: int = 12,
    hidden_dim: int = 64,
    num_layers: int = 2,
    num_heads: int = 4,
    output_dim: int = 1,
    dropout: float = 0.0,
    compute_backend: str = "torch",
    optimizer: str = "adamw",
    learning_rate: float = 1e-2,
    weight_decay: float = 0.0,
    max_grad_norm: float | None = 10.0,
    device: str = "cpu",
    device_policy: str = "fallback_cpu",
    random_seed: int = 42,
    run_name: str = "temporal_tft_forecast",
    resource_context: Mapping[str, Any] | None = None,
) -> Any:
    representation = NeuralGraphRepresentation(
        NeuralGraphRepresentationConfig(
            graph_spec=NeuralGraphSpec.temporal_tft(
                input_dim=int(input_dim),
                sequence_length=int(sequence_length),
                hidden_dim=int(hidden_dim),
                num_layers=int(num_layers),
                num_heads=int(num_heads),
                output_dim=int(output_dim),
                dropout=float(dropout),
            ),
            random_seed=int(random_seed),
            representation_name="temporal_tft_forecast",
        )
    )
    problem = TemporalNeuralForecastingProblem(data, head_name="forecast")
    return _build_neural_graph_trainer(
        problem=problem,
        representation=representation,
        compute_backend=compute_backend,
        optimizer=optimizer,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        max_grad_norm=max_grad_norm,
        device=device,
        device_policy=device_policy,
        random_seed=random_seed,
        run_name=run_name,
        resource_context=resource_context,
    )


def build_tabular_tabnet_classification_trainer(
    data: NumericDataView,
    *,
    input_dim: int | None = None,
    hidden_dim: int = 64,
    n_steps: int = 4,
    relaxation_factor: float = 1.5,
    ghost_bn: bool = True,
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
    run_name: str = "tabular_tabnet_classification",
    resource_context: Mapping[str, Any] | None = None,
) -> Any:
    representation = NeuralGraphRepresentation(
        NeuralGraphRepresentationConfig(
            graph_spec=NeuralGraphSpec.tabular_tabnet(
                input_dim=int(input_dim or data.X_train.shape[1]),
                hidden_dim=int(hidden_dim),
                n_steps=int(n_steps),
                relaxation_factor=float(relaxation_factor),
                ghost_bn=bool(ghost_bn),
                dropout=float(dropout),
                heads=({"kind": "classification", "name": "classification", "params": {"num_classes": int(num_classes)}},),
            ),
            random_seed=int(random_seed),
            representation_name="tabular_tabnet_classification",
        )
    )
    problem = TabularNeuralClassificationProblem(data)
    return _build_neural_graph_trainer(
        problem=problem,
        representation=representation,
        compute_backend=compute_backend,
        optimizer=optimizer,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        max_grad_norm=max_grad_norm,
        device=device,
        device_policy=device_policy,
        random_seed=random_seed,
        run_name=run_name,
        resource_context=resource_context,
    )


def build_tabular_tabnet_regression_trainer(
    data: NumericDataView,
    *,
    input_dim: int | None = None,
    hidden_dim: int = 64,
    n_steps: int = 4,
    relaxation_factor: float = 1.5,
    ghost_bn: bool = True,
    dropout: float = 0.0,
    output_dim: int = 1,
    compute_backend: str = "torch",
    optimizer: str = "adamw",
    learning_rate: float = 1e-2,
    weight_decay: float = 0.0,
    max_grad_norm: float | None = 10.0,
    device: str = "cpu",
    device_policy: str = "fallback_cpu",
    random_seed: int = 42,
    run_name: str = "tabular_tabnet_regression",
    resource_context: Mapping[str, Any] | None = None,
) -> Any:
    representation = NeuralGraphRepresentation(
        NeuralGraphRepresentationConfig(
            graph_spec=NeuralGraphSpec.tabular_tabnet(
                input_dim=int(input_dim or data.X_train.shape[1]),
                hidden_dim=int(hidden_dim),
                n_steps=int(n_steps),
                relaxation_factor=float(relaxation_factor),
                ghost_bn=bool(ghost_bn),
                dropout=float(dropout),
                heads=({"kind": "point", "name": "point", "params": {"output_dim": int(output_dim)}},),
            ),
            random_seed=int(random_seed),
            representation_name="tabular_tabnet_regression",
        )
    )
    problem = TabularNeuralRegressionProblem(data)
    return _build_neural_graph_trainer(
        problem=problem,
        representation=representation,
        compute_backend=compute_backend,
        optimizer=optimizer,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        max_grad_norm=max_grad_norm,
        device=device,
        device_policy=device_policy,
        random_seed=random_seed,
        run_name=run_name,
        resource_context=resource_context,
    )


def _build_optimization_adapter(method: str, **kwargs: Any) -> Any:
    from mlblack.integrations.nsgablack_optimization import build_optimization_adapter

    return build_optimization_adapter(method, **kwargs)


def _build_learning_solver(**kwargs: Any) -> Any:
    from mlblack.integrations.nsgablack_control import build_learning_solver

    return build_learning_solver(**kwargs)


def _build_neural_graph_trainer(
    *,
    problem: Any,
    representation: NeuralGraphRepresentation,
    compute_backend: str,
    optimizer: str,
    learning_rate: float,
    weight_decay: float,
    max_grad_norm: float | None,
    device: str,
    device_policy: str,
    random_seed: int,
    run_name: str,
    resource_context: Mapping[str, Any] | None = None,
) -> Any:
    backend_name = str(compute_backend or "torch").strip().lower()
    if backend_name == "torch":
        if (
            str(device or "cpu").strip().lower() not in {"", "cpu", "auto"}
            and resource_context is None
            and str(device_policy or "fallback_cpu").strip().lower() == "strict"
        ):
            raise ValueError(
                "strict accelerator use requires an injected Project L0 resource_context"
            )
        from mlblack.backends.torch_neural import TorchEvaluationProviderConfig
        from mlblack.integrations.nsgablack_gradient import build_gradient_trainer

        return build_gradient_trainer(
            problem=problem,
            representation=representation,
            method=optimizer,
            compute_backend="torch",
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            max_gradient_norm=max_grad_norm,
            resource_context=resource_context,
            provider_config=TorchEvaluationProviderConfig(
                random_seed=int(random_seed),
                publish_state_refs=True,
            ),
            run_name=run_name,
        )
    from mlblack.integrations.nsgablack_gradient import build_gradient_trainer

    return build_gradient_trainer(
        problem=problem,
        representation=representation,
        method=optimizer,
        compute_backend=backend_name,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        max_gradient_norm=max_grad_norm,
        run_name=run_name,
        resource_context=resource_context,
    )


def _compute_backend_spec(compute_backend: Any, device: str, device_policy: str) -> ComputeBackendSpec:
    if isinstance(compute_backend, Mapping):
        payload = dict(compute_backend)
        payload.setdefault("device", device)
        payload.setdefault("device_policy", device_policy)
        return ComputeBackendSpec.from_value(payload)
    return ComputeBackendSpec(name=str(compute_backend), device=str(device), device_policy=str(device_policy))
