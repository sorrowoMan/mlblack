from __future__ import annotations

from typing import Any, Mapping

from mlblack.bias import (
    BranchPolicyBias,
    DynamicPoolBias,
    L2ScaleBias,
    NoopBias,
    ObjectivePolicyBias,
    ObjectiveWeightBias,
    StateL2Bias,
)
from mlblack.capabilities import (
    CheckpointCapability,
    CheckpointConfig,
    ExperimentTrackerCapability,
    ExperimentTrackerConfig,
    ResourceAuditCapability,
    SQLiteExperimentStore,
)
from mlblack.core import Trainer
from mlblack.pipeline.data_views import NumericDataView
from mlblack.pipeline import (
    ConditionalPrimitiveFeatureComponent,
    DataPipeline,
    FeatureSpaceComponent,
    IdentityComponent,
    SelectColumnsComponent,
    ZScoreNormalizeComponent,
)
from mlblack.assembly.spec import BiasSpec, CapabilitySpec, ComponentSpec, DataPipelineSpec, TrainerAssemblySpec


def build_pipeline(spec: DataPipelineSpec | Mapping[str, Any] | None = None) -> DataPipeline:
    pipeline_spec = DataPipelineSpec.from_value(spec)
    components = []
    for component_spec in pipeline_spec.component_specs():
        if not component_spec.enabled:
            continue
        components.append(_build_pipeline_component(component_spec))
    return DataPipeline(components, name=pipeline_spec.name)


def build_trainer(spec: TrainerAssemblySpec | Mapping[str, Any], data: NumericDataView) -> Trainer:
    """Build one inner ML trainer.

    This function intentionally does not build workflow/stage/group/backend
    orchestration. nsgablack owns those layers. mlblack only assembles the
    ML-specific inner trainer surface.
    """

    trainer_spec = TrainerAssemblySpec.from_value(spec)
    preset = str(trainer_spec.preset).strip().lower()
    params = trainer_spec.effective_params()
    trainer = _build_preset_trainer(preset, data, params)
    resource_context = dict(trainer_spec.resource_context)
    if resource_context:
        trainer.set_resource_context(resource_context)
    for bias_spec in trainer_spec.bias_specs():
        if bias_spec.enabled:
            trainer.add_bias(_build_bias(bias_spec))
    for capability_spec in trainer_spec.capability_specs():
        if capability_spec.enabled:
            trainer.add_capability(_build_capability(capability_spec))
    return trainer


def _build_pipeline_component(spec: ComponentSpec) -> Any:
    name = str(spec.name).strip().lower()
    params = dict(spec.params)
    if name in {"", "identity", "none"}:
        return IdentityComponent()
    if name in {"zscore", "zscore_normalize", "standardize", "standard_scaler"}:
        return ZScoreNormalizeComponent(**params)
    if name in {"select_columns", "columns"}:
        return SelectColumnsComponent(**params)
    if name in {"feature_space", "features"}:
        return FeatureSpaceComponent(**params)
    if name in {"conditional_primitives", "conditional_features"}:
        return ConditionalPrimitiveFeatureComponent(**params)
    raise ValueError(f"unknown pipeline component: {spec.name}")


def _build_capability(spec: CapabilitySpec) -> Any:
    name = str(spec.name).strip().lower()
    params = dict(spec.params)
    if name in {"checkpoint", "checkpoint_capability"}:
        return CheckpointCapability(CheckpointConfig(**params))
    if name in {"resource_audit", "resource"}:
        return ResourceAuditCapability()
    if name in {"experiment_tracker", "tracker", "tracking"}:
        store_cfg = params.pop("store", None)
        store = None
        if isinstance(store_cfg, Mapping) and str(store_cfg.get("backend", "")).lower() == "sqlite":
            store = SQLiteExperimentStore(str(store_cfg.get("path", "runs/mlblack_experiments.sqlite3")))
        return ExperimentTrackerCapability(store=store, config=ExperimentTrackerConfig(**params))
    raise ValueError(f"unknown capability: {spec.name}")


def _build_bias(spec: BiasSpec) -> Any:
    name = str(spec.name).strip().lower()
    params = dict(spec.params)
    if name in {"noop", "noop_bias"}:
        return NoopBias()
    if name in {"objective_weight", "objective_weight_bias"}:
        return ObjectiveWeightBias(**params)
    if name in {"state_l2", "state_l2_bias", "l2"}:
        return StateL2Bias(**params)
    if name in {"l2_scale", "l2_scale_bias"}:
        return L2ScaleBias(**params)
    if name in {"objective_policy", "objective_policy_bias"}:
        return ObjectivePolicyBias(**params)
    if name in {"branch_policy", "branch_policy_bias"}:
        return BranchPolicyBias(**params)
    if name in {"dynamic_pool", "dynamic_pool_bias"}:
        return DynamicPoolBias(**params)
    raise ValueError(f"unknown bias: {spec.name}")


def _build_preset_trainer(preset: str, data: NumericDataView, params: Mapping[str, Any]) -> Trainer:
    kwargs = dict(params)
    if preset in {"orthogonal_linear_point", "linear_point", "orthogonal_point"}:
        from mlblack.presets import build_orthogonal_linear_point_trainer

        return build_orthogonal_linear_point_trainer(data, **kwargs)
    if preset in {"orthogonal_linear_interval", "linear_interval", "interval"}:
        from mlblack.presets import build_orthogonal_linear_interval_trainer

        return build_orthogonal_linear_interval_trainer(data, **kwargs)
    if preset in {"tree_estimator_search", "tree", "random_forest"}:
        from mlblack.presets import build_tree_estimator_search_trainer

        return build_tree_estimator_search_trainer(data, **kwargs)
    if preset in {"tree_boosting_estimator_search", "tree_boosting", "xgboost", "xgb"}:
        from mlblack.presets import build_tree_boosting_estimator_search_trainer

        return build_tree_boosting_estimator_search_trainer(data, **kwargs)
    if preset in {"numpy_mlp_torch_backprop", "torch_mlp", "neural_torch"}:
        from mlblack.presets import build_numpy_mlp_torch_backprop_trainer

        return build_numpy_mlp_torch_backprop_trainer(data, **kwargs)
    if preset in {"sklearn_mlp_estimator_search", "sklearn_mlp"}:
        from mlblack.presets import build_sklearn_mlp_estimator_search_trainer

        return build_sklearn_mlp_estimator_search_trainer(data, **kwargs)
    if preset in {"tiny_transformer_classification", "tiny_transformer_classifier", "mini_transformer_classification"}:
        from mlblack.presets import build_tiny_transformer_classification_trainer

        return build_tiny_transformer_classification_trainer(data, **kwargs)
    if preset in {"tiny_transformer_lm", "tiny_transformer_language_model", "mini_transformer_lm", "tiny_lm"}:
        from mlblack.presets import build_tiny_transformer_lm_trainer

        return build_tiny_transformer_lm_trainer(data, **kwargs)
    if preset in {"tiny_transformer_dpo", "tiny_transformer_preference", "tiny_transformer_dpo_preference"}:
        from mlblack.presets import build_tiny_transformer_dpo_preference_trainer

        return build_tiny_transformer_dpo_preference_trainer(data, **kwargs)
    if preset in {"tiny_cnn_image_classification", "tiny_cnn", "cnn_image"}:
        from mlblack.presets import build_tiny_cnn_image_classification_trainer

        return build_tiny_cnn_image_classification_trainer(data, **kwargs)
    if preset in {"tiny_gnn_graph_classification", "tiny_gnn", "gnn_graph"}:
        from mlblack.presets import build_tiny_gnn_graph_classification_trainer

        return build_tiny_gnn_graph_classification_trainer(data, **kwargs)
    if preset in {"tiny_cnn_image_contrastive", "tiny_cnn_retrieval", "image_retrieval"}:
        from mlblack.presets import build_tiny_cnn_image_contrastive_trainer

        return build_tiny_cnn_image_contrastive_trainer(data, **kwargs)
    if preset in {"orthogonal_logistic_classification", "logistic_classification", "binary_classification"}:
        from mlblack.presets import build_orthogonal_logistic_classification_trainer

        return build_orthogonal_logistic_classification_trainer(data, **kwargs)
    if preset in {"orthogonal_softmax_classification", "softmax_classification", "multiclass_classification"}:
        from mlblack.presets import build_orthogonal_softmax_classification_trainer

        return build_orthogonal_softmax_classification_trainer(data, **kwargs)
    if preset in {"temporal_lstm", "temporal_lstm_forecast"}:
        from mlblack.presets import build_temporal_lstm_forecast_trainer

        return build_temporal_lstm_forecast_trainer(data, **kwargs)
    if preset in {"temporal_tcn", "temporal_tcn_forecast"}:
        from mlblack.presets import build_temporal_tcn_forecast_trainer

        return build_temporal_tcn_forecast_trainer(data, **kwargs)
    if preset in {"temporal_transformer", "temporal_transformer_forecast"}:
        from mlblack.presets import build_temporal_transformer_forecast_trainer

        return build_temporal_transformer_forecast_trainer(data, **kwargs)
    if preset in {"temporal_nbeats", "temporal_nbeats_forecast"}:
        from mlblack.presets import build_temporal_nbeats_forecast_trainer

        return build_temporal_nbeats_forecast_trainer(data, **kwargs)
    if preset in {"temporal_deepar", "temporal_deepar_forecast"}:
        from mlblack.presets import build_temporal_deepar_forecast_trainer

        return build_temporal_deepar_forecast_trainer(data, **kwargs)
    if preset in {"temporal_patchtst", "temporal_patchtst_forecast"}:
        from mlblack.presets import build_temporal_patchtst_forecast_trainer

        return build_temporal_patchtst_forecast_trainer(data, **kwargs)
    if preset in {"temporal_tft", "temporal_tft_forecast"}:
        from mlblack.presets import build_temporal_tft_forecast_trainer

        return build_temporal_tft_forecast_trainer(data, **kwargs)
    if preset in {"tabular_tabnet_classification", "tabnet_classification", "tabnet_classifier"}:
        from mlblack.presets import build_tabular_tabnet_classification_trainer

        return build_tabular_tabnet_classification_trainer(data, **kwargs)
    if preset in {"tabular_tabnet_regression", "tabnet_regression", "tabnet_regressor"}:
        from mlblack.presets import build_tabular_tabnet_regression_trainer

        return build_tabular_tabnet_regression_trainer(data, **kwargs)
    raise ValueError(f"unknown trainer preset: {preset}")
