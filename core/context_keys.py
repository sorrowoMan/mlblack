from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence


# mlblack uses nsgablack-style string context keys. This registry is for
# validation and normalization only; components should declare plain strings in
# context_requires/context_provides/context_mutates/context_cache.
REGISTERED_CONTEXT_KEYS: tuple[str, ...] = (
    "adapter.best_state",
    "adapter.current_state",
    "adapter.search_state",
    "adapter.state",
    "artifact.model",
    "artifact.report",
    "artifact.state",
    "artifact.symbolic_basis_ref",
    "artifact.symbolic_task_ref",
    "artifact.viewer",
    "base_decoder",
    "backend.capability",
    "backend.contract",
    "backend.device",
    "backend.device_policy",
    "backend.name",
    "backend.requested_name",
    "backend.session",
    "basis.artifact_ref",
    "basis.candidate_ref",
    "basis.consensus",
    "basis.fitted_ref",
    "basis.metrics",
    "basis.overlap_report",
    "branch.spec",
    "bias.branch",
    "bias.dynamic_pool",
    "bias.l2_scale",
    "bias.noop",
    "bias.objective_policy",
    "bias.objective_weights",
    "bias.soft_preference",
    "bias.state_l2_penalty",
    "branch_representations",
    "capability.side_effect",
    "candidate.branch",
    "candidate.interval_model",
    "candidate.model",
    "candidate.model_spec",
    "candidate.output",
    "candidate.probability_model",
    "candidate.repaired_state",
    "candidate.symbolic_basis_model",
    "candidate.unknown_state",
    "candidate.forecast_model",
    "checkpoint.ref",
    "data",
    "data.feature_names",
    "data.graphs",
    "data.image_pairs",
    "data.images",
    "data.numeric_view",
    "data.preference_pairs",
    "data.raw_rows",
    "data.schema",
    "data.target",
    "data.time_series_view",
    "data.X_train",
    "data.X_valid",
    "data.y_train",
    "data.y_valid",
    "pretrained.model",
    "pretrained.checkpoint_map",
    "pretrained.checkpoint_report",
    "pretrained.tokenizer",
    "estimator.factory",
    "event.decision",
    "experiment.records",
    "feedback.constraints",
    "feedback.gradients",
    "feedback.loss",
    "feedback.metrics",
    "feedback.objectives",
    "feedback.residuals",
    "feedback.signals",
    "fitted_estimator",
    "head.output",
    "model.parameter_gradient",
    "model.predict",
    "model.predict_interval",
    "model.predict_proba",
    "model.route",
    "model.transform",
    "model.logits",
    "model.hidden_states",
    "model.embeddings",
    "model.ranking_scores",
    "model.preference_scores",
    "neural.graph_spec",
    "neural.parameter_layout",
    "neural.optimizer_state",
    "neural.transformer_spec",
    "neural.hidden_states",
    "neural.attention_maps",
    "neural.ffn_activations",
    "neural.audit",
    "neural.audit.attention_summary",
    "neural.audit.attention_head_corr",
    "neural.audit.ffn_summary",
    "neural.audit.ffn_activation_sparsity",
    "orthogonal_feature_map",
    "pipeline.feature_space",
    "pipeline.component_state",
    "pipeline.conditional_features",
    "pipeline.fit_state",
    "population.candidates",
    "population.feedback",
    "population.snapshot_ref",
    "preference.reference_model",
    "problem.data.X_train",
    "problem.data.y_train",
    "representation.numpy_mlp_point",
    "resource.audit",
    "resource.context",
    "resource.device",
    "resource.lease",
    "resource_context",
    "router",
    "signal.pool",
    "signal.budget.remaining_ratio",
    "signal.gate.enabled",
    "snapshot.ref",
    "stage.audit",
    "stage.id",
    "symbolic.artifact",
    "symbolic.artifact_schema",
    "symbolic.branch_report",
    "symbolic.basis_model",
    "symbolic.candidate_pool",
    "symbolic.candidate_lineage",
    "symbolic.candidate_score",
    "symbolic.decoder_spec",
    "symbolic.equivalence_report",
    "symbolic.evaluation_events",
    "symbolic.expression_spec",
    "symbolic.function_pool",
    "symbolic.function_space",
    "symbolic.fold_report",
    "symbolic.genome",
    "symbolic.graph_cache",
    "symbolic.gradient_signal",
    "symbolic.native_structure_score",
    "symbolic.overfit_guard",
    "symbolic.parameter_specs",
    "symbolic.parameter_values",
    "symbolic.path_memory",
    "symbolic.pool_delta",
    "symbolic.primitive_registry",
    "symbolic.replay_record",
    "symbolic.search_policy",
    "symbolic.search_space",
    "symbolic.simplification_trace",
    "symbolic.structure_guard",
    "symbolic.truth_contract_recovery",
    "task.fitted_model_ref",
    "task.metrics",
    "text.token_ids",
    "time_series.decomposition",
    "time_series.horizon",
    "time_series.min_train_size",
    "time_series.objective_metrics",
    "time_series.search_space",
    "time_series.validation_size",
    "time_series.window_config",
    "tokenizer.vocab",
    "trainer.context",
    "trainer.get_state",
    "trainer.report",
    "trainer.snapshot_store",
    "trainer.step",
    "training.result",
    "training.task",
)

CONTEXT_KEY_SET = frozenset(REGISTERED_CONTEXT_KEYS)

# Compatibility aliases let older prototype strings normalize into the current
# registry without forcing every component to import constants.
CONTEXT_KEY_ALIASES: Mapping[str, str] = {
    "candidate": "candidate.unknown_state",
    "feedback": "population.feedback",
    "gradients": "feedback.gradients",
    "objectives": "feedback.objectives",
    "metrics": "feedback.metrics",
    "resources": "resource.context",
    "resource": "resource.context",
    "snapshot": "snapshot.ref",
}

METRIC_KEYS: tuple[str, ...] = (
    "objective",
    "loss",
    "accuracy",
    "auc_roc",
    "average_precision",
    "f1",
    "f1_macro",
    "mse",
    "rmse",
    "mae",
    "r2",
)

METRIC_FALLBACKS: tuple[str, ...] = ("strict", "safe_zero", "nan", "skip")


def normalize_context_key(key: str) -> str:
    value = str(key).strip()
    return CONTEXT_KEY_ALIASES.get(value, value)


def normalize_context_keys(keys: Iterable[str] | None) -> tuple[str, ...]:
    seen: set[str] = set()
    out: list[str] = []
    for key in tuple(keys or ()): 
        normalized = normalize_context_key(str(key))
        if normalized and normalized not in seen:
            seen.add(normalized)
            out.append(normalized)
    return tuple(out)


def unknown_context_keys(keys: Iterable[str] | None) -> tuple[str, ...]:
    return tuple(key for key in normalize_context_keys(keys) if key not in CONTEXT_KEY_SET)


def validate_context_keys(keys: Iterable[str] | None, *, strict: bool = False) -> tuple[str, ...]:
    unknown = unknown_context_keys(keys)
    if strict and unknown:
        raise ValueError(f"unknown context keys: {unknown}")
    return unknown


def register_context_keys(keys: Sequence[str]) -> tuple[str, ...]:
    """Return a deterministic merged registry tuple for external tooling.

    The built-in registry is immutable on purpose; project-level tooling can use
    this helper to build an extended validation set without mutating framework
    globals.
    """

    return tuple(sorted({*CONTEXT_KEY_SET, *(normalize_context_key(key) for key in keys)}))
