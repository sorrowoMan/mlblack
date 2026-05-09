from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

from core.common.family_router import serialize_family_route_registry
from config import create_default_config
from core.orchestration.capabilities import FlowCapability
from core.linear.trainer_family import build_ridge_family_spec
from core.linear.trainer_family import LINEAR_FORMAL_PRESET_KEY, build_unified_linear_family_spec, linear_route_registry
from core.neural.trainer_family import (
    NEURAL_FORMAL_PRESET_KEY,
    build_sklearn_mlp_family_spec,
    build_torch_mlp_family_spec,
    build_unified_neural_family_spec,
    neural_route_registry,
)
from core.symbolic.artifact_schema import (
    merge_symbolic_artifact_schema_descriptors,
    symbolic_artifact_schema_descriptor,
)
from core.symbolic.trainer_family import (
    SYMBOLIC_FORMAL_PRESET_KEY,
    SYMBOLIC_LEGACY_PRESET_KEYS,
    build_unified_symbolic_family_spec,
    canonical_symbolic_preset_key,
    is_legacy_symbolic_preset,
    legacy_symbolic_family_spec,
    serialize_symbolic_route_registry,
    symbolic_route_registry,
)
from core.tree.trainer_family import (
    TREE_ENSEMBLE_FORMAL_PRESET_KEY,
    build_adaboost_family_spec,
    build_bagging_family_spec,
    build_extra_trees_family_spec,
    build_random_forest_family_spec,
    build_unified_tree_ensemble_family_spec,
    tree_ensemble_route_registry,
)
from core.tree_boosting.trainer_family import (
    TREE_BOOSTING_FORMAL_PRESET_KEY,
    build_unified_tree_boosting_family_spec,
    build_xgboost_family_spec,
    tree_boosting_route_registry,
)
from .i18n import build_entry_i18n_fields

ROOT = Path(__file__).resolve().parents[1]

_PROFILE_EXCLUDES: dict[str, frozenset[str]] = {
    "default": frozenset(),
    "framework-core": frozenset({"doc", "example"}),
}

_BASE_ENTRY_FIELDS: tuple[str, ...] = ("id", "key", "kind", "name", "source", "path", "tags", "summary")

_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "id": ("id", "key"),
    "key": ("key", "id"),
    "family": ("family", "families"),
    "families": ("families", "family"),
    "component": ("component", "name"),
    "components": ("components", "component"),
    "preset": ("preset", "trainer", "name"),
    "trainer": ("trainer", "preset", "name"),
    "head": ("head", "heads"),
    "heads": ("heads", "head"),
    "provider": ("provider", "name"),
    "providers": ("providers", "provider"),
    "plugin": ("plugin", "name"),
    "plugins": ("plugins", "plugin"),
    "tag": ("tags",),
}

_CANONICAL_FAMILY_INFO: dict[str, dict[str, str]] = {
    "linear": {
        "name": "linear",
        "summary": "Fixed linear function family with parameter fitting over a predefined affine form.",
    },
    "neural": {
        "name": "neural",
        "summary": "Fixed neural backbone family trained mainly through gradient-based optimization.",
    },
    "tree_ensemble": {
        "name": "tree_ensemble",
        "summary": "Tree ensemble family built from bagging or forest-style aggregation over tree learners.",
    },
    "tree_boosting": {
        "name": "tree_boosting",
        "summary": "Additive tree boosting family where sequential weak learners refine prediction state.",
    },
    "symbolic": {
        "name": "symbolic",
        "summary": "Structure-search family that jointly manages candidate structure and parameter fitting.",
    },
}

_HEAD_INFO: dict[str, dict[str, Any]] = {
    "point": {
        "summary": "Point prediction head that emits a single central estimate.",
        "default_outputs": ("mean",),
    },
    "interval": {
        "summary": "Interval prediction head that emits lower and upper bounds.",
        "default_outputs": ("lower", "upper"),
    },
}

_UI_FACET_FIELDS_BY_KIND: dict[str, tuple[str, ...]] = {
    "preset": (
        "family",
        "head",
        "runtime_backend",
        "status",
        "surface_status",
        "preset_kind",
        "supports_resume",
        "supports_warm_start",
        "family_route_keys",
        "family_route_formal_preset",
        "symbolic_route_keys",
        "symbolic_route_backends",
        "symbolic_route_tasks",
        "artifact_stability_fields",
        "search_mechanism_keys",
        "search_family_signature_mechanisms",
    ),
    "family": (
        "family",
        "heads",
        "runtime_backends",
        "parameter_backends",
        "supports_resume",
        "family_route_keys",
        "symbolic_route_keys",
        "symbolic_route_backends",
        "symbolic_route_tasks",
        "artifact_stability_fields",
        "search_mechanism_keys",
        "search_family_signature_mechanisms",
    ),
    "head": (
        "head",
        "families",
        "objective_families",
        "outputs",
        "family_route_keys",
        "symbolic_route_keys",
        "symbolic_route_backends",
        "symbolic_route_tasks",
        "artifact_stability_fields",
        "search_mechanism_keys",
        "search_family_signature_mechanisms",
    ),
    "component": ("component_surface", "component_kind", "mount_point", "applicable_families", "binding_level", "status"),
    "provider": ("provider_surface", "plane", "mount_point", "supports_batch", "supports_individual", "status"),
    "plugin": ("plugin_surface", "lifecycle_plane", "mount_point", "is_algorithmic", "enabled_by_default", "status"),
}

_PRESET_STATUS: dict[str, str] = {
    "linear": "stable",
    "ridge": "stable",
    "neural": "stable",
    "mlp_torch": "stable",
    "sklearn_mlp": "stable",
    "tree_ensemble": "stable",
    "tree_boosting": "stable",
    "xgboost": "stable",
    "random_forest": "stable",
    "extra_trees": "stable",
    "bagging": "stable",
    "adaboost": "stable",
    "symbolic": "stable",
    "symbolic_stagewise": "legacy",
    "symbolic_torch": "legacy",
    "symbolic_torch_interval": "legacy",
}

_PRESET_SURFACE_STATUS: dict[str, str] = {
    "linear": "formal",
    "ridge": "route_target",
    "neural": "formal",
    "mlp_torch": "route_target",
    "sklearn_mlp": "route_target",
    "tree_ensemble": "formal",
    "tree_boosting": "formal",
    "xgboost": "route_target",
    "random_forest": "route_target",
    "extra_trees": "route_target",
    "bagging": "route_target",
    "adaboost": "route_target",
    "symbolic": "formal",
    "symbolic_stagewise": "deprecated",
    "symbolic_torch": "deprecated",
    "symbolic_torch_interval": "deprecated",
}

_RUNTIME_MECHANISM_COMPONENT_INFO: dict[str, dict[str, Any]] = {
    "symbolic.benchmark.scenario_contracts": {
        "summary": "Framework-level symbolic benchmark scenario, truth-contract, and lane-contract definitions.",
        "component_kind": "benchmark_contracts",
        "applicable_presets": ("symbolic",),
        "signal_names": ("scenario_key", "strict_contract", "phase_equivalent_contract", "family_level_contract"),
        "provides_fields": ("SymbolicBenchmarkScenarioDefinition", "SymbolicBenchmarkTruthContract", "SymbolicBenchmarkLaneSpec"),
        "path": "core/symbolic/benchmark/contracts.py",
        "module": "core.symbolic.benchmark.contracts",
        "binding_level": "defining",
        "status": "stable",
    },
    "symbolic.benchmark.bundle_pipeline": {
        "summary": "Framework-level symbolic benchmark TrainDataBundle assembly and split pipeline.",
        "component_kind": "benchmark_bundle_pipeline",
        "applicable_presets": ("symbolic",),
        "signal_names": ("scenario_definition", "n_total", "train_ratio", "noise_std"),
        "provides_fields": ("TrainDataBundle", "truth_payload"),
        "path": "core/symbolic/benchmark/bundle_pipeline.py",
        "module": "core.symbolic.benchmark.bundle_pipeline",
        "binding_level": "defining",
        "status": "stable",
    },
    "symbolic.benchmark.hint_surface": {
        "summary": "Framework-level accessor surface for symbolic benchmark orchestration hints and lane policies.",
        "component_kind": "orchestration_hint_surface",
        "applicable_presets": ("symbolic",),
        "signal_names": ("orchestrator_hints", "search_hints"),
        "provides_fields": ("trainer_params_overrides", "lane_specs", "core_selection", "search_hints"),
        "path": "core/symbolic/benchmark/hints.py",
        "module": "core.symbolic.benchmark.hints",
        "binding_level": "bound",
        "status": "stable",
    },
    "symbolic.benchmark.outer_proxy_contracts": {
        "summary": "Framework-level outer-solver candidate/result and evaluation-proxy contracts for symbolic structure search.",
        "component_kind": "outer_solver_interface",
        "applicable_presets": ("symbolic",),
        "signal_names": ("basis_objects", "chart_variants", "realization_heads", "branch_specs"),
        "provides_fields": ("SymbolicOuterSearchCandidate", "SymbolicOuterEvaluationResult", "SymbolicOuterEvaluationProxyProtocol"),
        "path": "core/symbolic/benchmark/outer_proxy.py",
        "module": "core.symbolic.benchmark.outer_proxy",
        "binding_level": "defining",
        "status": "stable",
    },
    "symbolic.known_relation.problem_registry": {
        "summary": "Known-relation symbolic benchmark scenario instances and scenario registry.",
        "component_kind": "benchmark_instance_registry",
        "applicable_presets": ("symbolic",),
        "signal_names": ("scenario_key", "truth_expression", "feature_names"),
        "provides_fields": ("known_relation_benchmark_definition", "known_relation_scenario_keys"),
        "path": "my_project/known_relation_symbolic/problem/registry.py",
        "module": "my_project.known_relation_symbolic.problem.registry",
        "binding_level": "defining",
        "status": "scaffold",
    },
    "symbolic.known_relation.synthetic_generators": {
        "summary": "Known-relation project scenario generators (instance-level mechanism data source).",
        "component_kind": "benchmark_instance_generator",
        "applicable_presets": ("symbolic",),
        "signal_names": ("seed", "n_total", "noise_std"),
        "provides_fields": ("X", "y", "truth", "truth_components"),
        "path": "my_project/known_relation_symbolic/problem/generators.py",
        "module": "my_project.known_relation_symbolic.problem.generators",
        "binding_level": "defining",
        "status": "scaffold",
    },
    "symbolic.known_relation.scaffold_assembly": {
        "summary": "Standard known-relation symbolic project assembly surface under my_project.",
        "component_kind": "project_scaffold",
        "applicable_presets": ("symbolic",),
        "signal_names": ("KnownRelationSymbolicBuildConfig",),
        "provides_fields": ("evaluation_proxy", "scenario_keys", "outer_solver_backend"),
        "path": "my_project/known_relation_symbolic/build_solver.py",
        "module": "my_project.known_relation_symbolic.build_solver",
        "binding_level": "defining",
        "status": "scaffold",
    },
    "orthogonal_source.layer": {
        "summary": "Model-agnostic source governance layer that builds de-redundant, tagged orthogonal source objects for downstream learners.",
        "component_kind": "representation_layer",
        "applicable_presets": (
            "linear",
            "ridge",
            "tree_ensemble",
            "random_forest",
            "tree_boosting",
            "xgboost",
            "neural",
            "symbolic",
        ),
        "signal_names": ("source_object", "basis_matrix", "source_stability", "pair_abs_corr", "target_task"),
        "provides_fields": ("OrthogonalSourceLayer", "OrthogonalSourceResult", "source_rows", "basis_matrix"),
        "path": "core/orthogonal_source/layer.py",
        "module": "core.orthogonal_source.layer",
        "binding_level": "optional",
        "status": "scaffold",
    },
    "orthogonal_source.known_relation_baseline_scaffold": {
        "summary": "Standard project scaffold comparing raw features vs orthogonal source objects across fixed downstream learners.",
        "component_kind": "project_scaffold",
        "applicable_presets": ("ridge", "random_forest", "tree_boosting"),
        "signal_names": ("known_relation_scenario", "orthogonal_source_table", "baseline_table"),
        "provides_fields": ("run_suite", "baseline_rows", "orthogonal_source_rows"),
        "path": "my_project/orthogonal_source_baseline/build_solver.py",
        "module": "my_project.orthogonal_source_baseline.build_solver",
        "binding_level": "defining",
        "status": "scaffold",
    },
    "orthogonal_source.image_classification_scaffold": {
        "summary": "Standard project scaffold for searchable image objectification formulas followed by class-aware orthogonal source governance across fixed classifiers.",
        "component_kind": "project_scaffold",
        "applicable_presets": ("linear", "random_forest", "tree_boosting", "neural"),
        "signal_names": (
            "image_dataset",
            "flattened_pixels",
            "representation_formula_table",
            "orthogonal_source_table",
            "classification_table",
        ),
        "provides_fields": ("run_suite", "classification_rows", "representation_formula_rows", "orthogonal_source_rows"),
        "path": "my_project/orthogonal_source_image_classification/build_solver.py",
        "module": "my_project.orthogonal_source_image_classification.build_solver",
        "binding_level": "defining",
        "status": "scaffold",
    },
    "orthogonal_source.phi_bundle_evaluation_proxy": {
        "summary": "Evaluation proxy that materializes an outer PhiBundle into image representation objects, runs orthogonal source governance, and returns multi-objective metrics.",
        "component_kind": "evaluation_proxy",
        "applicable_presets": ("linear", "neural", "symbolic", "nsgablack_outer_solver"),
        "signal_names": (
            "phi_bundle",
            "representation_formula_table",
            "orthogonal_source_table",
            "classification_error",
            "redundancy",
            "complexity",
            "instability",
            "cost",
        ),
        "provides_fields": ("evaluate_phi_bundle", "PhiBundleEvaluationConfig", "objectives", "metrics", "artifact_paths"),
        "path": "my_project/orthogonal_source_image_classification/pipeline/phi_bundle_proxy.py",
        "module": "my_project.orthogonal_source_image_classification.pipeline.phi_bundle_proxy",
        "binding_level": "defining",
        "status": "scaffold",
    },
    "orthogonal_source.etf_quant_interval_proxy": {
        "summary": "Standard ETF-style quant interval scaffold that builds real ETF return panel source objects, applies orthogonal source governance, and reports point/interval/rolling metrics.",
        "component_kind": "project_scaffold",
        "applicable_presets": ("linear", "random_forest", "tree_boosting", "interval"),
        "signal_names": (
            "etf_return_panel",
            "future_horizon_return",
            "orthogonal_source_table",
            "interval_metrics",
            "rolling_metrics",
            "rank_backtest_metrics",
        ),
        "provides_fields": (
            "run_suite",
            "baseline_metrics",
            "interval_metrics",
            "rolling_metrics",
            "rank_backtest_metrics",
            "orthogonal_source_rows",
        ),
        "path": "my_project/etf_quant_interval_proxy/build_solver.py",
        "module": "my_project.etf_quant_interval_proxy.build_solver",
        "binding_level": "defining",
        "status": "scaffold",
    },
    "state_signal_view.prediction_residual": {
        "summary": "Parent-model prediction and residual view for downstream adaptive policies.",
        "component_kind": "state_signal_view",
        "applicable_presets": ("mlp_torch", "sklearn_mlp", "xgboost", "random_forest", "extra_trees", "bagging", "adaboost"),
        "signal_names": ("prediction", "residual", "loss", "uncertainty"),
        "provides_fields": ("prediction_ref", "residual_ref", "per_sample_loss_ref", "uncertainty_ref"),
        "status": "stable",
    },
    "state_signal_view.gradient_norm": {
        "summary": "Gradient-norm state signal specialized for neural backbones that expose per-sample gradient magnitude.",
        "component_kind": "state_signal_view",
        "applicable_presets": ("mlp_torch",),
        "signal_names": ("gradient_norm",),
        "provides_fields": ("gradient_norm_ref",),
        "status": "stable",
    },
    "sample_weighting.loss_adaptive": {
        "summary": "Adaptive sample-weighting component driven by loss, uncertainty, or other state signals.",
        "component_kind": "sample_weighting",
        "applicable_presets": ("mlp_torch", "sklearn_mlp", "xgboost", "random_forest", "extra_trees", "bagging", "adaboost"),
        "signal_names": ("loss", "uncertainty", "sample_weight"),
        "provides_fields": ("sample_weight_ref",),
        "status": "stable",
    },
    "sampling.batch_priority_subsample": {
        "summary": "Priority-driven batch sampling component that pulls rows by runtime score instead of plain pre-fit slicing.",
        "component_kind": "sampling",
        "applicable_presets": ("mlp_torch", "sklearn_mlp", "xgboost", "random_forest", "extra_trees", "bagging", "adaboost"),
        "signal_names": ("gradient_norm", "loss", "uncertainty", "sample_weight"),
        "provides_fields": ("sample_index_ref", "batch_index_ref"),
        "status": "stable",
    },
    "sampling.row_feature_subsample": {
        "summary": "Row/feature subsampling component that trims training views without changing the preset identity.",
        "component_kind": "sampling",
        "applicable_presets": ("mlp_torch", "sklearn_mlp", "xgboost", "random_forest", "extra_trees", "bagging", "adaboost"),
        "signal_names": ("sample_index", "feature_index"),
        "provides_fields": ("sample_index_ref", "feature_index_ref"),
        "status": "stable",
    },
    "aggregation.ensemble_summary": {
        "summary": "Post-fit aggregation summary component that records active runtime signals and ensemble structure metadata.",
        "component_kind": "aggregation",
        "applicable_presets": ("mlp_torch", "sklearn_mlp", "xgboost", "random_forest", "extra_trees", "bagging", "adaboost"),
        "signal_names": ("aggregation_summary",),
        "provides_fields": ("aggregated_output_ref",),
        "status": "stable",
    },
}

_PROVIDER_INFO: dict[str, dict[str, Any]] = {
    "symbolic_scenario_evaluation_proxy": {
        "summary": "Framework-level symbolic scenario evaluation proxy for nsgablack outer solver integration.",
        "provider_surface": "evaluation_proxy",
        "plane": "nsgablack_outer_solver_bridge",
        "supports_batch": True,
        "supports_individual": True,
        "module": "core.symbolic.benchmark.outer_proxy",
        "path": "core/symbolic/benchmark/outer_proxy.py",
        "status": "stable",
    },
    "known_relation_symbolic_evaluation_proxy": {
        "summary": "Known-relation project instance of the framework symbolic scenario evaluation proxy.",
        "provider_surface": "evaluation_proxy",
        "plane": "nsgablack_outer_solver_bridge",
        "supports_batch": True,
        "supports_individual": True,
        "module": "my_project.known_relation_symbolic.mlblack_side.evaluation_proxy",
        "path": "my_project/known_relation_symbolic/mlblack_side/evaluation_proxy.py",
        "status": "scaffold",
    },
    "decision_evaluation_bridge": {
        "summary": "Bridge provider that decodes external decisions and delegates scoring to MLBLACK-side evaluators.",
        "provider_surface": "bridge",
        "plane": "problem_evaluation",
        "supports_batch": True,
        "supports_individual": True,
        "module": "problem.bridge",
        "path": "problem/bridge.py",
        "status": "stable",
    },
    "batch_evaluation_proxy_provider": {
        "summary": "Proxy provider that exposes MLBLACK batch evaluation to external solver control planes.",
        "provider_surface": "proxy",
        "plane": "solver_bridge",
        "supports_batch": True,
        "supports_individual": False,
        "module": "problem.proxy",
        "path": "problem/proxy.py",
        "status": "stable",
    },
}

_PLUGIN_CLASS_INFO: dict[str, dict[str, str]] = {
    "report_writer": {
        "module": "plugins.report_writer_plugin",
        "path": "plugins/report_writer_plugin.py",
        "surface": "flow_plugin",
        "summary": "Writes final summary artifacts through the plugin/capability plane.",
        "lifecycle_plane": "experiment",
        "hook_events": "on_experiment_finish",
        "priority": "200",
        "enabled_by_default": "true",
        "is_algorithmic": "false",
        "context_requires": "report_payload",
        "context_provides": "summary_path",
        "context_mutates": "summary_path",
        "context_cache": "",
    },
    "reproducibility": {
        "module": "plugins.reproducibility_plugin",
        "path": "plugins/reproducibility_plugin.py",
        "surface": "flow_plugin",
        "summary": "Applies deterministic seeding and records reproducibility metadata.",
        "lifecycle_plane": "experiment",
        "hook_events": "on_experiment_start",
        "priority": "10",
        "enabled_by_default": "true",
        "is_algorithmic": "false",
        "context_requires": "",
        "context_provides": "reproducibility,runtime_seed",
        "context_mutates": "reproducibility,runtime_seed",
        "context_cache": "",
    },
    "runtime_resource_cleanup": {
        "module": "plugins.runtime_resource_plugin",
        "path": "plugins/runtime_resource_plugin.py",
        "surface": "flow_plugin",
        "summary": "Closes runtime resources and clears cached handles on finish or error.",
        "lifecycle_plane": "experiment",
        "hook_events": "on_stage_error,on_experiment_finish,on_experiment_error",
        "priority": "300",
        "enabled_by_default": "true",
        "is_algorithmic": "false",
        "context_requires": "graph_cache_resource",
        "context_provides": "",
        "context_mutates": "graph_cache_resource",
        "context_cache": "graph_cache_resource",
    },
    "trainer_state_checkpoint": {
        "module": "plugins.trainer_state_checkpoint_plugin",
        "path": "plugins/trainer_state_checkpoint_plugin.py",
        "surface": "flow_plugin",
        "summary": "Persists trainer_state through the flow plugin plane.",
        "lifecycle_plane": "flow",
        "hook_events": "on_pre_persist",
        "priority": "250",
        "enabled_by_default": "true",
        "is_algorithmic": "false",
        "context_requires": "trainer,trainer_state",
        "context_provides": "trainer_state_checkpoint_path",
        "context_mutates": "report",
        "context_cache": "",
    },
}


@dataclass(frozen=True)
class CatalogEntry:
    key: str
    kind: str
    name: str
    source: str
    path: str | None = None
    tags: tuple[str, ...] = field(default_factory=tuple)
    summary: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)
    fields: Mapping[str, Any] = field(default_factory=dict)
    relations: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": str(self.key),
            "kind": str(self.kind),
            "name": str(self.name),
            "source": str(self.source),
            "path": self.path,
            "tags": list(self.tags),
            "summary": str(self.summary),
            "metadata": _jsonable(dict(self.metadata)),
            "fields": _jsonable(dict(self.fields)),
            "relations": _jsonable(dict(self.relations)),
        }


@dataclass(frozen=True)
class _PresetRow:
    key: str
    family: str
    default_head: str
    heads: tuple[str, ...]
    objective_family: str
    outputs: tuple[str, ...]
    head_profiles: Mapping[str, Mapping[str, Any]]
    backend: str
    runtime_backend: str
    parameter_backend: str
    supports_resume: bool
    supports_warm_start: bool
    supports_incremental: bool
    preset_kind: str
    mechanism_keys: tuple[str, ...]
    mechanism_kinds: tuple[str, ...]
    summary: str
    metadata: Mapping[str, Any]
    fields: Mapping[str, Any]
    relations: Mapping[str, Any]


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_jsonable(v) for v in value]
    return str(value)


def _normalize_profile(profile: str) -> str:
    key = str(profile).strip().lower()
    if key not in _PROFILE_EXCLUDES:
        known = ", ".join(sorted(_PROFILE_EXCLUDES.keys()))
        raise ValueError(f"Unknown catalog profile '{profile}'. Available: [{known}]")
    return key


def _normalize_kind(kind: str | None) -> str | None:
    if kind is None:
        return None
    key = str(kind).strip().lower()
    return key or None


def _normalize_field_name(name: str) -> str:
    return str(name).strip().lower().replace("-", "_")


def _normalize_field_filters(
    field_filters: Mapping[str, Any] | Sequence[tuple[str, Any]] | None,
) -> tuple[tuple[str, str], ...]:
    if not field_filters:
        return tuple()

    if isinstance(field_filters, Mapping):
        items = field_filters.items()
    else:
        items = tuple(field_filters)

    normalized: list[tuple[str, str]] = []
    for raw_key, raw_value in items:
        key = _normalize_field_name(str(raw_key))
        if not key:
            continue
        for value in _flatten_scalars(raw_value):
            text = str(value).strip().lower()
            if not text:
                continue
            normalized.append((key, text))
    return tuple(normalized)


def _canonical_family(trainer_key: str, metadata: Mapping[str, Any]) -> str:
    key = str(trainer_key).strip().lower()
    raw = str(metadata.get("family", "")).strip().lower()

    if key == "ridge" or raw == "linear":
        return "linear"
    if key in {"mlp_torch", "sklearn_mlp"} or raw in {"neural", "neural_network"}:
        return "neural"
    if key == "xgboost" or raw == "tree_boosting":
        return "tree_boosting"
    if key in {"random_forest", "extra_trees", "bagging", "adaboost"} or raw == "tree_ensemble":
        return "tree_ensemble"
    if key.startswith("symbolic") or raw.startswith("symbolic"):
        return "symbolic"
    return raw or key


@lru_cache(maxsize=None)
def _family_spec_description(trainer_key: str) -> dict[str, Any]:
    key = str(trainer_key).strip().lower()
    if key == LINEAR_FORMAL_PRESET_KEY:
        return build_unified_linear_family_spec().description_dict()
    if key == "ridge":
        return build_ridge_family_spec().description_dict()
    if key == NEURAL_FORMAL_PRESET_KEY:
        return build_unified_neural_family_spec().description_dict()
    if key == "mlp_torch":
        return build_torch_mlp_family_spec().description_dict()
    if key == "sklearn_mlp":
        return build_sklearn_mlp_family_spec().description_dict()
    if key == TREE_BOOSTING_FORMAL_PRESET_KEY:
        return build_unified_tree_boosting_family_spec().description_dict()
    if key == "xgboost":
        return build_xgboost_family_spec().description_dict()
    if key == TREE_ENSEMBLE_FORMAL_PRESET_KEY:
        return build_unified_tree_ensemble_family_spec().description_dict()
    if key == "random_forest":
        return build_random_forest_family_spec().description_dict()
    if key == "extra_trees":
        return build_extra_trees_family_spec().description_dict()
    if key == "bagging":
        return build_bagging_family_spec().description_dict()
    if key == "adaboost":
        return build_adaboost_family_spec().description_dict()
    if key == "symbolic":
        return build_unified_symbolic_family_spec().description_dict()
    if key in {"symbolic_stagewise", "symbolic_torch", "symbolic_torch_interval"}:
        return legacy_symbolic_family_spec(key).description_dict()
    return {}


def _supported_heads(trainer_key: str, default_head: str) -> tuple[str, ...]:
    key = str(trainer_key).strip().lower()
    if key == "symbolic":
        return ("point", "interval")
    return (default_head,)


def _family_route_rows_for_family(family: str) -> tuple[dict[str, Any], ...]:
    family_key = str(family).strip().lower()
    if family_key == "linear":
        return tuple(dict(row) for row in serialize_family_route_registry(linear_route_registry()))
    if family_key == "neural":
        return tuple(dict(row) for row in serialize_family_route_registry(neural_route_registry()))
    if family_key == "tree_ensemble":
        return tuple(dict(row) for row in serialize_family_route_registry(tree_ensemble_route_registry()))
    if family_key == "tree_boosting":
        return tuple(dict(row) for row in serialize_family_route_registry(tree_boosting_route_registry()))
    if family_key == "symbolic":
        return tuple(
            dict(row)
            for row in serialize_family_route_registry(
                tuple(route.as_family_route_spec() for route in symbolic_route_registry())
            )
        )
    return tuple()


def _formal_preset_for_family(family: str) -> str:
    family_key = str(family).strip().lower()
    if family_key == "linear":
        return LINEAR_FORMAL_PRESET_KEY
    if family_key == "neural":
        return NEURAL_FORMAL_PRESET_KEY
    if family_key == "tree_ensemble":
        return TREE_ENSEMBLE_FORMAL_PRESET_KEY
    if family_key == "tree_boosting":
        return TREE_BOOSTING_FORMAL_PRESET_KEY
    if family_key == "symbolic":
        return SYMBOLIC_FORMAL_PRESET_KEY
    return ""


def _family_routes_for_trainer(trainer_key: str, family: str) -> tuple[dict[str, Any], ...]:
    key = str(trainer_key).strip().lower()
    formal_preset = _formal_preset_for_family(family)
    route_rows = _family_route_rows_for_family(family)
    if key == formal_preset:
        return route_rows
    return tuple(row for row in route_rows if str(row.get("route_key", "")).strip().lower() == key)


def _symbolic_routes_for_trainer(trainer_key: str) -> tuple[dict[str, Any], ...]:
    key = str(trainer_key).strip().lower()
    route_rows = tuple(dict(row) for row in serialize_symbolic_route_registry())
    if key == SYMBOLIC_FORMAL_PRESET_KEY:
        return route_rows
    if key in SYMBOLIC_LEGACY_PRESET_KEYS:
        return tuple(row for row in route_rows if str(row.get("route_key", "")).strip().lower() == key)
    return ()


def _head_profiles_for_trainer(
    trainer_key: str,
    *,
    default_head: str,
    objective_family: str,
    outputs: tuple[str, ...],
) -> dict[str, dict[str, Any]]:
    task_head = dict(_family_spec_description(trainer_key).get("task_head", {}))
    profiles: dict[str, dict[str, Any]] = {
        str(default_head): {
            "objective_family": str(objective_family),
            "outputs": tuple(outputs),
            "calibration_mode": str(task_head.get("calibration_mode", "none")),
        }
    }
    key = str(trainer_key).strip().lower()
    if key == "symbolic":
        interval_payload = build_unified_symbolic_family_spec(task="interval").description_dict()
        interval_head = dict(interval_payload.get("task_head", {}))
        profiles["interval"] = {
            "objective_family": str(interval_head.get("objective_family", "quantile_interval")),
            "outputs": tuple(
                str(v).strip().lower()
                for v in tuple(interval_head.get("outputs", ("lower", "upper")))
                if str(v).strip()
            )
            or ("lower", "upper"),
            "calibration_mode": str(interval_head.get("calibration_mode", "none")),
        }
    return profiles


def _symbolic_artifact_descriptor_from_head_profiles(
    head_profiles: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    descriptors = []
    for head_name, profile in sorted(dict(head_profiles).items(), key=lambda item: item[0]):
        descriptors.append(
            symbolic_artifact_schema_descriptor(
                task=str(head_name),
                outputs=tuple(str(v) for v in tuple(dict(profile).get("outputs", ()))),
                objective_family=str(dict(profile).get("objective_family", "regression")),
                calibration_mode=str(dict(profile).get("calibration_mode", "none")),
                supports_piecewise=str(head_name).strip().lower() == "interval",
            )
        )
    return merge_symbolic_artifact_schema_descriptors(tuple(descriptors))


def _extract_backend_fields(family: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    if family == "linear":
        backend = dict(payload.get("backend", {}))
        return {
            "backend": str(backend.get("solver_kind", "ridge")),
            "runtime_backend": str(backend.get("runtime_backend", "numpy")),
            "parameter_backend": str(backend.get("parameter_backend", "closed_form")),
            "supports_resume": bool(backend.get("supports_resume", False)),
            "supports_warm_start": bool(backend.get("supports_warm_start", False)),
            "supports_incremental": bool(backend.get("supports_incremental", False)),
            "preset_kind": "closed_form_backend",
        }

    if family == "neural":
        backend = dict(payload.get("backend", {}))
        meta = dict(payload.get("metadata", {}))
        return {
            "backend": str(backend.get("runtime_backend", backend.get("parameter_backend", "")) or ""),
            "runtime_backend": str(backend.get("runtime_backend", "")),
            "parameter_backend": str(backend.get("parameter_backend", "")),
            "supports_resume": bool(backend.get("supports_resume", False)),
            "supports_warm_start": bool(backend.get("supports_warm_start", False)),
            "supports_incremental": bool(backend.get("supports_incremental", False)),
            "preset_kind": str(meta.get("preset_kind", "backend_variant") or "backend_variant"),
        }

    if family == "tree_boosting":
        backend = dict(payload.get("backend", {}))
        return {
            "backend": str(backend.get("backend", "xgboost")),
            "runtime_backend": str(backend.get("backend", "xgboost")),
            "parameter_backend": str(backend.get("backend", "xgboost")),
            "supports_resume": bool(backend.get("supports_resume", False)),
            "supports_warm_start": bool(backend.get("supports_warm_start", False)),
            "supports_incremental": bool(backend.get("supports_incremental", False)),
            "preset_kind": str(backend.get("trainer_kind", "gradient_boosted_trees") or "gradient_boosted_trees"),
        }

    if family == "tree_ensemble":
        ensemble = dict(payload.get("ensemble", {}))
        return {
            "backend": str(ensemble.get("ensemble_kind", "tree_ensemble")),
            "runtime_backend": str(ensemble.get("backend", "sklearn")),
            "parameter_backend": str(ensemble.get("backend", "sklearn")),
            "supports_resume": bool(ensemble.get("supports_resume", False)),
            "supports_warm_start": bool(ensemble.get("supports_warm_start", False)),
            "supports_incremental": bool(ensemble.get("supports_incremental", False)),
            "preset_kind": str(ensemble.get("ensemble_kind", "tree_ensemble") or "tree_ensemble"),
        }

    if family == "symbolic":
        backend = dict(payload.get("parameter_backend", {}))
        meta = dict(payload.get("metadata", {}))
        return {
            "backend": str(backend.get("backend", "")),
            "runtime_backend": str(backend.get("backend", "")),
            "parameter_backend": str(backend.get("backend", "")),
            "supports_resume": bool(backend.get("supports_resume", False)),
            "supports_warm_start": bool(backend.get("supports_warm_start", False)),
            "supports_incremental": bool(backend.get("supports_incremental", False)),
            "preset_kind": str(meta.get("preset_kind", "symbolic_variant") or "symbolic_variant"),
        }

    return {
        "backend": "",
        "runtime_backend": "",
        "parameter_backend": "",
        "supports_resume": False,
        "supports_warm_start": False,
        "supports_incremental": False,
        "preset_kind": "unknown",
    }


def _build_preset_row(trainer_key: str, metadata: Mapping[str, Any]) -> _PresetRow:
    payload = _family_spec_description(trainer_key)
    family = _canonical_family(trainer_key, metadata)
    task_head = dict(payload.get("task_head", {}))
    default_head = str(task_head.get("task", "point") or "point").strip().lower()
    heads = _supported_heads(trainer_key, default_head)
    outputs = tuple(str(v).strip().lower() for v in tuple(task_head.get("outputs", ())) if str(v).strip())
    objective_family = str(task_head.get("objective_family", "regression") or "regression").strip().lower()
    head_profiles = _head_profiles_for_trainer(
        trainer_key,
        default_head=default_head,
        objective_family=objective_family,
        outputs=outputs,
    )
    backend_fields = _extract_backend_fields(family, payload)
    mechanism_rows = tuple(dict(row) for row in metadata.get("mechanism_bindings", ()) if isinstance(row, Mapping))
    mechanism_keys = tuple(sorted({str(row.get("mechanism_key", "")).strip() for row in mechanism_rows if str(row.get("mechanism_key", "")).strip()}))
    mechanism_kinds = tuple(sorted({str(row.get("mechanism_kind", "")).strip() for row in mechanism_rows if str(row.get("mechanism_kind", "")).strip()}))
    search_contract_rows = tuple(
        dict(row) for row in payload.get("search_mechanism_contracts", ()) if isinstance(row, Mapping)
    )
    search_mechanism_keys = tuple(
        sorted(
            {
                str(row.get("mechanism_key", "")).strip()
                for row in search_contract_rows
                if str(row.get("mechanism_key", "")).strip()
            }
        )
    )
    search_mechanism_kinds = tuple(
        sorted(
            {
                str(row.get("mechanism_kind", "")).strip()
                for row in search_contract_rows
                if str(row.get("mechanism_kind", "")).strip()
            }
        )
    )
    search_checkpointable_mechanisms = tuple(
        sorted(
            {
                str(row.get("mechanism_key", "")).strip()
                for row in search_contract_rows
                if bool(row.get("checkpointable")) and str(row.get("mechanism_key", "")).strip()
            }
        )
    )
    search_replayable_mechanisms = tuple(
        sorted(
            {
                str(row.get("mechanism_key", "")).strip()
                for row in search_contract_rows
                if bool(row.get("replayable")) and str(row.get("mechanism_key", "")).strip()
            }
        )
    )
    search_family_signature_mechanisms = tuple(
        sorted(
            {
                str(row.get("mechanism_key", "")).strip()
                for row in search_contract_rows
                if bool(row.get("affects_family_signature")) and str(row.get("mechanism_key", "")).strip()
            }
        )
    )
    trainer_key_norm = str(trainer_key).strip().lower()
    status = str(_PRESET_STATUS.get(trainer_key_norm, "stable"))
    surface_status = str(_PRESET_SURFACE_STATUS.get(trainer_key_norm, "formal"))
    summary = (
        f"{family} preset '{trainer_key}' with default head '{default_head}' and backend '{backend_fields['runtime_backend'] or backend_fields['backend']}'."
    )
    if is_legacy_symbolic_preset(trainer_key):
        summary = (
            f"Deprecated symbolic legacy facade '{trainer_key}' that should migrate to preset '{SYMBOLIC_FORMAL_PRESET_KEY}'."
        )
    elif trainer_key_norm == SYMBOLIC_FORMAL_PRESET_KEY:
        summary = (
            "Formal symbolic family preset that routes backend/task/head combinations through one unified symbolic surface."
        )
    elif trainer_key_norm == LINEAR_FORMAL_PRESET_KEY:
        summary = "Formal linear family preset that routes linear family specs through the shared family router surface."
    elif trainer_key_norm == NEURAL_FORMAL_PRESET_KEY:
        summary = "Formal neural family preset that routes backend/runtime variants through the shared family router surface."
    elif trainer_key_norm == TREE_BOOSTING_FORMAL_PRESET_KEY:
        summary = "Formal tree_boosting family preset that routes boosting backend variants through the shared family router surface."
    symbolic_artifact_descriptor = (
        _symbolic_artifact_descriptor_from_head_profiles(head_profiles) if family == "symbolic" else {}
    )
    canonical_preset = (
        canonical_symbolic_preset_key(trainer_key)
        if family == "symbolic"
        else str(trainer_key)
    )
    legacy_facade_presets = tuple(SYMBOLIC_LEGACY_PRESET_KEYS) if trainer_key_norm == SYMBOLIC_FORMAL_PRESET_KEY else ()
    family_route_rows = _family_routes_for_trainer(trainer_key, family)
    family_route_keys = tuple(
        str(row.get("route_key", "")).strip()
        for row in family_route_rows
        if str(row.get("route_key", "")).strip()
    )
    family_route_match_fields = tuple(
        sorted(
            {
                str(field_name).strip()
                for row in family_route_rows
                for field_name in dict(row.get("match_fields", {})).keys()
                if str(field_name).strip()
            }
        )
    )
    family_route_statuses = tuple(
        sorted(
            {
                str(row.get("status", "")).strip()
                for row in family_route_rows
                if str(row.get("status", "")).strip()
            }
        )
    )
    family_route_formal_preset = _formal_preset_for_family(family)
    symbolic_route_rows = _symbolic_routes_for_trainer(trainer_key) if family == "symbolic" else ()
    symbolic_route_keys = tuple(
        str(row.get("route_key", "")).strip()
        for row in symbolic_route_rows
        if str(row.get("route_key", "")).strip()
    )
    symbolic_route_backends = tuple(
        sorted(
            {
                str(row.get("parameter_backend", "")).strip()
                for row in symbolic_route_rows
                if str(row.get("parameter_backend", "")).strip()
            }
        )
    )
    symbolic_route_tasks = tuple(
        sorted(
            {
                str(row.get("task", "")).strip()
                for row in symbolic_route_rows
                if str(row.get("task", "")).strip()
            }
        )
    )
    symbolic_route_structure_modes = tuple(
        sorted(
            {
                str(value).strip()
                for row in symbolic_route_rows
                for value in tuple(row.get("structure_modes", ()))
                if str(value).strip()
            }
        )
    )
    symbolic_route_statuses = tuple(
        sorted(
            {
                str(row.get("status", "")).strip()
                for row in symbolic_route_rows
                if str(row.get("status", "")).strip()
            }
        )
    )

    fields = {
        "id": f"preset:{trainer_key}",
        "preset": str(trainer_key),
        "trainer": str(trainer_key),
        "family": family,
        "head": default_head,
        "heads": heads,
        "head_profiles": head_profiles,
        "objective_family": objective_family,
        "outputs": outputs,
        "backend": str(backend_fields["backend"]),
        "runtime_backend": str(backend_fields["runtime_backend"]),
        "parameter_backend": str(backend_fields["parameter_backend"]),
        "supports_resume": bool(backend_fields["supports_resume"]),
        "supports_warm_start": bool(backend_fields["supports_warm_start"]),
        "supports_incremental": bool(backend_fields["supports_incremental"]),
        "preset_kind": str(backend_fields["preset_kind"]),
        "status": status,
        "surface_status": surface_status,
        "deprecated_surface": surface_status == "deprecated",
        "canonical_preset": canonical_preset,
        "migration_target": SYMBOLIC_FORMAL_PRESET_KEY if is_legacy_symbolic_preset(trainer_key) else "",
        "legacy_facade_presets": legacy_facade_presets,
        "family_route_count": len(family_route_rows),
        "family_route_keys": family_route_keys,
        "family_route_match_fields": family_route_match_fields,
        "family_route_statuses": family_route_statuses,
        "family_route_formal_preset": family_route_formal_preset,
        "symbolic_route_count": len(symbolic_route_rows),
        "symbolic_route_keys": symbolic_route_keys,
        "symbolic_route_backends": symbolic_route_backends,
        "symbolic_route_tasks": symbolic_route_tasks,
        "symbolic_route_structure_modes": symbolic_route_structure_modes,
        "symbolic_route_statuses": symbolic_route_statuses,
        "mechanism_keys": mechanism_keys,
        "mechanism_kinds": mechanism_kinds,
        "search_mechanism_keys": search_mechanism_keys,
        "search_mechanism_kinds": search_mechanism_kinds,
        "search_checkpointable_mechanisms": search_checkpointable_mechanisms,
        "search_replayable_mechanisms": search_replayable_mechanisms,
        "search_family_signature_mechanisms": search_family_signature_mechanisms,
        "artifact_schema_key": str(symbolic_artifact_descriptor.get("schema_key", "")),
        "artifact_schema_version": int(symbolic_artifact_descriptor.get("schema_version", 0) or 0),
        "artifact_schema_fields": tuple(symbolic_artifact_descriptor.get("artifact_schema_fields", ())),
        "artifact_complexity_fields": tuple(symbolic_artifact_descriptor.get("complexity_fields", ())),
        "artifact_explainability_fields": tuple(symbolic_artifact_descriptor.get("explainability_fields", ())),
        "artifact_stability_fields": tuple(symbolic_artifact_descriptor.get("stability_fields", ())),
        "artifact_schema_heads": tuple(symbolic_artifact_descriptor.get("heads", ())),
        "artifact_supports_piecewise": bool(symbolic_artifact_descriptor.get("supports_piecewise", False)),
        "field_version": 1,
        **build_entry_i18n_fields(
            kind="preset",
            key=f"preset:{trainer_key}",
            name=str(trainer_key),
            summary=summary,
            metadata=metadata,
        ),
    }
    relations = {
        "family": (f"family:{family}",),
        "heads": tuple(f"head:{head}" for head in heads),
    }
    if family_route_keys:
        relations["router_targets"] = tuple(f"preset:{key}" for key in family_route_keys)
    if family_route_formal_preset:
        relations["formal_preset"] = (f"preset:{family_route_formal_preset}",)
    if family == "symbolic":
        relations["canonical_preset"] = (f"preset:{canonical_preset}",)
    if legacy_facade_presets:
        relations["legacy_facades"] = tuple(f"preset:{key}" for key in legacy_facade_presets)
    if is_legacy_symbolic_preset(trainer_key):
        relations["migration_target"] = (f"preset:{SYMBOLIC_FORMAL_PRESET_KEY}",)

    return _PresetRow(
        key=str(trainer_key),
        family=family,
        default_head=default_head,
        heads=heads,
        objective_family=objective_family,
        outputs=outputs,
        head_profiles=head_profiles,
        backend=str(backend_fields["backend"]),
        runtime_backend=str(backend_fields["runtime_backend"]),
        parameter_backend=str(backend_fields["parameter_backend"]),
        supports_resume=bool(backend_fields["supports_resume"]),
        supports_warm_start=bool(backend_fields["supports_warm_start"]),
        supports_incremental=bool(backend_fields["supports_incremental"]),
        preset_kind=str(backend_fields["preset_kind"]),
        mechanism_keys=mechanism_keys,
        mechanism_kinds=mechanism_kinds,
        summary=summary,
        metadata=dict(metadata),
        fields=fields,
        relations=relations,
    )


@lru_cache(maxsize=1)
def _preset_rows() -> tuple[_PresetRow, ...]:
    cfg = create_default_config()
    rows: list[_PresetRow] = []
    for item in cfg.trainers.describe():
        key = str(item["key"])
        metadata = dict(item.get("metadata", {}))
        rows.append(_build_preset_row(key, metadata))
    rows.sort(key=lambda row: row.key)
    return tuple(rows)


@lru_cache(maxsize=1)
def _all_family_keys() -> tuple[str, ...]:
    return tuple(sorted({row.family for row in _preset_rows()}))


@lru_cache(maxsize=1)
def _all_preset_keys() -> tuple[str, ...]:
    return tuple(sorted(row.key for row in _preset_rows()))


def _families_for_preset_keys(preset_keys: Sequence[str]) -> tuple[str, ...]:
    allowed = {str(key).strip().lower() for key in tuple(preset_keys) if str(key).strip()}
    return tuple(sorted({row.family for row in _preset_rows() if row.key.lower() in allowed}))


def _flow_hook_events(capability: Any) -> tuple[str, ...]:
    events = (
        "on_flow_start",
        "on_data_ready",
        "on_pre_fit",
        "on_post_fit",
        "on_pre_eval",
        "on_post_eval",
        "on_pre_persist",
        "on_post_persist",
        "on_flow_finish",
        "on_flow_error",
        "on_experiment_start",
        "on_stage_start",
        "on_stage_end",
        "on_stage_error",
        "on_experiment_finish",
        "on_experiment_error",
    )
    implemented: list[str] = []
    cls = capability.__class__
    for event_name in events:
        candidate = getattr(cls, event_name, None)
        if candidate is None:
            continue
        base = getattr(FlowCapability, event_name, None)
        if base is None or candidate is not base:
            implemented.append(str(event_name))
    return tuple(implemented)


def _flow_lifecycle_plane(hook_events: Sequence[str]) -> str:
    event_names = tuple(str(name) for name in hook_events)
    has_flow = any(name.startswith("on_flow") or name.startswith("on_pre_") or name.startswith("on_post_") for name in event_names)
    has_experiment = any(name.startswith("on_experiment") or name.startswith("on_stage") for name in event_names)
    if has_flow and has_experiment:
        return "hybrid"
    if has_experiment:
        return "experiment"
    return "flow"


def _component_mount_contract(
    *,
    component_surface: str,
    component_kind: str,
    signal_names: Sequence[str] = tuple(),
    provides_fields: Sequence[str] = tuple(),
) -> dict[str, Any]:
    surface = str(component_surface).strip().lower()
    kind = str(component_kind).strip().lower()
    if surface == "bias":
        return {
            "mount_plane": "trainer",
            "mount_point": "bias_stack",
            "orchestration_phases": ("pre_fit", "fit", "predict"),
            "contract_consumes": tuple(),
            "contract_provides": tuple(),
            "contract_mutates": tuple(),
        }
    phase_map = {
        "state_signal_view": ("pre_fit", "fit", "post_fit"),
        "sample_weighting": ("pre_fit", "fit"),
        "sampling": ("pre_fit", "fit"),
        "aggregation": ("post_fit", "post_eval"),
    }
    return {
        "mount_plane": "trainer",
        "mount_point": "runtime_mechanism_stack",
        "orchestration_phases": tuple(phase_map.get(kind, ("fit",))),
        "contract_consumes": tuple(str(v) for v in tuple(signal_names)),
        "contract_provides": tuple(str(v) for v in tuple(provides_fields)),
        "contract_mutates": tuple(str(v) for v in tuple(provides_fields)),
    }


def _provider_mount_contract(info: Mapping[str, Any]) -> dict[str, Any]:
    plane = str(info.get("plane", "runtime")).strip().lower()
    phases: list[str] = []
    if bool(info.get("supports_individual", False)):
        phases.append("evaluate_individual")
    if bool(info.get("supports_batch", False)):
        phases.append("evaluate_population")
    consumes: list[str] = []
    if bool(info.get("supports_individual", False)):
        consumes.append("decision_vector")
    if bool(info.get("supports_batch", False)):
        consumes.append("decision_batch")
    return {
        "mount_plane": plane,
        "mount_point": f"{plane}.provider",
        "orchestration_phases": tuple(phases),
        "contract_consumes": tuple(consumes),
        "contract_provides": ("evaluation_result",),
        "contract_mutates": tuple(),
    }


@lru_cache(maxsize=1)
def _component_entries() -> tuple[CatalogEntry, ...]:
    cfg = create_default_config()
    out: list[CatalogEntry] = []

    all_preset_keys = _all_preset_keys()
    all_family_keys = _all_family_keys()
    for bias_item in cfg.biases.describe():
        bias_key = str(bias_item["key"])
        metadata = dict(bias_item.get("metadata", {}))
        component_key = f"component:bias.{bias_key}"
        mount_contract = _component_mount_contract(component_surface="bias", component_kind="bias")
        out.append(
            CatalogEntry(
                key=component_key,
                kind="component",
                name=bias_key,
                source="registry",
                tags=("component", "bias", "framework"),
                summary=str(metadata.get("purpose", "") or f"Bias component '{bias_key}'."),
                metadata=metadata,
                fields={
                    "id": component_key,
                    "component": bias_key,
                    "component_surface": "bias",
                    "component_kind": "bias",
                    "applicable_families": all_family_keys,
                    "applicable_presets": all_preset_keys,
                    **mount_contract,
                    "params": tuple(sorted(str(key) for key in dict(metadata.get("params", {})).keys())),
                    "status": "stable",
                    "field_version": 1,
                    **build_entry_i18n_fields(
                        kind="component",
                        key=component_key,
                        name=bias_key,
                        summary=str(metadata.get("purpose", "") or f"Bias component '{bias_key}'."),
                        metadata=metadata,
                    ),
                },
                relations={
                    "families": tuple(f"family:{family_key}" for family_key in all_family_keys),
                    "presets": tuple(f"preset:{preset_key}" for preset_key in all_preset_keys),
                    "legacy_bias_entry": (f"bias:{bias_key}",),
                },
            )
        )

    for mechanism_key, info in sorted(_RUNTIME_MECHANISM_COMPONENT_INFO.items(), key=lambda kv: kv[0]):
        preset_keys = tuple(str(v) for v in tuple(info.get("applicable_presets", ())) if str(v).strip())
        family_keys = _families_for_preset_keys(preset_keys)
        component_key = f"component:{mechanism_key}"
        mount_contract = _component_mount_contract(
            component_surface="runtime_mechanism",
            component_kind=str(info.get("component_kind", "component")),
            signal_names=tuple(info.get("signal_names", ())),
            provides_fields=tuple(info.get("provides_fields", ())),
        )
        out.append(
            CatalogEntry(
                key=component_key,
                kind="component",
                name=str(mechanism_key),
                source="derived_registry",
                path=str(info.get("path", "core/mechanisms/runtime.py")),
                tags=("component", "runtime_mechanism", str(info.get("component_kind", "component"))),
                summary=str(info.get("summary", f"Runtime mechanism component '{mechanism_key}'.")),
                fields={
                    "id": component_key,
                    "component": str(mechanism_key),
                    "component_surface": "runtime_mechanism",
                    "component_kind": str(info.get("component_kind", "component")),
                    "applicable_families": family_keys,
                    "applicable_presets": preset_keys,
                    **mount_contract,
                    "binding_level": str(
                        info.get(
                            "binding_level",
                            "optional" if str(info.get("component_kind", "")) != "aggregation" else "bound",
                        )
                    ),
                    "signal_names": tuple(str(v) for v in tuple(info.get("signal_names", ()))),
                    "provides_fields": tuple(str(v) for v in tuple(info.get("provides_fields", ()))),
                    "module": str(info.get("module", "")),
                    "status": str(info.get("status", "stable")),
                    "field_version": 1,
                    **build_entry_i18n_fields(
                        kind="component",
                        key=component_key,
                        name=str(mechanism_key),
                        summary=str(info.get("summary", f"Runtime mechanism component '{mechanism_key}'.")),
                        metadata=info,
                    ),
                },
                relations={
                    "families": tuple(f"family:{family_key}" for family_key in family_keys),
                    "presets": tuple(f"preset:{preset_key}" for preset_key in preset_keys),
                },
            )
        )
    return tuple(out)


@lru_cache(maxsize=1)
def _provider_entries() -> tuple[CatalogEntry, ...]:
    out: list[CatalogEntry] = []
    all_preset_keys = _all_preset_keys()
    all_family_keys = _all_family_keys()
    for provider_key, info in sorted(_PROVIDER_INFO.items(), key=lambda kv: kv[0]):
        catalog_key = f"provider:{provider_key}"
        mount_contract = _provider_mount_contract(info)
        out.append(
            CatalogEntry(
                key=catalog_key,
                kind="provider",
                name=str(provider_key),
                source="filesystem",
                path=str(info.get("path")) if info.get("path") else None,
                tags=("provider", str(info.get("provider_surface", "provider")), "framework"),
                summary=str(info.get("summary", f"Provider '{provider_key}'.")),
                fields={
                    "id": catalog_key,
                    "provider": str(provider_key),
                    "provider_surface": str(info.get("provider_surface", "provider")),
                    "plane": str(info.get("plane", "runtime")),
                    "applicable_families": all_family_keys,
                    "applicable_presets": all_preset_keys,
                    **mount_contract,
                    "supports_batch": bool(info.get("supports_batch", False)),
                    "supports_individual": bool(info.get("supports_individual", False)),
                    "module": str(info.get("module", "")),
                    "status": str(info.get("status", "stable")),
                    "field_version": 1,
                    **build_entry_i18n_fields(
                        kind="provider",
                        key=catalog_key,
                        name=str(provider_key),
                        summary=str(info.get("summary", f"Provider '{provider_key}'.")),
                        metadata=info,
                    ),
                },
                relations={
                    "families": tuple(f"family:{family_key}" for family_key in all_family_keys),
                    "presets": tuple(f"preset:{preset_key}" for preset_key in all_preset_keys),
                },
            )
        )
    return tuple(out)


@lru_cache(maxsize=1)
def _plugin_entries() -> tuple[CatalogEntry, ...]:
    cfg = create_default_config()
    out: list[CatalogEntry] = []
    all_preset_keys = _all_preset_keys()
    all_family_keys = _all_family_keys()

    for capability_item in cfg.capabilities.describe():
        capability_key = str(capability_item["key"])
        metadata = dict(capability_item.get("metadata", {}))
        capability_kwargs: dict[str, Any] = {}
        if capability_key == "metric_guard":
            capability_kwargs["threshold"] = 1.0
        capability = cfg.capabilities.create(capability_key, **capability_kwargs)
        hook_events = _flow_hook_events(capability)
        contract = capability.get_context_contract()
        catalog_key = f"plugin:{capability_key}"
        out.append(
            CatalogEntry(
                key=catalog_key,
                kind="plugin",
                name=str(capability_key),
                source="registry",
                tags=("plugin", "capability", "framework"),
                summary=str(metadata.get("purpose", "") or capability.context_notes or f"Plugin '{capability_key}'."),
                metadata=metadata,
                fields={
                    "id": catalog_key,
                    "plugin": str(capability_key),
                    "plugin_surface": "capability_registry",
                    "applicable_families": all_family_keys,
                    "applicable_presets": all_preset_keys,
                    "lifecycle_plane": _flow_lifecycle_plane(hook_events),
                    "mount_plane": _flow_lifecycle_plane(hook_events),
                    "mount_point": "capability_registry",
                    "orchestration_phases": hook_events,
                    "hook_events": hook_events,
                    "priority": int(getattr(capability, "priority", 0)),
                    "enabled_by_default": bool(getattr(capability, "enabled", True)),
                    "is_algorithmic": bool(getattr(capability, "is_algorithmic", False)),
                    "contract_requires": tuple(contract.get("requires", ())),
                    "contract_consumes": tuple(contract.get("requires", ())),
                    "contract_provides": tuple(contract.get("provides", ())),
                    "contract_mutates": tuple(contract.get("mutates", ())),
                    "contract_cache": tuple(contract.get("cache", ())),
                    "context_requires": tuple(contract.get("requires", ())),
                    "context_provides": tuple(contract.get("provides", ())),
                    "context_mutates": tuple(contract.get("mutates", ())),
                    "context_cache": tuple(contract.get("cache", ())),
                    "status": "stable",
                    "field_version": 1,
                    **build_entry_i18n_fields(
                        kind="plugin",
                        key=catalog_key,
                        name=str(capability_key),
                        summary=str(metadata.get("purpose", "") or capability.context_notes or f"Plugin '{capability_key}'."),
                        metadata=metadata,
                    ),
                },
                relations={
                    "families": tuple(f"family:{family_key}" for family_key in all_family_keys),
                    "presets": tuple(f"preset:{preset_key}" for preset_key in all_preset_keys),
                },
            )
        )

    for plugin_name, raw_info in sorted(_PLUGIN_CLASS_INFO.items(), key=lambda kv: kv[0]):
        info = dict(raw_info)
        hook_events = tuple(str(v).strip() for v in str(info.get("hook_events", "")).split(",") if str(v).strip())
        context_requires = tuple(str(v).strip() for v in str(info.get("context_requires", "")).split(",") if str(v).strip())
        context_provides = tuple(str(v).strip() for v in str(info.get("context_provides", "")).split(",") if str(v).strip())
        context_mutates = tuple(str(v).strip() for v in str(info.get("context_mutates", "")).split(",") if str(v).strip())
        context_cache = tuple(str(v).strip() for v in str(info.get("context_cache", "")).split(",") if str(v).strip())
        catalog_key = f"plugin:{plugin_name}"
        out.append(
            CatalogEntry(
                key=catalog_key,
                kind="plugin",
                name=str(plugin_name),
                source="filesystem",
                path=str(info.get("path")) if info.get("path") else None,
                tags=("plugin", str(info.get("surface", "flow_plugin")), "framework"),
                summary=str(info.get("summary", f"Plugin '{plugin_name}'.")),
                fields={
                    "id": catalog_key,
                    "plugin": str(plugin_name),
                    "plugin_surface": str(info.get("surface", "flow_plugin")),
                    "applicable_families": all_family_keys,
                    "applicable_presets": all_preset_keys,
                    "lifecycle_plane": str(info.get("lifecycle_plane", _flow_lifecycle_plane(hook_events))),
                    "mount_plane": str(info.get("lifecycle_plane", _flow_lifecycle_plane(hook_events))),
                    "mount_point": str(info.get("surface", "flow_plugin")),
                    "orchestration_phases": hook_events,
                    "hook_events": hook_events,
                    "priority": int(str(info.get("priority", "0"))),
                    "enabled_by_default": str(info.get("enabled_by_default", "true")).strip().lower() == "true",
                    "is_algorithmic": str(info.get("is_algorithmic", "false")).strip().lower() == "true",
                    "contract_requires": context_requires,
                    "contract_consumes": context_requires,
                    "contract_provides": context_provides,
                    "contract_mutates": context_mutates,
                    "contract_cache": context_cache,
                    "context_requires": context_requires,
                    "context_provides": context_provides,
                    "context_mutates": context_mutates,
                    "context_cache": context_cache,
                    "status": "stable",
                    "field_version": 1,
                    **build_entry_i18n_fields(
                        kind="plugin",
                        key=catalog_key,
                        name=str(plugin_name),
                        summary=str(info.get("summary", f"Plugin '{plugin_name}'.")),
                        metadata=info,
                    ),
                },
                relations={
                    "families": tuple(f"family:{family_key}" for family_key in all_family_keys),
                    "presets": tuple(f"preset:{preset_key}" for preset_key in all_preset_keys),
                },
            )
        )
    return tuple(out)


def _preset_component_entries(trainer_key: str) -> tuple[CatalogEntry, ...]:
    target = str(trainer_key).strip().lower()
    family = _canonical_family(target, {})
    route_targets = {
        str(row.get("route_key", "")).strip().lower()
        for row in _family_routes_for_trainer(target, family)
        if str(row.get("route_key", "")).strip()
    }
    allowed_targets = {target, *route_targets}
    return tuple(
        entry
        for entry in _component_entries()
        if allowed_targets.intersection(
            {str(value).strip().lower() for value in tuple(dict(entry.fields).get("applicable_presets", ()))}
        )
    )


def _preset_provider_entries(_trainer_key: str) -> tuple[CatalogEntry, ...]:
    target = str(_trainer_key).strip().lower()
    family = _canonical_family(target, {})
    route_targets = {
        str(row.get("route_key", "")).strip().lower()
        for row in _family_routes_for_trainer(target, family)
        if str(row.get("route_key", "")).strip()
    }
    allowed_targets = {target, *route_targets}
    return tuple(
        entry
        for entry in _provider_entries()
        if allowed_targets.intersection(
            {str(value).strip().lower() for value in tuple(dict(entry.fields).get("applicable_presets", ()))}
        )
    )


def _preset_plugin_entries(_trainer_key: str) -> tuple[CatalogEntry, ...]:
    target = str(_trainer_key).strip().lower()
    family = _canonical_family(target, {})
    route_targets = {
        str(row.get("route_key", "")).strip().lower()
        for row in _family_routes_for_trainer(target, family)
        if str(row.get("route_key", "")).strip()
    }
    allowed_targets = {target, *route_targets}
    return tuple(
        entry
        for entry in _plugin_entries()
        if allowed_targets.intersection(
            {str(value).strip().lower() for value in tuple(dict(entry.fields).get("applicable_presets", ()))}
        )
    )


def _preset_stack_payload(trainer_key: str) -> tuple[dict[str, Any], dict[str, tuple[str, ...]]]:
    component_entries = _preset_component_entries(trainer_key)
    provider_entries = _preset_provider_entries(trainer_key)
    plugin_entries = _preset_plugin_entries(trainer_key)

    component_keys = tuple(entry.key for entry in component_entries)
    provider_keys = tuple(entry.key for entry in provider_entries)
    plugin_keys = tuple(entry.key for entry in plugin_entries)

    fields = {
        "components": tuple(entry.name for entry in component_entries),
        "component_surfaces": tuple(sorted({str(dict(entry.fields).get("component_surface", "")) for entry in component_entries if str(dict(entry.fields).get("component_surface", "")).strip()})),
        "component_count": int(len(component_entries)),
        "providers": tuple(entry.name for entry in provider_entries),
        "provider_planes": tuple(sorted({str(dict(entry.fields).get("plane", "")) for entry in provider_entries if str(dict(entry.fields).get("plane", "")).strip()})),
        "provider_count": int(len(provider_entries)),
        "plugins": tuple(entry.name for entry in plugin_entries),
        "plugin_surfaces": tuple(sorted({str(dict(entry.fields).get("plugin_surface", "")) for entry in plugin_entries if str(dict(entry.fields).get("plugin_surface", "")).strip()})),
        "plugin_count": int(len(plugin_entries)),
    }
    relations = {
        "components": component_keys,
        "providers": provider_keys,
        "plugins": plugin_keys,
    }
    return fields, relations


def _registry_entries() -> list[CatalogEntry]:
    cfg = create_default_config()
    out: list[CatalogEntry] = []

    kind_map = {
        "pipelines": "pipeline",
        "biases": "bias",
        "numericizers": "numericizer",
    }
    for reg_name, kind in kind_map.items():
        reg = getattr(cfg, reg_name)
        for item in reg.describe():
            key = str(item["key"])
            md = dict(item.get("metadata", {}))
            relations: dict[str, tuple[str, ...]] = {}
            if kind == "bias":
                relations["canonical_component"] = (f"component:bias.{key}",)
            out.append(
                CatalogEntry(
                    key=f"{kind}:{key}",
                    kind=kind,
                    name=key,
                    source="registry",
                    metadata=md,
                    tags=(kind, "framework"),
                    summary=str(md.get("purpose", "") or f"Registered {kind} '{key}'."),
                    fields={
                        "id": f"{kind}:{key}",
                        "name": key,
                        "kind": kind,
                        "field_version": 1,
                        **build_entry_i18n_fields(
                            kind=kind,
                            key=f"{kind}:{key}",
                            name=key,
                            summary=str(md.get("purpose", "") or f"Registered {kind} '{key}'."),
                            metadata=md,
                        ),
                    },
                    relations=relations,
                )
            )

    for row in _preset_rows():
        trainer_key = row.key
        trainer_md = dict(row.metadata)
        stack_fields, stack_relations = _preset_stack_payload(trainer_key)
        trainer_tags = ("trainer", "framework", f"family:{row.family}", f"head:{row.default_head}")
        trainer_relations = {
            "canonical_preset": (f"preset:{trainer_key}",),
            **dict(row.relations),
            **stack_relations,
        }
        trainer_fields = {
            **dict(row.fields),
            **stack_fields,
            "id": f"trainer:{trainer_key}",
            "canonical_kind": "preset",
            **build_entry_i18n_fields(
                kind="trainer",
                key=f"trainer:{trainer_key}",
                name=trainer_key,
                summary=row.summary,
                metadata=trainer_md,
            ),
        }
        out.append(
            CatalogEntry(
                key=f"trainer:{trainer_key}",
                kind="trainer",
                name=trainer_key,
                source="registry",
                metadata=trainer_md,
                tags=trainer_tags,
                summary=row.summary,
                fields=trainer_fields,
                relations=trainer_relations,
            )
        )
        out.append(
            CatalogEntry(
                key=f"preset:{trainer_key}",
                kind="preset",
                name=trainer_key,
                source="registry",
                metadata=trainer_md,
                tags=("preset", "framework", f"family:{row.family}", f"head:{row.default_head}"),
                summary=row.summary,
                fields={**dict(row.fields), **stack_fields},
                relations={
                    **dict(row.relations),
                    **stack_relations,
                    "legacy_trainer_entry": (f"trainer:{trainer_key}",),
                },
            )
        )

    out.extend(_component_entries())
    out.extend(_provider_entries())
    out.extend(_plugin_entries())
    out.extend(_family_entries())
    out.extend(_head_entries())
    return out


def _family_entries() -> list[CatalogEntry]:
    grouped: dict[str, list[_PresetRow]] = {}
    for row in _preset_rows():
        grouped.setdefault(row.family, []).append(row)

    out: list[CatalogEntry] = []
    for family_key, members in sorted(grouped.items(), key=lambda kv: kv[0]):
        info = dict(_CANONICAL_FAMILY_INFO.get(family_key, {}))
        heads = tuple(sorted({head for row in members for head in row.heads}))
        presets = tuple(f"preset:{row.key}" for row in members)
        runtime_backends = tuple(sorted({row.runtime_backend or row.backend for row in members if row.runtime_backend or row.backend}))
        parameter_backends = tuple(sorted({row.parameter_backend for row in members if row.parameter_backend}))
        artifact_schema_fields = tuple(
            sorted(
                {
                    str(value)
                    for row in members
                    for value in tuple(dict(row.fields).get("artifact_schema_fields", ()))
                    if str(value).strip()
                }
            )
        )
        artifact_complexity_fields = tuple(
            sorted(
                {
                    str(value)
                    for row in members
                    for value in tuple(dict(row.fields).get("artifact_complexity_fields", ()))
                    if str(value).strip()
                }
            )
        )
        artifact_explainability_fields = tuple(
            sorted(
                {
                    str(value)
                    for row in members
                    for value in tuple(dict(row.fields).get("artifact_explainability_fields", ()))
                    if str(value).strip()
                }
            )
        )
        artifact_stability_fields = tuple(
            sorted(
                {
                    str(value)
                    for row in members
                    for value in tuple(dict(row.fields).get("artifact_stability_fields", ()))
                    if str(value).strip()
                }
            )
        )
        search_mechanism_keys = tuple(
            sorted(
                {
                    str(value)
                    for row in members
                    for value in tuple(dict(row.fields).get("search_mechanism_keys", ()))
                    if str(value).strip()
                }
            )
        )
        search_mechanism_kinds = tuple(
            sorted(
                {
                    str(value)
                    for row in members
                    for value in tuple(dict(row.fields).get("search_mechanism_kinds", ()))
                    if str(value).strip()
                }
            )
        )
        search_checkpointable_mechanisms = tuple(
            sorted(
                {
                    str(value)
                    for row in members
                    for value in tuple(dict(row.fields).get("search_checkpointable_mechanisms", ()))
                    if str(value).strip()
                }
            )
        )
        search_replayable_mechanisms = tuple(
            sorted(
                {
                    str(value)
                    for row in members
                    for value in tuple(dict(row.fields).get("search_replayable_mechanisms", ()))
                    if str(value).strip()
                }
            )
        )
        search_family_signature_mechanisms = tuple(
            sorted(
                {
                    str(value)
                    for row in members
                    for value in tuple(dict(row.fields).get("search_family_signature_mechanisms", ()))
                    if str(value).strip()
                }
            )
        )
        family_route_keys = tuple(
            sorted(
                {
                    str(value)
                    for row in members
                    for value in tuple(dict(row.fields).get("family_route_keys", ()))
                    if str(value).strip()
                }
            )
        )
        family_route_match_fields = tuple(
            sorted(
                {
                    str(value)
                    for row in members
                    for value in tuple(dict(row.fields).get("family_route_match_fields", ()))
                    if str(value).strip()
                }
            )
        )
        family_route_statuses = tuple(
            sorted(
                {
                    str(value)
                    for row in members
                    for value in tuple(dict(row.fields).get("family_route_statuses", ()))
                    if str(value).strip()
                }
            )
        )
        symbolic_route_keys = tuple(
            sorted(
                {
                    str(value)
                    for row in members
                    for value in tuple(dict(row.fields).get("symbolic_route_keys", ()))
                    if str(value).strip()
                }
            )
        )
        symbolic_route_backends = tuple(
            sorted(
                {
                    str(value)
                    for row in members
                    for value in tuple(dict(row.fields).get("symbolic_route_backends", ()))
                    if str(value).strip()
                }
            )
        )
        symbolic_route_tasks = tuple(
            sorted(
                {
                    str(value)
                    for row in members
                    for value in tuple(dict(row.fields).get("symbolic_route_tasks", ()))
                    if str(value).strip()
                }
            )
        )
        symbolic_route_structure_modes = tuple(
            sorted(
                {
                    str(value)
                    for row in members
                    for value in tuple(dict(row.fields).get("symbolic_route_structure_modes", ()))
                    if str(value).strip()
                }
            )
        )
        artifact_schema_heads = tuple(
            sorted(
                {
                    str(value)
                    for row in members
                    for value in tuple(dict(row.fields).get("artifact_schema_heads", ()))
                    if str(value).strip()
                }
            )
        )
        fields = {
            "id": f"family:{family_key}",
            "family": family_key,
            "heads": heads,
            "presets": tuple(row.key for row in members),
            "components": tuple(
                entry.name
                for entry in _component_entries()
                if family_key in tuple(dict(entry.fields).get("applicable_families", ()))
            ),
            "runtime_backends": runtime_backends,
            "parameter_backends": parameter_backends,
            "supports_resume": any(row.supports_resume for row in members),
            "supports_warm_start": any(row.supports_warm_start for row in members),
            "supports_incremental": any(row.supports_incremental for row in members),
            "preset_count": len(members),
            "head_count": len(heads),
            "family_route_count": len(family_route_keys),
            "family_route_keys": family_route_keys,
            "family_route_match_fields": family_route_match_fields,
            "family_route_statuses": family_route_statuses,
            "family_route_formal_preset": _formal_preset_for_family(family_key),
            "symbolic_route_count": len(symbolic_route_keys),
            "symbolic_route_keys": symbolic_route_keys,
            "symbolic_route_backends": symbolic_route_backends,
            "symbolic_route_tasks": symbolic_route_tasks,
            "symbolic_route_structure_modes": symbolic_route_structure_modes,
            "artifact_schema_key": "symbolic_artifact_v1" if family_key == "symbolic" and artifact_schema_fields else "",
            "artifact_schema_version": 1 if family_key == "symbolic" and artifact_schema_fields else 0,
            "artifact_schema_fields": artifact_schema_fields,
            "artifact_complexity_fields": artifact_complexity_fields,
            "artifact_explainability_fields": artifact_explainability_fields,
            "artifact_stability_fields": artifact_stability_fields,
            "artifact_schema_heads": artifact_schema_heads,
            "artifact_supports_piecewise": any(
                bool(dict(row.fields).get("artifact_supports_piecewise", False)) for row in members
            ),
            "search_mechanism_keys": search_mechanism_keys,
            "search_mechanism_kinds": search_mechanism_kinds,
            "search_checkpointable_mechanisms": search_checkpointable_mechanisms,
            "search_replayable_mechanisms": search_replayable_mechanisms,
            "search_family_signature_mechanisms": search_family_signature_mechanisms,
            "field_version": 1,
            **build_entry_i18n_fields(
                kind="family",
                key=f"family:{family_key}",
                name=str(info.get("name", family_key) or family_key),
                summary=str(info.get("summary", f"Family '{family_key}' derived from registered presets.")),
                metadata=info,
            ),
        }
        out.append(
            CatalogEntry(
                key=f"family:{family_key}",
                kind="family",
                name=str(info.get("name", family_key) or family_key),
                source="derived_registry",
                tags=("family", "framework"),
                summary=str(info.get("summary", f"Family '{family_key}' derived from registered presets.")),
                fields=fields,
                relations={
                    "presets": presets,
                    "heads": tuple(f"head:{head}" for head in heads),
                    "components": tuple(
                        entry.key
                        for entry in _component_entries()
                        if family_key in tuple(dict(entry.fields).get("applicable_families", ()))
                    ),
                    "router_targets": tuple(f"preset:{key}" for key in family_route_keys),
                },
            )
        )
    return out


def _head_entries() -> list[CatalogEntry]:
    grouped: dict[str, list[_PresetRow]] = {}
    for row in _preset_rows():
        for head in row.heads:
            grouped.setdefault(head, []).append(row)

    out: list[CatalogEntry] = []
    for head_key, members in sorted(grouped.items(), key=lambda kv: kv[0]):
        info = dict(_HEAD_INFO.get(head_key, {}))
        families = tuple(sorted({row.family for row in members}))
        presets = tuple(sorted({row.key for row in members}))
        outputs = tuple(
            sorted(
                {
                    value
                    for row in members
                    for value in tuple(dict(row.head_profiles.get(head_key, {})).get("outputs", ()))
                }
            )
        )
        objective_families = tuple(
            sorted(
                {
                    str(dict(row.head_profiles.get(head_key, {})).get("objective_family", "")).strip().lower()
                    for row in members
                    if str(dict(row.head_profiles.get(head_key, {})).get("objective_family", "")).strip()
                }
            )
        )
        search_mechanism_keys = tuple(
            sorted(
                {
                    str(value)
                    for row in members
                    for value in tuple(dict(row.fields).get("search_mechanism_keys", ()))
                    if row.family == "symbolic" and str(value).strip()
                }
            )
        )
        search_mechanism_kinds = tuple(
            sorted(
                {
                    str(value)
                    for row in members
                    for value in tuple(dict(row.fields).get("search_mechanism_kinds", ()))
                    if row.family == "symbolic" and str(value).strip()
                }
            )
        )
        search_checkpointable_mechanisms = tuple(
            sorted(
                {
                    str(value)
                    for row in members
                    for value in tuple(dict(row.fields).get("search_checkpointable_mechanisms", ()))
                    if row.family == "symbolic" and str(value).strip()
                }
            )
        )
        search_replayable_mechanisms = tuple(
            sorted(
                {
                    str(value)
                    for row in members
                    for value in tuple(dict(row.fields).get("search_replayable_mechanisms", ()))
                    if row.family == "symbolic" and str(value).strip()
                }
            )
        )
        search_family_signature_mechanisms = tuple(
            sorted(
                {
                    str(value)
                    for row in members
                    for value in tuple(dict(row.fields).get("search_family_signature_mechanisms", ()))
                    if row.family == "symbolic" and str(value).strip()
                }
            )
        )
        family_route_keys = tuple(
            sorted(
                {
                    str(value)
                    for row in members
                    for value in tuple(dict(row.fields).get("family_route_keys", ()))
                    if str(value).strip()
                }
            )
        )
        family_route_match_fields = tuple(
            sorted(
                {
                    str(value)
                    for row in members
                    for value in tuple(dict(row.fields).get("family_route_match_fields", ()))
                    if str(value).strip()
                }
            )
        )
        family_route_statuses = tuple(
            sorted(
                {
                    str(value)
                    for row in members
                    for value in tuple(dict(row.fields).get("family_route_statuses", ()))
                    if str(value).strip()
                }
            )
        )
        symbolic_route_keys = tuple(
            sorted(
                {
                    str(value)
                    for row in members
                    for value in tuple(dict(row.fields).get("symbolic_route_keys", ()))
                    if row.family == "symbolic" and str(value).strip()
                }
            )
        )
        symbolic_route_backends = tuple(
            sorted(
                {
                    str(value)
                    for row in members
                    for value in tuple(dict(row.fields).get("symbolic_route_backends", ()))
                    if row.family == "symbolic" and str(value).strip()
                }
            )
        )
        symbolic_route_tasks = tuple(
            sorted(
                {
                    str(value)
                    for row in members
                    for value in tuple(dict(row.fields).get("symbolic_route_tasks", ()))
                    if row.family == "symbolic" and str(value).strip()
                }
            )
        )
        symbolic_route_structure_modes = tuple(
            sorted(
                {
                    str(value)
                    for row in members
                    for value in tuple(dict(row.fields).get("symbolic_route_structure_modes", ()))
                    if row.family == "symbolic" and str(value).strip()
                }
            )
        )
        symbolic_descriptor = (
            _symbolic_artifact_descriptor_from_head_profiles(
                {
                    str(head_key): dict(row.head_profiles.get(head_key, {}))
                    for row in members
                    if row.family == "symbolic" and head_key in dict(row.head_profiles)
                }
            )
            if any(row.family == "symbolic" for row in members)
            else {}
        )
        out.append(
            CatalogEntry(
                key=f"head:{head_key}",
                kind="head",
                name=head_key,
                source="derived_registry",
                tags=("head", "framework"),
                summary=str(info.get("summary", f"Head '{head_key}' derived from registered presets.")),
                fields={
                    "id": f"head:{head_key}",
                    "head": head_key,
                    "families": families,
                    "presets": presets,
                    "objective_families": objective_families,
                    "outputs": outputs or tuple(info.get("default_outputs", ())),
                    "family_count": len(families),
                    "preset_count": len(presets),
                    "family_route_count": len(family_route_keys),
                    "family_route_keys": family_route_keys,
                    "family_route_match_fields": family_route_match_fields,
                    "family_route_statuses": family_route_statuses,
                    "symbolic_route_count": len(symbolic_route_keys),
                    "symbolic_route_keys": symbolic_route_keys,
                    "symbolic_route_backends": symbolic_route_backends,
                    "symbolic_route_tasks": symbolic_route_tasks,
                    "symbolic_route_structure_modes": symbolic_route_structure_modes,
                    "artifact_schema_key": str(symbolic_descriptor.get("schema_key", "")),
                    "artifact_schema_version": int(symbolic_descriptor.get("schema_version", 0) or 0),
                    "artifact_schema_fields": tuple(symbolic_descriptor.get("artifact_schema_fields", ())),
                    "artifact_complexity_fields": tuple(symbolic_descriptor.get("complexity_fields", ())),
                    "artifact_explainability_fields": tuple(symbolic_descriptor.get("explainability_fields", ())),
                    "artifact_stability_fields": tuple(symbolic_descriptor.get("stability_fields", ())),
                    "artifact_supports_piecewise": bool(symbolic_descriptor.get("supports_piecewise", False)),
                    "search_mechanism_keys": search_mechanism_keys,
                    "search_mechanism_kinds": search_mechanism_kinds,
                    "search_checkpointable_mechanisms": search_checkpointable_mechanisms,
                    "search_replayable_mechanisms": search_replayable_mechanisms,
                    "search_family_signature_mechanisms": search_family_signature_mechanisms,
                    "field_version": 1,
                    **build_entry_i18n_fields(
                        kind="head",
                        key=f"head:{head_key}",
                        name=head_key,
                        summary=str(info.get("summary", f"Head '{head_key}' derived from registered presets.")),
                        metadata=info,
                    ),
                },
                relations={
                    "families": tuple(f"family:{family}" for family in families),
                    "presets": tuple(f"preset:{preset}" for preset in presets),
                    "router_targets": tuple(f"preset:{key}" for key in family_route_keys),
                },
            )
        )
    return out


def _doc_entries() -> list[CatalogEntry]:
    docs_root = ROOT / "docs"
    if not docs_root.exists():
        return []

    out: list[CatalogEntry] = []
    for path in sorted(docs_root.rglob("*.md")):
        rel = path.relative_to(ROOT).as_posix()
        out.append(
            CatalogEntry(
                key=f"doc:{rel}",
                kind="doc",
                name=path.stem,
                source="filesystem",
                path=rel,
                tags=("doc",),
                summary=f"Documentation page at {rel}",
                metadata={"ext": ".md"},
                fields={
                    "id": f"doc:{rel}",
                    "ext": ".md",
                    "field_version": 1,
                    **build_entry_i18n_fields(
                        kind="doc",
                        key=f"doc:{rel}",
                        name=path.stem,
                        summary=f"Documentation page at {rel}",
                        metadata={"path": rel},
                    ),
                },
            )
        )
    return out


def _example_entries() -> list[CatalogEntry]:
    ex_root = ROOT / "examples"
    if not ex_root.exists():
        return []

    out: list[CatalogEntry] = []
    for path in sorted(ex_root.glob("run_*.py")):
        rel = path.relative_to(ROOT).as_posix()
        out.append(
            CatalogEntry(
                key=f"example:{path.stem}",
                kind="example",
                name=path.stem,
                source="filesystem",
                path=rel,
                tags=("example", "script"),
                summary=f"Example script at {rel}",
                metadata={"ext": ".py"},
                fields={
                    "id": f"example:{path.stem}",
                    "ext": ".py",
                    "field_version": 1,
                    **build_entry_i18n_fields(
                        kind="example",
                        key=f"example:{path.stem}",
                        name=path.stem,
                        summary=f"Example script at {rel}",
                        metadata={"path": rel},
                    ),
                },
            )
        )
    return out


@lru_cache(maxsize=1)
def _all_entries() -> tuple[CatalogEntry, ...]:
    entries = [*_registry_entries(), *_doc_entries(), *_example_entries()]
    entries.sort(key=lambda x: (x.kind, x.key))
    return tuple(entries)


def _apply_profile(entries: Sequence[CatalogEntry], profile: str) -> tuple[CatalogEntry, ...]:
    p = _normalize_profile(profile)
    excluded = _PROFILE_EXCLUDES[p]
    if not excluded:
        return tuple(entries)
    return tuple(e for e in entries if e.kind not in excluded)


def _flatten_scalars(value: Any) -> Iterable[str]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value.strip().lower(),) if value.strip() else ()
    if isinstance(value, (bool, int, float)):
        return (str(value).strip().lower(),)
    if isinstance(value, Path):
        return (value.as_posix().strip().lower(),)
    if isinstance(value, Mapping):
        out: list[str] = []
        for k, v in value.items():
            out.extend(_flatten_scalars(k))
            out.extend(_flatten_scalars(v))
        return tuple(out)
    if isinstance(value, (tuple, list, set, frozenset)):
        out: list[str] = []
        for item in value:
            out.extend(_flatten_scalars(item))
        return tuple(out)
    return (str(value).strip().lower(),)


def _field_aliases(name: str) -> tuple[str, ...]:
    key = _normalize_field_name(name)
    aliases = _FIELD_ALIASES.get(key)
    if aliases is not None:
        return aliases
    if key.endswith("s") and key[:-1]:
        return (key, key[:-1])
    return (key,)


def _entry_base_fields(entry: CatalogEntry) -> dict[str, Any]:
    return {
        "id": entry.key,
        "key": entry.key,
        "kind": entry.kind,
        "name": entry.name,
        "source": entry.source,
        "path": entry.path,
        "tags": entry.tags,
        "summary": entry.summary,
    }


def _entry_field_values(entry: CatalogEntry, field_name: str, *, include_relations: bool = True) -> tuple[Any, ...]:
    values: list[Any] = []
    base_fields = _entry_base_fields(entry)
    for alias in _field_aliases(field_name):
        if alias in base_fields:
            values.append(base_fields[alias])
        if alias in entry.fields:
            values.append(entry.fields[alias])
        if include_relations and alias in entry.relations:
            values.append(entry.relations[alias])
    return tuple(values)


def _matches_field_filters(entry: CatalogEntry, field_filters: tuple[tuple[str, str], ...]) -> bool:
    if not field_filters:
        return True
    for field_name, expected in field_filters:
        values = _entry_field_values(entry, field_name)
        if not values:
            return False
        matched = False
        for value in values:
            flattened = _flatten_scalars(value)
            if expected in flattened:
                matched = True
                break
        if not matched:
            return False
    return True


def list_entries(
    *,
    profile: str = "default",
    kind: str | None = None,
    limit: int | None = None,
    field_filters: Mapping[str, Any] | Sequence[tuple[str, Any]] | None = None,
) -> tuple[CatalogEntry, ...]:
    entries = _apply_profile(_all_entries(), profile)
    k = _normalize_kind(kind)
    if k is not None:
        entries = tuple(e for e in entries if e.kind == k)

    filters = _normalize_field_filters(field_filters)
    if filters:
        entries = tuple(e for e in entries if _matches_field_filters(e, filters))

    if limit is not None:
        n = max(0, int(limit))
        entries = entries[:n]

    return tuple(entries)


def _search_text(entry: CatalogEntry) -> str:
    chunks = [entry.key, entry.kind, entry.name, entry.source, entry.summary]
    if entry.path:
        chunks.append(entry.path)
    chunks.extend(str(t) for t in entry.tags)
    for payload in (entry.metadata, entry.fields, entry.relations):
        chunks.extend(_flatten_scalars(payload))
    return " ".join(chunks).lower()


def search_entries(
    query: str,
    *,
    profile: str = "default",
    kind: str | None = None,
    limit: int = 20,
    field_filters: Mapping[str, Any] | Sequence[tuple[str, Any]] | None = None,
) -> tuple[CatalogEntry, ...]:
    q = str(query).strip().lower()
    if not q:
        return list_entries(profile=profile, kind=kind, limit=limit, field_filters=field_filters)

    entries = list_entries(profile=profile, kind=kind, limit=None, field_filters=field_filters)
    scored: list[tuple[int, CatalogEntry]] = []
    for e in entries:
        text = _search_text(e)
        idx = text.find(q)
        if idx >= 0:
            scored.append((idx, e))

    scored.sort(key=lambda x: (x[0], x[1].kind, x[1].key))
    n = max(0, int(limit))
    return tuple(e for _, e in scored[:n])


def show_entry(key: str, *, profile: str = "default") -> CatalogEntry | None:
    target = str(key).strip().lower()
    if not target:
        return None
    for e in list_entries(profile=profile):
        if e.key.lower() == target:
            return e
    return None


def catalog_summary(*, profile: str = "default") -> Dict[str, Any]:
    items = list_entries(profile=profile)
    by_kind: dict[str, int] = {}
    for e in items:
        by_kind[e.kind] = int(by_kind.get(e.kind, 0) + 1)
    return {
        "profile": _normalize_profile(profile),
        "total": int(len(items)),
        "by_kind": dict(sorted(by_kind.items(), key=lambda kv: kv[0])),
    }


def field_values(
    field_name: str,
    *,
    profile: str = "default",
    kind: str | None = None,
    limit: int | None = None,
) -> tuple[str, ...]:
    target = _normalize_field_name(field_name)
    values: set[str] = set()
    for entry in list_entries(profile=profile, kind=kind, limit=None):
        for value in _entry_field_values(entry, target, include_relations=False):
            for scalar in _flatten_scalars(value):
                if scalar:
                    values.add(scalar)
    ordered = tuple(sorted(values))
    if limit is not None:
        return ordered[: max(0, int(limit))]
    return ordered


def catalog_schema(*, profile: str = "default", kind: str | None = None) -> Dict[str, Any]:
    items = list_entries(profile=profile, kind=kind, limit=None)
    by_kind: dict[str, dict[str, set[str]]] = {}
    for entry in items:
        bucket = by_kind.setdefault(entry.kind, {"fields": set(), "relations": set()})
        bucket["fields"].update(str(k) for k in entry.fields.keys())
        bucket["relations"].update(str(k) for k in entry.relations.keys())

    if kind is not None:
        k = _normalize_kind(kind) or ""
        payload = by_kind.get(k, {"fields": set(), "relations": set()})
        return {
            "profile": _normalize_profile(profile),
            "kind": k,
            "base_fields": list(_BASE_ENTRY_FIELDS),
            "fields": sorted(payload["fields"]),
            "relations": sorted(payload["relations"]),
            "count": len(items),
        }

    return {
        "profile": _normalize_profile(profile),
        "base_fields": list(_BASE_ENTRY_FIELDS),
        "kinds": {
            entry_kind: {
                "fields": sorted(payload["fields"]),
                "relations": sorted(payload["relations"]),
                "count": int(len([e for e in items if e.kind == entry_kind])),
            }
            for entry_kind, payload in sorted(by_kind.items(), key=lambda kv: kv[0])
        },
    }


def catalog_neighbors(
    key: str,
    *,
    profile: str = "default",
) -> Dict[str, Any]:
    entry = show_entry(key, profile=profile)
    if entry is None:
        return {
            "profile": _normalize_profile(profile),
            "key": str(key),
            "entry": None,
            "neighbors": {},
        }

    neighbor_payload: dict[str, list[dict[str, Any]]] = {}
    for relation_name, relation_value in dict(entry.relations).items():
        rows: list[dict[str, Any]] = []
        for candidate_key in _flatten_scalars(relation_value):
            target = show_entry(candidate_key, profile=profile)
            if target is None:
                rows.append({"key": str(candidate_key), "kind": None, "name": None, "missing": True})
                continue
            rows.append(
                {
                    "key": target.key,
                    "kind": target.kind,
                    "name": target.name,
                    "summary": target.summary,
                    "fields": _jsonable(dict(target.fields)),
                }
            )
        neighbor_payload[str(relation_name)] = rows

    return {
        "profile": _normalize_profile(profile),
        "key": entry.key,
        "entry": entry.to_dict(),
        "neighbors": neighbor_payload,
    }


def catalog_facets(
    *,
    profile: str = "default",
    kind: str | None = None,
    query: str | None = None,
    field_filters: Mapping[str, Any] | Sequence[tuple[str, Any]] | None = None,
    fields: Sequence[str] | None = None,
    limit_per_field: int = 25,
) -> Dict[str, Any]:
    filters = _normalize_field_filters(field_filters)
    if query and str(query).strip():
        items = search_entries(
            str(query),
            profile=profile,
            kind=kind,
            limit=10_000,
            field_filters=filters,
        )
    else:
        items = list_entries(profile=profile, kind=kind, limit=None, field_filters=filters)

    schema = catalog_schema(profile=profile, kind=kind)
    target_fields = tuple(str(v) for v in (fields or schema.get("fields", [])) if str(v).strip())
    facets: dict[str, list[dict[str, Any]]] = {}
    for field_name in target_fields:
        counts: dict[str, int] = {}
        for entry in items:
            seen: set[str] = set()
            for value in _entry_field_values(entry, field_name, include_relations=False):
                for scalar in _flatten_scalars(value):
                    if not scalar or scalar in seen:
                        continue
                    seen.add(scalar)
                    counts[scalar] = int(counts.get(scalar, 0) + 1)
        ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        facets[field_name] = [
            {"value": str(value), "count": int(count)}
            for value, count in ordered[: max(0, int(limit_per_field))]
        ]

    return {
        "profile": _normalize_profile(profile),
        "kind": _normalize_kind(kind),
        "query": str(query or ""),
        "filters": [{"field": str(name), "value": str(value)} for name, value in filters],
        "total": int(len(items)),
        "facets": facets,
    }


def catalog_ui_snapshot(
    *,
    profile: str = "default",
    kind: str | None = None,
    query: str | None = None,
    field_filters: Mapping[str, Any] | Sequence[tuple[str, Any]] | None = None,
    limit: int = 200,
    selected_key: str | None = None,
) -> Dict[str, Any]:
    filters = _normalize_field_filters(field_filters)
    if query and str(query).strip():
        items = search_entries(
            str(query),
            profile=profile,
            kind=kind,
            limit=limit,
            field_filters=filters,
        )
    else:
        items = list_entries(
            profile=profile,
            kind=kind,
            limit=limit,
            field_filters=filters,
        )

    schema = catalog_schema(profile=profile, kind=kind)
    kind_key = _normalize_kind(kind) or ""
    facet_fields = tuple(_UI_FACET_FIELDS_BY_KIND.get(kind_key, ())) or tuple(schema.get("fields", []))
    facets = catalog_facets(
        profile=profile,
        kind=kind,
        query=query,
        field_filters=filters,
        fields=facet_fields,
        limit_per_field=20,
    )
    selected_entry = None
    neighbors = None
    if selected_key:
        selected_entry = show_entry(str(selected_key), profile=profile)
        neighbors = catalog_neighbors(str(selected_key), profile=profile)

    return {
        "profile": _normalize_profile(profile),
        "kind": _normalize_kind(kind),
        "query": str(query or ""),
        "filters": [{"field": str(name), "value": str(value)} for name, value in filters],
        "summary": catalog_summary(profile=profile),
        "schema": schema,
        "facets": facets,
        "items": [entry.to_dict() for entry in items],
        "selected": selected_entry.to_dict() if selected_entry is not None else None,
        "neighbors": neighbors,
    }
