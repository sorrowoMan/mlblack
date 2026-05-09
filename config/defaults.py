from __future__ import annotations

import math
from typing import Any, Dict, Mapping, MutableMapping, Sequence

from bias import L2ScaleBias, NoOpBias
from core.common.family_router import serialize_family_route_registry
from core.mechanisms import (
    build_adaboost_mechanism_bindings,
    build_bagging_mechanism_bindings,
    build_extra_trees_mechanism_bindings,
    build_linear_family_mechanism_bindings,
    build_neural_family_mechanism_bindings,
    build_random_forest_mechanism_bindings,
    build_symbolic_family_mechanism_bindings,
    build_tree_boosting_family_mechanism_bindings,
    serialize_family_bindings,
)
from core.symbolic.trainer_family import (
    SYMBOLIC_FORMAL_PRESET_KEY,
    SymbolicTrainerFamilySpec,
    build_unified_symbolic_family_spec,
    canonical_symbolic_preset_key,
    coerce_symbolic_family_spec,
    legacy_symbolic_family_spec,
    resolve_symbolic_route_spec,
    resolve_symbolic_router_target,
    serialize_symbolic_route_registry,
    symbolic_route_registry,
)
from core.symbolic.search_mechanism_contract import (
    build_symbolic_search_mechanism_contracts,
    serialize_symbolic_search_mechanism_contracts,
)
from core.linear.trainer_family import (
    LINEAR_FORMAL_PRESET_KEY,
    LinearTrainerFamilySpec,
    build_unified_linear_family_spec,
    build_ridge_family_spec,
    coerce_linear_family_spec,
    linear_route_registry,
    resolve_linear_route_spec,
    resolve_linear_router_target,
)
from core.neural.trainer_family import (
    NEURAL_FORMAL_PRESET_KEY,
    NeuralTrainerFamilySpec,
    build_unified_neural_family_spec,
    build_sklearn_mlp_family_spec,
    build_torch_mlp_family_spec,
    coerce_neural_family_spec,
    neural_route_registry,
    resolve_neural_route_spec,
    resolve_neural_router_target,
)
from core.tree.trainer_family import (
    TREE_ENSEMBLE_FORMAL_PRESET_KEY,
    TreeTrainerFamilySpec,
    build_adaboost_family_spec,
    build_bagging_family_spec,
    build_extra_trees_family_spec,
    build_random_forest_family_spec,
    build_unified_tree_ensemble_family_spec,
    coerce_tree_family_spec,
    resolve_tree_ensemble_route_spec,
    resolve_tree_ensemble_router_target,
    tree_ensemble_route_registry,
)
from core.tree_boosting.trainer_family import (
    TREE_BOOSTING_FORMAL_PRESET_KEY,
    TreeBoostingTrainerFamilySpec,
    build_unified_tree_boosting_family_spec,
    build_xgboost_family_spec,
    coerce_tree_boosting_family_spec,
    resolve_tree_boosting_route_spec,
    resolve_tree_boosting_router_target,
    tree_boosting_route_registry,
)
from core.trainers.sklearn_mlp_trainer import SklearnMLPSurrogateTrainer, SklearnMLPTrainerConfig
from core.trainers.symbolic_torch_interval_trainer import SymbolicTorchIntervalTrainer, SymbolicTorchIntervalTrainerConfig
from core.trainers.symbolic_torch_trainer import SymbolicTorchSurrogateTrainer, SymbolicTorchTrainerConfig
from core.trainers.symbolic_stagewise_trainer import SymbolicStagewiseSurrogateTrainer, SymbolicStagewiseTrainerConfig
from core.trainers.symbolic_orthogonal_trainer import SymbolicOrthogonalSurrogateTrainer, SymbolicOrthogonalTrainerConfig
from core.trainers.torch_trainer import TorchMLPSurrogateTrainer, TorchMLPTrainerConfig
from core.trainers.trainer import RidgeSurrogateTrainer, RidgeTrainerConfig
from core.flow_experiment_tracker import build_experiment_tracker_capability
from pipeline import IdentityPipeline, ZScorePipeline
from numericizer import DefaultNumericizer

from .registry import MLBlackConfig


def _split_numericizer_options(cfg: Dict[str, Any]) -> tuple[Dict[str, Any], Any, Any, Any, Any, Any]:
    work = dict(cfg)
    numericizer = work.pop("numericizer", None)
    modality_encoders = work.pop("modality_encoders", None)
    target_codecs = work.pop("target_codecs", None)
    target_codec = work.pop("target_codec", None)
    categorical_unknown = work.pop("categorical_unknown", None)
    return work, numericizer, modality_encoders, target_codecs, target_codec, categorical_unknown


_SYMBOLIC_STAGEWISE_GROUP_FIELD_MAPS: Dict[str, Dict[str, str]] = {
    "strategy": {
        "force_linear_base": "force_linear_base",
        "keep_search_trace": "keep_search_trace",
    },
    "auto_mode": {
        "val_ratio": "auto_val_ratio",
        "min_val_samples": "auto_min_val_samples",
        "random_seed": "auto_random_seed",
        "term_penalty": "auto_term_penalty",
        "depth_penalty": "auto_depth_penalty",
        "grad_penalty": "auto_grad_penalty",
    },
    "search_core": {
        "max_added_terms": "search_max_added_terms",
        "topk_features": "search_topk_features",
        "max_pair_terms": "search_max_pair_terms",
        "max_candidates_per_iter": "search_max_candidates_per_iter",
        "candidate_keep_top": "search_candidate_keep_top",
        "max_arity": "search_max_arity",
        "max_expr_depth": "search_max_expr_depth",
        "min_actual_rmse_gain": "search_min_actual_rmse_gain",
        "ridge_l2": "search_ridge_l2",
        "min_score": "search_min_score",
        "min_projected_gain": "search_min_projected_gain",
        "score_complexity_penalty": "search_score_complexity_penalty",
        "score_corr_bonus": "search_score_corr_bonus",
    },
    "search_overfit": {
        "enabled": "search_overfit_guard_enabled",
        "val_ratio": "search_overfit_guard_val_ratio",
        "min_val_samples": "search_overfit_guard_min_val_samples",
        "random_seed": "search_overfit_guard_random_seed",
        "min_val_rmse_gain": "search_overfit_guard_min_val_rmse_gain",
        "max_gap_increase": "search_overfit_guard_max_gap_increase",
        "patience": "search_overfit_guard_patience",
        "snapshot_min_improve": "search_overfit_guard_snapshot_min_improve",
        "tabu_rounds": "search_overfit_guard_tabu_rounds",
        "replace_topk": "search_overfit_guard_replace_topk",
        "replace_drop_topk": "search_overfit_guard_replace_drop_topk",
    },
    "search_gradient": {
        "guidance_bonus": "search_grad_guidance_bonus",
        "focus_topk": "search_grad_focus_topk",
        "min_priority": "search_grad_min_priority",
        "slope_mode": "search_grad_slope_mode",
        "slope_bins": "search_grad_slope_bins",
        "slope_min_bin_samples": "search_grad_slope_min_bin_samples",
        "adv_check": "search_grad_adv_check",
        "adv_trials": "search_grad_adv_trials",
        "adv_noise_std": "search_grad_adv_noise_std",
        "adv_min_stability": "search_grad_adv_min_stability",
        "adv_random_seed": "search_grad_adv_random_seed",
        "enable_residual_projection": "search_enable_grad_residual_projection",
        "projection_topk_focus": "search_grad_projection_topk_focus",
        "projection_partner_pool": "search_grad_projection_partner_pool",
        "projection_topk_partners": "search_grad_projection_topk_partners",
        "projection_topk_unary": "search_grad_projection_topk_unary",
        "projection_focus_include_transforms": "search_grad_projection_focus_include_transforms",
        "projection_focus_topk_transforms": "search_grad_projection_focus_topk_transforms",
        "projection_partner_orders": "search_grad_projection_partner_orders",
        "projection_enable_pair_dictionary": "search_grad_projection_enable_pair_dictionary",
        "projection_min_abs_corr": "search_grad_projection_min_abs_corr",
        "projection_max_generated": "search_grad_projection_max_generated",
        "interaction_grad_projection_budget_boost": "search_interaction_grad_projection_budget_boost",
    },
    "search_family": {
        "include_hinge": "search_include_hinge",
        "hinge_quantiles": "search_hinge_quantiles",
        "unary_ops": "search_unary_ops",
        "nested_mode": "search_nested_mode",
        "nested_unary_patterns": "search_nested_unary_patterns",
        "auto_nested_allowed_ops": "search_auto_nested_allowed_ops",
        "auto_nested_min_depth": "search_auto_nested_min_depth",
        "auto_nested_max_depth": "search_auto_nested_max_depth",
        "auto_nested_beam_width": "search_auto_nested_beam_width",
        "auto_nested_max_patterns_per_feature": "search_auto_nested_max_patterns_per_feature",
        "interaction_budget_mode": "search_interaction_budget_mode",
        "interaction_diag_threshold": "search_interaction_diag_threshold",
        "interaction_diag_topk_features": "search_interaction_diag_topk_features",
        "interaction_pair_budget_boost": "search_interaction_pair_budget_boost",
    },
    "search_prune": {
        "enabled": "search_enable_prune",
        "rmse_tolerance": "search_prune_rmse_tolerance",
        "max_removed_per_iter": "search_prune_max_removed_per_iter",
    },
    "search_path_memory": {
        "enabled": "search_path_memory_enabled",
        "db_path": "search_path_memory_db_path",
        "namespace": "search_path_memory_namespace",
        "prior_bonus": "search_path_memory_prior_bonus",
        "tabu_penalty": "search_path_memory_tabu_penalty",
        "min_outcomes": "search_path_memory_min_outcomes",
        "hard_tabu": "search_path_memory_hard_tabu",
        "hard_tabu_accept_rate": "search_path_memory_hard_tabu_accept_rate",
    },
    "search_graph_cache": {
        "enabled": "search_graph_cache_enabled",
        "max_value_entries": "search_graph_cache_max_value_entries",
        "max_derivative_entries": "search_graph_cache_max_derivative_entries",
        "backend": "search_graph_cache_backend",
        "db_path": "search_graph_cache_db_path",
        "namespace": "search_graph_cache_namespace",
        "persist_values": "search_graph_cache_persist_values",
    },
    "search_online_beam": {
        "enabled": "search_online_beam_enabled",
        "width": "search_online_beam_width",
        "bundle_size": "search_online_bundle_size",
        "branches_per_beam": "search_online_branches_per_beam",
        "jitter": "search_online_beam_jitter",
        "early_stop_rounds": "search_online_early_stop_rounds",
    },
    "search_joint_bundle": {
        "enabled": "search_joint_bundle_enabled",
        "max_terms": "search_joint_bundle_max_terms",
        "preselect_topk": "search_joint_bundle_preselect_topk",
        "max_combos": "search_joint_bundle_max_combos",
        "l1_alpha": "search_joint_bundle_l1_alpha",
        "l1_iters": "search_joint_bundle_l1_iters",
    },
    "search_inner_opt": {
        "enabled": "search_inner_opt_enabled",
        "method": "search_inner_opt_method",
        "device": "search_inner_opt_device",
        "random_seed": "search_inner_opt_random_seed",
        "adam_steps": "search_inner_opt_adam_steps",
        "adam_lr": "search_inner_opt_adam_lr",
        "adam_weight_decay": "search_inner_opt_adam_weight_decay",
        "lbfgs_steps": "search_inner_opt_lbfgs_steps",
        "lbfgs_lr": "search_inner_opt_lbfgs_lr",
        "l2": "search_inner_opt_l2",
        "accept_rmse_tol": "search_inner_opt_accept_rmse_tol",
    },
    "artifact_runtime": {
        "ood_z_threshold": "ood_z_threshold",
        "epsilon": "epsilon",
    },
}


def _normalize_symbolic_stagewise_config_dict(cfg: Dict[str, Any]) -> Dict[str, Any]:
    work = dict(cfg)
    nested_flat: Dict[str, Any] = {}

    for group_name, field_map in _SYMBOLIC_STAGEWISE_GROUP_FIELD_MAPS.items():
        section = work.pop(group_name, None)
        if section is None:
            continue
        if not isinstance(section, Mapping):
            raise TypeError(f"symbolic_stagewise config section '{group_name}' must be a mapping")

        unknown = [str(k) for k in section.keys() if str(k) not in field_map]
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(
                f"symbolic_stagewise config section '{group_name}' contains unknown key(s): {names}"
            )

        for nested_key, flat_key in field_map.items():
            if nested_key in section:
                nested_flat[flat_key] = section[nested_key]

    merged = dict(nested_flat)
    merged.update(work)
    return merged


def _as_str_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return tuple()
    if isinstance(value, str):
        s = value.strip()
        return tuple() if not s else (s,)
    if isinstance(value, Sequence):
        return tuple(str(x) for x in value)
    return (str(value),)


def _normalize_symbolic_torch_like_dict(cfg: Dict[str, Any]) -> Dict[str, Any]:
    work = dict(cfg)
    if "library_ops" in work:
        work["library_ops"] = tuple(str(x) for x in work["library_ops"])
    if "v2_continuous_ops" in work:
        work["v2_continuous_ops"] = tuple(str(x) for x in work["v2_continuous_ops"])
    if "v2_binary_ops" in work:
        work["v2_binary_ops"] = tuple(str(x) for x in work["v2_binary_ops"])
    if "v2_hinge_quantiles" in work:
        work["v2_hinge_quantiles"] = tuple(float(x) for x in work["v2_hinge_quantiles"])
    if "genome" in work and work["genome"] is not None:
        work["genome"] = tuple(work["genome"])
    return work


def _resolve_symbolic_family_from_config(
    cfg: Dict[str, Any] | SymbolicTrainerFamilySpec | None,
    *,
    default_backend: str = "ridge",
    default_task: str = "point",
) -> tuple[SymbolicTrainerFamilySpec, Dict[str, Any]]:
    if isinstance(cfg, SymbolicTrainerFamilySpec):
        return cfg, {}
    work = dict(cfg or {})
    raw_family = work.pop("family_spec", None)
    raw_structure_engine = work.pop("structure_engine", None)
    raw_parameter_backend_spec = work.pop("parameter_backend_spec", None)
    raw_parameter_backend = work.pop("parameter_backend", None)
    raw_task_head = work.pop("task_head", None)
    raw_task = work.pop("task", None)
    raw_calibration_mode = work.pop("calibration_mode", None)

    family_payload: Dict[str, Any] | None
    if raw_family is None:
        family_payload = None
    elif isinstance(raw_family, SymbolicTrainerFamilySpec):
        return raw_family, work
    else:
        family_payload = dict(raw_family)

    if family_payload is None:
        family_payload = {}
    if raw_structure_engine is not None:
        family_payload["structure_engine"] = raw_structure_engine
    if raw_parameter_backend_spec is not None:
        family_payload["parameter_backend"] = raw_parameter_backend_spec
    elif raw_parameter_backend is not None:
        family_payload["parameter_backend"] = raw_parameter_backend
    if raw_task_head is not None:
        family_payload["task_head"] = raw_task_head
    else:
        if raw_task is not None:
            family_payload["task"] = raw_task
        if raw_calibration_mode is not None:
            family_payload["calibration_mode"] = raw_calibration_mode

    if family_payload:
        spec = coerce_symbolic_family_spec(
            family_payload,
            trainer_key="symbolic",
            default_backend=default_backend,
            default_task=default_task,
        )
    else:
        spec = build_unified_symbolic_family_spec(
            trainer_key="symbolic",
            parameter_backend=default_backend,
            task=default_task,
        )
    return spec, work


def _attach_symbolic_family(trainer: Any, family_spec: SymbolicTrainerFamilySpec) -> Any:
    try:
        route_spec = resolve_symbolic_route_spec(family_spec)
        setattr(trainer, "symbolic_family_spec", family_spec)
        setattr(trainer, "symbolic_family_metadata", family_spec.description_dict())
        setattr(trainer, "symbolic_router_target", route_spec.route_key)
        setattr(trainer, "symbolic_route_spec", route_spec.as_dict())
        setattr(trainer, "symbolic_route_registry", serialize_symbolic_route_registry())
        _attach_family_route_contract(
            trainer,
            family_key=SYMBOLIC_FORMAL_PRESET_KEY,
            route_key=route_spec.route_key,
            route_spec=route_spec,
            route_registry=tuple(route_spec.as_family_route_spec() for route_spec in symbolic_route_registry()),
        )
    except Exception:
        return trainer
    return trainer


_NEURAL_GROUP_FIELD_MAPS: Dict[str, Dict[str, str]] = {
    "backend": {
        "parameter_backend": "parameter_backend",
        "runtime_backend": "runtime_backend",
        "trainer_kind": "trainer_kind",
        "continuation_mode": "continuation_mode",
        "trainer_state_enabled": "trainer_state_enabled",
        "supports_resume": "supports_resume",
        "supports_warm_start": "supports_warm_start",
        "supports_incremental": "supports_incremental",
    },
    "backbone": {
        "hidden_layers": "hidden_layers",
        "activation": "activation",
        "dropout": "dropout",
    },
    "optimization": {
        "objective": "objective",
        "optimizer": "optimizer",
        "optimizer_params": "optimizer_params",
        "solver": "solver",
        "lr": "lr",
        "weight_decay": "weight_decay",
        "alpha": "alpha",
        "learning_rate_init": "learning_rate_init",
        "max_steps": "max_steps",
        "tol": "tol",
        "n_iter_no_change": "n_iter_no_change",
        "early_stopping": "early_stopping",
        "early_stop_patience": "early_stop_patience",
        "early_stop_min_delta": "early_stop_min_delta",
        "random_seed": "random_seed",
    },
    "batching": {
        "batch_size": "batch_size",
        "shuffle": "batch_shuffle",
        "drop_last": "batch_drop_last",
        "num_workers": "batch_num_workers",
        "pin_memory": "batch_pin_memory",
        "val_ratio": "val_ratio",
        "validation_fraction": "validation_fraction",
    },
    "task_head": {
        "task": "task",
        "objective_family": "objective_family",
        "outputs": "outputs",
        "uncertainty_mode": "uncertainty_mode",
    },
}


def _resolve_neural_family_from_config(
    raw_config: Dict[str, Any] | object | None,
    *,
    trainer_key: str = "mlp_torch",
    default_builder: Any = build_torch_mlp_family_spec,
) -> tuple[NeuralTrainerFamilySpec, Dict[str, Any]]:
    if raw_config is None:
        return default_builder(trainer_key=trainer_key), {}
    if isinstance(raw_config, NeuralTrainerFamilySpec):
        return raw_config, {}
    if not isinstance(raw_config, dict):
        raise TypeError("neural family config must be dict, NeuralTrainerFamilySpec, or None")

    work = dict(raw_config)
    raw_family = work.pop("family_spec", None)
    if isinstance(raw_family, NeuralTrainerFamilySpec):
        return raw_family, work

    family_payload: Dict[str, Any] = {}
    if isinstance(raw_family, Mapping):
        family_payload.update(dict(raw_family))

    alias_values = {
        "hidden_dims": work.pop("hidden_dims", None),
        "hidden_layer_sizes": work.pop("hidden_layer_sizes", None),
        "epochs": work.pop("epochs", None),
        "max_iter": work.pop("max_iter", None),
    }
    if alias_values["hidden_dims"] is not None and "hidden_layer_sizes" not in work:
        family_payload.setdefault("backbone", {})
        family_payload["backbone"] = {
            **dict(family_payload.get("backbone", {}) or {}),
            "hidden_layers": tuple(int(x) for x in tuple(alias_values["hidden_dims"])),
        }
    if alias_values["hidden_layer_sizes"] is not None:
        family_payload.setdefault("backbone", {})
        family_payload["backbone"] = {
            **dict(family_payload.get("backbone", {}) or {}),
            "hidden_layers": tuple(int(x) for x in tuple(alias_values["hidden_layer_sizes"])),
        }
    if alias_values["epochs"] is not None:
        family_payload.setdefault("optimization", {})
        family_payload["optimization"] = {
            **dict(family_payload.get("optimization", {}) or {}),
            "max_steps": int(alias_values["epochs"]),
        }
    if alias_values["max_iter"] is not None:
        family_payload.setdefault("optimization", {})
        family_payload["optimization"] = {
            **dict(family_payload.get("optimization", {}) or {}),
            "max_steps": int(alias_values["max_iter"]),
        }

    for group_name, field_map in _NEURAL_GROUP_FIELD_MAPS.items():
        group_payload = dict(family_payload.get(group_name, {}) or {})
        raw_group = work.pop(group_name, None)
        if isinstance(raw_group, Mapping):
            group_payload.update(dict(raw_group))
        for nested_field, flat_field in field_map.items():
            if flat_field in work:
                raw_value = work.pop(flat_field)
                if nested_field not in group_payload:
                    group_payload[nested_field] = raw_value
        if group_payload:
            family_payload[group_name] = group_payload

    if family_payload:
        base_payload = default_builder(trainer_key=trainer_key).as_dict()
        merged_payload = dict(base_payload)
        merged_payload["trainer_key"] = trainer_key
        for section_name, section_payload in family_payload.items():
            if isinstance(section_payload, Mapping) and isinstance(merged_payload.get(section_name), Mapping):
                merged_payload[section_name] = {
                    **dict(merged_payload.get(section_name, {})),
                    **dict(section_payload),
                }
            else:
                merged_payload[section_name] = section_payload
        spec = coerce_neural_family_spec(merged_payload, trainer_key=trainer_key)
    else:
        spec = default_builder(trainer_key=trainer_key)
    return spec, work


def _attach_family_route_contract(
    trainer: Any,
    *,
    family_key: str,
    route_key: str,
    route_spec: Any,
    route_registry: Any,
) -> Any:
    try:
        spec_payload = route_spec.as_dict() if hasattr(route_spec, "as_dict") else dict(route_spec)
        registry_payload = serialize_family_route_registry(tuple(route_registry))
        setattr(trainer, "family_router_family", str(family_key))
        setattr(trainer, "family_router_target", str(route_key))
        setattr(trainer, "family_route_spec", spec_payload)
        setattr(trainer, "family_route_registry", registry_payload)
    except Exception:
        return trainer
    return trainer


def _attach_neural_family(trainer: Any, family_spec: NeuralTrainerFamilySpec) -> Any:
    try:
        route_spec = resolve_neural_route_spec(family_spec)
        setattr(trainer, "neural_family_spec", family_spec)
        setattr(trainer, "neural_family_metadata", family_spec.description_dict())
        setattr(trainer, "neural_router_target", route_spec.route_key)
        _attach_family_route_contract(
            trainer,
            family_key=NEURAL_FORMAL_PRESET_KEY,
            route_key=route_spec.route_key,
            route_spec=route_spec,
            route_registry=neural_route_registry(),
        )
    except Exception:
        return trainer
    return trainer


_TREE_GROUP_FIELD_MAPS: Dict[str, Dict[str, str]] = {
    "ensemble": {
        "ensemble_kind": "ensemble_kind",
        "backend": "backend",
        "n_estimators": "n_estimators",
        "aggregation": "aggregation",
        "learning_rate": "learning_rate",
        "loss": "loss",
        "oob_score": "oob_score",
        "n_jobs": "n_jobs",
        "random_seed": "random_seed",
        "warm_start_enabled": "warm_start_enabled",
        "trainer_state_enabled": "trainer_state_enabled",
        "supports_resume": "supports_resume",
        "supports_warm_start": "supports_warm_start",
        "supports_incremental": "supports_incremental",
    },
    "sampling": {
        "bootstrap": "bootstrap",
        "bootstrap_features": "bootstrap_features",
        "max_samples": "max_samples",
        "max_features": "max_features",
        "class_weight": "class_weight",
    },
    "splitter": {
        "criterion": "criterion",
        "splitter": "splitter",
        "min_impurity_decrease": "min_impurity_decrease",
    },
    "regularization": {
        "max_depth": "max_depth",
        "min_samples_split": "min_samples_split",
        "min_samples_leaf": "min_samples_leaf",
        "min_weight_fraction_leaf": "min_weight_fraction_leaf",
        "max_leaf_nodes": "max_leaf_nodes",
        "ccp_alpha": "ccp_alpha",
    },
    "task_head": {
        "task": "task",
        "objective_family": "objective_family",
        "outputs": "outputs",
        "uncertainty_mode": "uncertainty_mode",
    },
}


def _resolve_tree_family_from_config(
    raw_config: Dict[str, Any] | object | None,
    *,
    trainer_key: str = "random_forest",
    default_builder: Any = build_random_forest_family_spec,
) -> tuple[TreeTrainerFamilySpec, Dict[str, Any]]:
    if raw_config is None:
        return default_builder(trainer_key=trainer_key), {}
    if isinstance(raw_config, TreeTrainerFamilySpec):
        return raw_config, {}
    if not isinstance(raw_config, dict):
        raise TypeError("tree family config must be dict, TreeTrainerFamilySpec, or None")

    work = dict(raw_config)
    raw_family = work.pop("family_spec", None)
    if isinstance(raw_family, TreeTrainerFamilySpec):
        return raw_family, work

    family_payload: Dict[str, Any] = {}
    if isinstance(raw_family, Mapping):
        family_payload.update(dict(raw_family))

    for group_name, field_map in _TREE_GROUP_FIELD_MAPS.items():
        group_payload = dict(family_payload.get(group_name, {}) or {})
        raw_group = work.pop(group_name, None)
        if isinstance(raw_group, Mapping):
            group_payload.update(dict(raw_group))
        for nested_field, flat_field in field_map.items():
            if flat_field in work:
                raw_value = work.pop(flat_field)
                if nested_field not in group_payload:
                    group_payload[nested_field] = raw_value
        if group_payload:
            family_payload[group_name] = group_payload

    if family_payload:
        base_payload = default_builder(trainer_key=trainer_key).as_dict()
        merged_payload = dict(base_payload)
        merged_payload["trainer_key"] = trainer_key
        for section_name, section_payload in family_payload.items():
            if isinstance(section_payload, Mapping) and isinstance(merged_payload.get(section_name), Mapping):
                merged_payload[section_name] = {
                    **dict(merged_payload.get(section_name, {})),
                    **dict(section_payload),
                }
            else:
                merged_payload[section_name] = section_payload
        spec = coerce_tree_family_spec(merged_payload, trainer_key=trainer_key)
    else:
        spec = default_builder(trainer_key=trainer_key)
    return spec, work


def _attach_tree_family(trainer: Any, family_spec: TreeTrainerFamilySpec) -> Any:
    try:
        route_spec = resolve_tree_ensemble_route_spec(family_spec)
        setattr(trainer, "tree_family_spec", family_spec)
        setattr(trainer, "tree_family_metadata", family_spec.description_dict())
        setattr(trainer, "tree_ensemble_router_target", route_spec.route_key)
        _attach_family_route_contract(
            trainer,
            family_key=TREE_ENSEMBLE_FORMAL_PRESET_KEY,
            route_key=route_spec.route_key,
            route_spec=route_spec,
            route_registry=tree_ensemble_route_registry(),
        )
    except Exception:
        return trainer
    return trainer


_LINEAR_GROUP_FIELD_MAPS: Dict[str, Dict[str, str]] = {
    "backend": {
        "parameter_backend": "parameter_backend",
        "runtime_backend": "runtime_backend",
        "solver_kind": "solver_kind",
        "continuation_mode": "continuation_mode",
        "trainer_state_enabled": "trainer_state_enabled",
        "supports_resume": "supports_resume",
        "supports_warm_start": "supports_warm_start",
        "supports_incremental": "supports_incremental",
    },
    "function_class": {
        "basis": "basis",
        "fit_intercept": "fit_intercept",
    },
    "regularization": {
        "penalty": "penalty",
        "l2": "l2",
    },
    "task_head": {
        "task": "task",
        "objective_family": "objective_family",
        "outputs": "outputs",
        "uncertainty_mode": "uncertainty_mode",
    },
}


def _resolve_linear_family_from_config(
    raw_config: Dict[str, Any] | object | None,
    *,
    trainer_key: str = "ridge",
    default_builder: Any = build_ridge_family_spec,
) -> tuple[LinearTrainerFamilySpec, Dict[str, Any]]:
    if raw_config is None:
        return default_builder(trainer_key=trainer_key), {}
    if isinstance(raw_config, LinearTrainerFamilySpec):
        return raw_config, {}
    if not isinstance(raw_config, dict):
        raise TypeError("linear family config must be dict, LinearTrainerFamilySpec, or None")

    work = dict(raw_config)
    raw_family = work.pop("family_spec", None)
    if isinstance(raw_family, LinearTrainerFamilySpec):
        return raw_family, work

    family_payload: Dict[str, Any] = {}
    if isinstance(raw_family, Mapping):
        family_payload.update(dict(raw_family))

    for group_name, field_map in _LINEAR_GROUP_FIELD_MAPS.items():
        group_payload = dict(family_payload.get(group_name, {}) or {})
        raw_group = work.pop(group_name, None)
        if isinstance(raw_group, Mapping):
            group_payload.update(dict(raw_group))
        for nested_field, flat_field in field_map.items():
            if flat_field in work:
                raw_value = work.pop(flat_field)
                if nested_field not in group_payload:
                    group_payload[nested_field] = raw_value
        if group_payload:
            family_payload[group_name] = group_payload

    if family_payload:
        base_payload = default_builder(trainer_key=trainer_key).as_dict()
        merged_payload = dict(base_payload)
        merged_payload["trainer_key"] = trainer_key
        for section_name, section_payload in family_payload.items():
            if isinstance(section_payload, Mapping) and isinstance(merged_payload.get(section_name), Mapping):
                merged_payload[section_name] = {
                    **dict(merged_payload.get(section_name, {})),
                    **dict(section_payload),
                }
            else:
                merged_payload[section_name] = section_payload
        spec = coerce_linear_family_spec(merged_payload, trainer_key=trainer_key)
    else:
        spec = default_builder(trainer_key=trainer_key)
    return spec, work


def _attach_linear_family(trainer: Any, family_spec: LinearTrainerFamilySpec) -> Any:
    try:
        route_spec = resolve_linear_route_spec(family_spec)
        setattr(trainer, "linear_family_spec", family_spec)
        setattr(trainer, "linear_family_metadata", family_spec.description_dict())
        setattr(trainer, "linear_router_target", route_spec.route_key)
        _attach_family_route_contract(
            trainer,
            family_key=LINEAR_FORMAL_PRESET_KEY,
            route_key=route_spec.route_key,
            route_spec=route_spec,
            route_registry=linear_route_registry(),
        )
    except Exception:
        return trainer
    return trainer


_TREE_BOOSTING_GROUP_FIELD_MAPS: Dict[str, Dict[str, str]] = {
    "backend": {
        "backend": "backend",
        "booster": "booster",
        "trainer_kind": "trainer_kind",
        "continuation_mode": "continuation_mode",
        "trainer_state_enabled": "trainer_state_enabled",
        "supports_resume": "supports_resume",
        "supports_warm_start": "supports_warm_start",
        "supports_incremental": "supports_incremental",
    },
    "boosting": {
        "n_estimators": "n_estimators",
        "learning_rate": "learning_rate",
        "objective": "objective",
        "tree_method": "tree_method",
        "verbosity": "verbosity",
        "aggregation": "aggregation",
    },
    "sampling": {
        "subsample": "subsample",
        "colsample_bytree": "colsample_bytree",
    },
    "regularization": {
        "max_depth": "max_depth",
        "min_child_weight": "min_child_weight",
        "gamma": "gamma",
        "reg_lambda": "reg_lambda",
        "reg_alpha": "reg_alpha",
    },
    "execution": {
        "n_jobs": "n_jobs",
        "random_seed": "random_seed",
    },
    "task_head": {
        "task": "task",
        "objective_family": "objective_family",
        "outputs": "outputs",
        "uncertainty_mode": "uncertainty_mode",
    },
}


def _resolve_tree_boosting_family_from_config(
    raw_config: Dict[str, Any] | object | None,
    *,
    trainer_key: str = "xgboost",
    default_builder: Any = build_xgboost_family_spec,
) -> tuple[TreeBoostingTrainerFamilySpec, Dict[str, Any]]:
    if raw_config is None:
        return default_builder(trainer_key=trainer_key), {}
    if isinstance(raw_config, TreeBoostingTrainerFamilySpec):
        return raw_config, {}
    if not isinstance(raw_config, dict):
        raise TypeError("tree boosting family config must be dict, TreeBoostingTrainerFamilySpec, or None")

    work = dict(raw_config)
    raw_family = work.pop("family_spec", None)
    if isinstance(raw_family, TreeBoostingTrainerFamilySpec):
        return raw_family, work

    family_payload: Dict[str, Any] = {}
    if isinstance(raw_family, Mapping):
        family_payload.update(dict(raw_family))

    for group_name, field_map in _TREE_BOOSTING_GROUP_FIELD_MAPS.items():
        group_payload = dict(family_payload.get(group_name, {}) or {})
        raw_group = work.pop(group_name, None)
        if isinstance(raw_group, Mapping):
            group_payload.update(dict(raw_group))
        for nested_field, flat_field in field_map.items():
            if flat_field in work:
                raw_value = work.pop(flat_field)
                if nested_field not in group_payload:
                    group_payload[nested_field] = raw_value
        if group_payload:
            family_payload[group_name] = group_payload

    if family_payload:
        base_payload = default_builder(trainer_key=trainer_key).as_dict()
        merged_payload = dict(base_payload)
        merged_payload["trainer_key"] = trainer_key
        for section_name, section_payload in family_payload.items():
            if isinstance(section_payload, Mapping) and isinstance(merged_payload.get(section_name), Mapping):
                merged_payload[section_name] = {
                    **dict(merged_payload.get(section_name, {})),
                    **dict(section_payload),
                }
            else:
                merged_payload[section_name] = section_payload
        spec = coerce_tree_boosting_family_spec(merged_payload, trainer_key=trainer_key)
    else:
        spec = default_builder(trainer_key=trainer_key)
    return spec, work


def _attach_tree_boosting_family(trainer: Any, family_spec: TreeBoostingTrainerFamilySpec) -> Any:
    try:
        route_spec = resolve_tree_boosting_route_spec(family_spec)
        setattr(trainer, "tree_boosting_family_spec", family_spec)
        setattr(trainer, "tree_boosting_family_metadata", family_spec.description_dict())
        setattr(trainer, "tree_boosting_router_target", route_spec.route_key)
        _attach_family_route_contract(
            trainer,
            family_key=TREE_BOOSTING_FORMAL_PRESET_KEY,
            route_key=route_spec.route_key,
            route_spec=route_spec,
            route_registry=tree_boosting_route_registry(),
        )
    except Exception:
        return trainer
    return trainer


class _NoOpFlowCapability:
    """Default capability object compatible with flow capability protocol."""

    def __init__(
        self,
        *,
        name: str,
        priority: int = 0,
        enabled: bool = True,
        is_algorithmic: bool = False,
        config: Dict[str, Any] | None = None,
        context_requires: Sequence[str] = tuple(),
        context_provides: Sequence[str] = tuple(),
        context_mutates: Sequence[str] = tuple(),
        context_cache: Sequence[str] = tuple(),
        context_notes: str | None = None,
    ) -> None:
        self.name = str(name)
        self.priority = int(priority)
        self.enabled = bool(enabled)
        self.is_algorithmic = bool(is_algorithmic)
        self.config = dict(config or {})
        self.context_requires = tuple(str(x) for x in context_requires)
        self.context_provides = tuple(str(x) for x in context_provides)
        self.context_mutates = tuple(str(x) for x in context_mutates)
        self.context_cache = tuple(str(x) for x in context_cache)
        self.context_notes = None if context_notes is None else str(context_notes)

    def on_flow_start(self, context: MutableMapping[str, Any]) -> None:
        return None

    def on_data_ready(self, context: MutableMapping[str, Any]) -> None:
        return None

    def on_pre_fit(self, context: MutableMapping[str, Any]) -> None:
        return None

    def on_post_fit(self, context: MutableMapping[str, Any]) -> None:
        return None

    def on_pre_eval(self, context: MutableMapping[str, Any]) -> None:
        return None

    def on_post_eval(self, context: MutableMapping[str, Any]) -> None:
        return None

    def on_pre_persist(self, context: MutableMapping[str, Any]) -> None:
        return None

    def on_post_persist(self, context: MutableMapping[str, Any]) -> None:
        return None

    def on_flow_finish(self, context: MutableMapping[str, Any]) -> None:
        return None

    def get_context_contract(self) -> Dict[str, Any]:
        return {
            "requires": tuple(self.context_requires),
            "provides": tuple(self.context_provides),
            "mutates": tuple(self.context_mutates),
            "cache": tuple(self.context_cache),
            "notes": self.context_notes,
        }


def _build_noop_flow_capability(**kwargs: Any) -> _NoOpFlowCapability:
    params = dict(kwargs)
    name = str(params.pop("name", "noop_capability"))
    priority = int(params.pop("priority", 0))
    enabled = bool(params.pop("enabled", True))
    is_algorithmic = bool(params.pop("is_algorithmic", False))
    context_requires = _as_str_tuple(params.pop("context_requires", tuple()))
    context_provides = _as_str_tuple(params.pop("context_provides", tuple()))
    context_mutates = _as_str_tuple(params.pop("context_mutates", tuple()))
    context_cache = _as_str_tuple(params.pop("context_cache", tuple()))
    context_notes = params.pop("context_notes", None)
    extra_config = dict(params.pop("config", {}))
    extra_config.update(params)
    return _NoOpFlowCapability(
        name=name,
        priority=priority,
        enabled=enabled,
        is_algorithmic=is_algorithmic,
        config=extra_config,
        context_requires=context_requires,
        context_provides=context_provides,
        context_mutates=context_mutates,
        context_cache=context_cache,
        context_notes=context_notes,
    )


class _MetricGuardCapability:
    """Evaluate post-fit metrics against declarative threshold rules."""

    _OPS = {
        "le": lambda value, threshold: value <= threshold,
        "lt": lambda value, threshold: value < threshold,
        "ge": lambda value, threshold: value >= threshold,
        "gt": lambda value, threshold: value > threshold,
    }

    def __init__(
        self,
        *,
        name: str,
        rules: Sequence[Mapping[str, Any]],
        hard_fail: bool = True,
        report_key: str = "metric_guard",
        priority: int = 0,
        enabled: bool = True,
        is_algorithmic: bool = False,
        config: Dict[str, Any] | None = None,
        context_requires: Sequence[str] = ("metrics",),
        context_provides: Sequence[str] = ("metric_guard",),
        context_mutates: Sequence[str] = ("report",),
        context_cache: Sequence[str] = tuple(),
        context_notes: str | None = "Checks metrics against declarative threshold rules.",
    ) -> None:
        self.name = str(name)
        self.priority = int(priority)
        self.enabled = bool(enabled)
        self.is_algorithmic = bool(is_algorithmic)
        self.config = dict(config or {})
        self.hard_fail = bool(hard_fail)
        self.report_key = str(report_key)
        self.context_requires = tuple(str(x) for x in context_requires)
        self.context_provides = tuple(str(x) for x in context_provides)
        self.context_mutates = tuple(str(x) for x in context_mutates)
        self.context_cache = tuple(str(x) for x in context_cache)
        self.context_notes = None if context_notes is None else str(context_notes)
        self.rules = tuple(self._normalize_rule(rule, index=i) for i, rule in enumerate(tuple(rules)))
        if not self.rules:
            raise ValueError("metric_guard requires at least one rule")

    def _normalize_rule(self, rule: Mapping[str, Any], *, index: int) -> Dict[str, Any]:
        item = dict(rule)
        split = str(item.get("split", "test")).strip().lower()
        metric = str(item.get("metric", "rmse")).strip().lower()
        op = str(item.get("op", "le")).strip().lower()
        if op not in self._OPS:
            known = ", ".join(sorted(self._OPS.keys()))
            raise ValueError(f"metric_guard rule[{index}] invalid op '{op}', expected one of [{known}]")
        if "threshold" not in item:
            raise ValueError(f"metric_guard rule[{index}] missing threshold")
        threshold = float(item["threshold"])
        if not math.isfinite(threshold):
            raise ValueError(f"metric_guard rule[{index}] threshold must be finite")
        return {
            "split": split,
            "metric": metric,
            "op": op,
            "threshold": threshold,
        }

    def on_flow_start(self, context: MutableMapping[str, Any]) -> None:
        return None

    def on_data_ready(self, context: MutableMapping[str, Any]) -> None:
        return None

    def on_pre_fit(self, context: MutableMapping[str, Any]) -> None:
        return None

    def on_post_fit(self, context: MutableMapping[str, Any]) -> None:
        return None

    def on_pre_eval(self, context: MutableMapping[str, Any]) -> None:
        return None

    def on_post_eval(self, context: MutableMapping[str, Any]) -> None:
        metrics_raw = context.get("metrics", {})
        metrics: Mapping[str, Any] = metrics_raw if isinstance(metrics_raw, Mapping) else {}
        results: list[Dict[str, Any]] = []
        violations: list[Dict[str, Any]] = []

        for idx, rule in enumerate(self.rules):
            split = str(rule["split"])
            metric = str(rule["metric"])
            op = str(rule["op"])
            threshold = float(rule["threshold"])

            row: Dict[str, Any] = {
                "index": int(idx),
                "split": split,
                "metric": metric,
                "op": op,
                "threshold": threshold,
                "passed": False,
                "status": "unknown",
            }

            split_metrics_raw = metrics.get(split)
            if not isinstance(split_metrics_raw, Mapping):
                row["status"] = "missing_split"
                row["message"] = f"split '{split}' not found"
                violations.append(dict(row))
                results.append(row)
                continue

            if metric not in split_metrics_raw:
                row["status"] = "missing_metric"
                row["message"] = f"metric '{metric}' not found in split '{split}'"
                violations.append(dict(row))
                results.append(row)
                continue

            try:
                value = float(split_metrics_raw[metric])
            except Exception:
                row["status"] = "invalid_metric"
                row["message"] = f"metric '{metric}' in split '{split}' cannot be converted to float"
                violations.append(dict(row))
                results.append(row)
                continue

            if not math.isfinite(value):
                row["status"] = "non_finite_metric"
                row["value"] = value
                row["message"] = f"metric '{metric}' in split '{split}' is non-finite"
                violations.append(dict(row))
                results.append(row)
                continue

            passed = bool(self._OPS[op](value, threshold))
            row["value"] = value
            row["passed"] = bool(passed)
            row["status"] = "ok" if passed else "violation"
            if not passed:
                row["message"] = f"{split}.{metric}={value:.6f} does not satisfy {op} {threshold:.6f}"
                violations.append(dict(row))
            results.append(row)

        summary = {
            "name": str(self.name),
            "ok": bool(len(violations) == 0),
            "hard_fail": bool(self.hard_fail),
            "rules": [dict(x) for x in results],
            "violations": [dict(x) for x in violations],
        }
        context[self.report_key] = summary

        if violations and self.hard_fail:
            head = "; ".join(str(v.get("message", "")) for v in violations[:3])
            raise RuntimeError(f"metric_guard violation(s): {head}")

    def on_pre_persist(self, context: MutableMapping[str, Any]) -> None:
        report_raw = context.get("report")
        if not isinstance(report_raw, dict):
            return
        info = context.get(self.report_key)
        if isinstance(info, Mapping):
            report_raw[self.report_key] = dict(info)

    def on_post_persist(self, context: MutableMapping[str, Any]) -> None:
        return None

    def on_flow_finish(self, context: MutableMapping[str, Any]) -> None:
        return None

    def get_context_contract(self) -> Dict[str, Any]:
        return {
            "requires": tuple(self.context_requires),
            "provides": tuple(self.context_provides),
            "mutates": tuple(self.context_mutates),
            "cache": tuple(self.context_cache),
            "notes": self.context_notes,
        }


def _build_metric_guard_capability(**kwargs: Any) -> _MetricGuardCapability:
    params = dict(kwargs)

    name = str(params.pop("name", "metric_guard"))
    priority = int(params.pop("priority", 0))
    enabled = bool(params.pop("enabled", True))
    is_algorithmic = bool(params.pop("is_algorithmic", False))
    hard_fail = bool(params.pop("hard_fail", True))
    report_key = str(params.pop("report_key", "metric_guard"))

    context_requires = _as_str_tuple(params.pop("context_requires", ("metrics",)))
    context_provides = _as_str_tuple(params.pop("context_provides", ("metric_guard",)))
    context_mutates = _as_str_tuple(params.pop("context_mutates", ("report",)))
    context_cache = _as_str_tuple(params.pop("context_cache", tuple()))
    context_notes = params.pop("context_notes", "Checks metrics against declarative threshold rules.")

    rules_raw = params.pop("rules", None)
    if rules_raw is None:
        split = str(params.pop("split", "test"))
        metric = str(params.pop("metric", "rmse"))
        threshold = params.pop("threshold", None)
        op = str(params.pop("op", "le"))
        max_value = params.pop("max_value", None)
        min_value = params.pop("min_value", None)

        rules: list[Dict[str, Any]] = []
        if threshold is not None:
            rules.append({"split": split, "metric": metric, "op": op, "threshold": threshold})
        if max_value is not None:
            rules.append({"split": split, "metric": metric, "op": "le", "threshold": max_value})
        if min_value is not None:
            rules.append({"split": split, "metric": metric, "op": "ge", "threshold": min_value})
        if not rules:
            raise ValueError("metric_guard requires rules or threshold/max_value/min_value")
    else:
        if isinstance(rules_raw, Mapping):
            rules = [dict(rules_raw)]
        elif isinstance(rules_raw, Sequence) and not isinstance(rules_raw, (str, bytes)):
            rules = [dict(x) for x in rules_raw]
        else:
            raise TypeError("metric_guard rules must be a mapping or sequence of mappings")

    extra_config = dict(params.pop("config", {}))
    extra_config.update(params)

    return _MetricGuardCapability(
        name=name,
        rules=tuple(rules),
        hard_fail=hard_fail,
        report_key=report_key,
        priority=priority,
        enabled=enabled,
        is_algorithmic=is_algorithmic,
        config=extra_config,
        context_requires=context_requires,
        context_provides=context_provides,
        context_mutates=context_mutates,
        context_cache=context_cache,
        context_notes=None if context_notes is None else str(context_notes),
    )


def _build_ridge_trainer(*, pipeline: Any, biases: Any, config: Dict[str, Any] | RidgeTrainerConfig | None = None):
    if config is None:
        family_spec = build_ridge_family_spec(trainer_key="ridge")
        trainer_cfg = RidgeTrainerConfig(
            l2=float(family_spec.regularization.l2),
            family_spec=family_spec,
        )
        numericizer = None
        modality_encoders = None
        target_codecs = None
        target_codec = None
        categorical_unknown = None
    elif isinstance(config, RidgeTrainerConfig):
        family_raw = getattr(config, "family_spec", None)
        if family_raw is None:
            family_spec = build_ridge_family_spec(
                trainer_key="ridge",
                l2=float(config.l2),
            )
        else:
            family_spec = coerce_linear_family_spec(family_raw, trainer_key="ridge")
        trainer_cfg = RidgeTrainerConfig(
            l2=float(family_spec.regularization.l2),
            ood_z_threshold=float(config.ood_z_threshold),
            artifact_id=str(config.artifact_id),
            family_spec=family_spec,
        )
        numericizer = None
        modality_encoders = None
        target_codecs = None
        target_codec = None
        categorical_unknown = None
    elif isinstance(config, LinearTrainerFamilySpec):
        family_spec = config
        trainer_cfg = RidgeTrainerConfig(
            l2=float(family_spec.regularization.l2),
            family_spec=family_spec,
        )
        numericizer = None
        modality_encoders = None
        target_codecs = None
        target_codec = None
        categorical_unknown = None
    elif isinstance(config, dict):
        cfg, numericizer, modality_encoders, target_codecs, target_codec, categorical_unknown = _split_numericizer_options(config)
        family_spec, cfg = _resolve_linear_family_from_config(
            cfg,
            trainer_key="ridge",
            default_builder=build_ridge_family_spec,
        )
        trainer_cfg = RidgeTrainerConfig(
            l2=float(family_spec.regularization.l2),
            ood_z_threshold=float(cfg.pop("ood_z_threshold", 4.0)),
            artifact_id=str(cfg.pop("artifact_id", "ridge_surrogate_v1")),
            family_spec=family_spec,
        )
        if cfg:
            unknown = ", ".join(sorted(str(k) for k in cfg.keys()))
            raise TypeError(f"unknown ridge trainer config fields: {unknown}")
    else:
        raise TypeError("trainer config must be dict, LinearTrainerFamilySpec, RidgeTrainerConfig, or None")

    trainer = RidgeSurrogateTrainer(
        config=trainer_cfg,
        pipeline=pipeline,
        biases=biases,
        numericizer=numericizer,
        modality_encoders=modality_encoders,
        target_codecs=target_codecs,
        target_codec=target_codec,
        categorical_unknown=("error" if categorical_unknown is None else str(categorical_unknown)),
    )
    return _attach_linear_family(trainer, family_spec)


def _build_linear_family_trainer(
    *,
    pipeline: Any,
    biases: Any,
    config: Dict[str, Any] | object | None = None,
):
    family_spec, cfg = _resolve_linear_family_from_config(
        config,
        trainer_key=LINEAR_FORMAL_PRESET_KEY,
        default_builder=build_unified_linear_family_spec,
    )
    route_target = resolve_linear_router_target(family_spec)
    concrete_cfg = dict(cfg)
    concrete_cfg["family_spec"] = family_spec
    route_builders = {
        "ridge": _build_ridge_trainer,
    }
    route_builder = route_builders.get(route_target)
    if route_builder is None:
        raise ValueError(
            f"unsupported linear route target '{route_target}' resolved for trainer_key='{LINEAR_FORMAL_PRESET_KEY}'"
        )
    trainer = route_builder(
        pipeline=pipeline,
        biases=biases,
        config=concrete_cfg,
    )
    return _attach_linear_family(trainer, family_spec)


def _build_torch_mlp_trainer(*, pipeline: Any, biases: Any, config: Dict[str, Any] | TorchMLPTrainerConfig | None = None):
    def _coerce_torch_batch_size(value: Any) -> int:
        if isinstance(value, str):
            text = str(value).strip().lower()
            if text == "auto":
                return int(TorchMLPTrainerConfig().batch_size)
            return max(1, int(text))
        return max(1, int(value))

    def _build_torch_config_from_family(
        family_spec: NeuralTrainerFamilySpec,
        extra: Dict[str, Any] | None = None,
    ) -> TorchMLPTrainerConfig:
        payload = {
            "hidden_dims": tuple(int(v) for v in family_spec.backbone.hidden_layers),
            "activation": str(family_spec.backbone.activation),
            "dropout": float(family_spec.backbone.dropout),
            "epochs": int(family_spec.optimization.max_steps),
            "batch_size": _coerce_torch_batch_size(family_spec.batching.batch_size),
            "batch_shuffle": bool(family_spec.batching.shuffle),
            "batch_drop_last": bool(family_spec.batching.drop_last),
            "batch_num_workers": int(family_spec.batching.num_workers),
            "batch_pin_memory": bool(family_spec.batching.pin_memory),
            "lr": float(family_spec.optimization.lr if family_spec.optimization.lr is not None else 1e-3),
            "weight_decay": float(
                family_spec.optimization.weight_decay if family_spec.optimization.weight_decay is not None else 1e-4
            ),
            "optimizer": str(family_spec.optimization.optimizer or "adamw"),
            "optimizer_params": dict(family_spec.optimization.optimizer_params),
            "objective": str(family_spec.optimization.objective),
            "val_ratio": float(family_spec.batching.val_ratio if family_spec.batching.val_ratio is not None else 0.15),
            "early_stop_patience": int(
                family_spec.optimization.early_stop_patience
                if family_spec.optimization.early_stop_patience is not None
                else 20
            ),
            "early_stop_min_delta": float(
                family_spec.optimization.early_stop_min_delta
                if family_spec.optimization.early_stop_min_delta is not None
                else 1e-6
            ),
            "random_seed": int(family_spec.optimization.random_seed),
        }
        payload.update(dict(extra or {}))
        return TorchMLPTrainerConfig(**payload)

    if config is None:
        family_spec = build_torch_mlp_family_spec(trainer_key="mlp_torch")
        trainer_cfg = _build_torch_config_from_family(family_spec)
        numericizer = None
        modality_encoders = None
        target_codecs = None
        target_codec = None
        categorical_unknown = None
    elif isinstance(config, NeuralTrainerFamilySpec):
        family_spec = config
        trainer_cfg = _build_torch_config_from_family(family_spec)
        numericizer = None
        modality_encoders = None
        target_codecs = None
        target_codec = None
        categorical_unknown = None
    elif isinstance(config, TorchMLPTrainerConfig):
        family_spec = build_torch_mlp_family_spec(
            trainer_key="mlp_torch",
            hidden_layers=tuple(int(v) for v in tuple(config.hidden_dims)),
            activation=str(config.activation),
            dropout=float(config.dropout),
            optimizer=str(config.optimizer),
            objective=str(config.objective),
            lr=float(config.lr),
            weight_decay=float(config.weight_decay),
            epochs=int(config.epochs),
            batch_size=int(config.batch_size),
            shuffle=bool(config.batch_shuffle),
            drop_last=bool(config.batch_drop_last),
            num_workers=int(config.batch_num_workers),
            pin_memory=bool(config.batch_pin_memory),
            val_ratio=float(config.val_ratio),
            early_stopping=True,
            early_stop_patience=int(config.early_stop_patience),
            early_stop_min_delta=float(config.early_stop_min_delta),
            random_seed=int(config.random_seed),
            metadata={"preset_kind": "torch_backend"},
        )
        trainer_cfg = config
        numericizer = None
        modality_encoders = None
        target_codecs = None
        target_codec = None
        categorical_unknown = None
    elif isinstance(config, dict):
        cfg, numericizer, modality_encoders, target_codecs, target_codec, categorical_unknown = _split_numericizer_options(config)
        family_spec, cfg = _resolve_neural_family_from_config(
            cfg,
            trainer_key="mlp_torch",
            default_builder=build_torch_mlp_family_spec,
        )
        trainer_cfg = _build_torch_config_from_family(family_spec, cfg)
    else:
        raise TypeError("trainer config must be dict, TorchMLPTrainerConfig, or None")

    trainer = TorchMLPSurrogateTrainer(
        config=trainer_cfg,
        pipeline=pipeline,
        biases=biases,
        numericizer=numericizer,
        modality_encoders=modality_encoders,
        target_codecs=target_codecs,
        target_codec=target_codec,
        categorical_unknown=("error" if categorical_unknown is None else str(categorical_unknown)),
    )
    return _attach_neural_family(trainer, family_spec)


def _build_sklearn_mlp_trainer(
    *, pipeline: Any, biases: Any, config: Dict[str, Any] | SklearnMLPTrainerConfig | None = None
):
    def _build_sklearn_config_from_family(
        family_spec: NeuralTrainerFamilySpec,
        extra: Dict[str, Any] | None = None,
    ) -> SklearnMLPTrainerConfig:
        payload = {
            "hidden_layer_sizes": tuple(int(v) for v in family_spec.backbone.hidden_layers),
            "activation": str(family_spec.backbone.activation),
            "solver": str(family_spec.optimization.solver or "adam"),
            "alpha": float(family_spec.optimization.alpha if family_spec.optimization.alpha is not None else 1e-4),
            "batch_size": family_spec.batching.batch_size,
            "learning_rate_init": float(
                family_spec.optimization.learning_rate_init
                if family_spec.optimization.learning_rate_init is not None
                else 1e-3
            ),
            "max_iter": int(family_spec.optimization.max_steps),
            "tol": float(family_spec.optimization.tol if family_spec.optimization.tol is not None else 1e-4),
            "n_iter_no_change": int(
                family_spec.optimization.n_iter_no_change
                if family_spec.optimization.n_iter_no_change is not None
                else 20
            ),
            "validation_fraction": float(
                family_spec.batching.validation_fraction
                if family_spec.batching.validation_fraction is not None
                else 0.15
            ),
            "early_stopping": bool(family_spec.optimization.early_stopping),
            "random_seed": int(family_spec.optimization.random_seed),
        }
        payload.update(dict(extra or {}))
        return SklearnMLPTrainerConfig(**payload)

    if config is None:
        family_spec = build_sklearn_mlp_family_spec(trainer_key="sklearn_mlp")
        trainer_cfg = _build_sklearn_config_from_family(family_spec)
        numericizer = None
        modality_encoders = None
        target_codecs = None
        target_codec = None
        categorical_unknown = None
    elif isinstance(config, NeuralTrainerFamilySpec):
        family_spec = config
        trainer_cfg = _build_sklearn_config_from_family(family_spec)
        numericizer = None
        modality_encoders = None
        target_codecs = None
        target_codec = None
        categorical_unknown = None
    elif isinstance(config, SklearnMLPTrainerConfig):
        family_spec = build_sklearn_mlp_family_spec(
            trainer_key="sklearn_mlp",
            hidden_layers=tuple(int(v) for v in tuple(config.hidden_layer_sizes)),
            activation=str(config.activation),
            solver=str(config.solver),
            alpha=float(config.alpha),
            learning_rate_init=float(config.learning_rate_init),
            max_iter=int(config.max_iter),
            tol=float(config.tol),
            n_iter_no_change=int(config.n_iter_no_change),
            validation_fraction=float(config.validation_fraction),
            early_stopping=bool(config.early_stopping),
            batch_size=config.batch_size,
            random_seed=int(config.random_seed),
            metadata={"preset_kind": "sklearn_backend"},
        )
        trainer_cfg = config
        numericizer = None
        modality_encoders = None
        target_codecs = None
        target_codec = None
        categorical_unknown = None
    elif isinstance(config, dict):
        cfg, numericizer, modality_encoders, target_codecs, target_codec, categorical_unknown = _split_numericizer_options(config)
        family_spec, cfg = _resolve_neural_family_from_config(
            cfg,
            trainer_key="sklearn_mlp",
            default_builder=build_sklearn_mlp_family_spec,
        )
        trainer_cfg = _build_sklearn_config_from_family(family_spec, cfg)
    else:
        raise TypeError("trainer config must be dict, SklearnMLPTrainerConfig, or None")

    trainer = SklearnMLPSurrogateTrainer(
        config=trainer_cfg,
        pipeline=pipeline,
        biases=biases,
        numericizer=numericizer,
        modality_encoders=modality_encoders,
        target_codecs=target_codecs,
        target_codec=target_codec,
        categorical_unknown=("error" if categorical_unknown is None else str(categorical_unknown)),
    )
    return _attach_neural_family(trainer, family_spec)


def _build_neural_family_trainer(
    *,
    pipeline: Any,
    biases: Any,
    config: Dict[str, Any] | object | None = None,
):
    family_spec, cfg = _resolve_neural_family_from_config(
        config,
        trainer_key=NEURAL_FORMAL_PRESET_KEY,
        default_builder=build_unified_neural_family_spec,
    )
    route_target = resolve_neural_router_target(family_spec)
    concrete_cfg = dict(cfg)
    concrete_cfg["family_spec"] = family_spec
    route_builders = {
        "mlp_torch": _build_torch_mlp_trainer,
        "sklearn_mlp": _build_sklearn_mlp_trainer,
    }
    route_builder = route_builders.get(route_target)
    if route_builder is None:
        raise ValueError(
            f"unsupported neural route target '{route_target}' resolved for trainer_key='{NEURAL_FORMAL_PRESET_KEY}'"
        )
    trainer = route_builder(
        pipeline=pipeline,
        biases=biases,
        config=concrete_cfg,
    )
    return _attach_neural_family(trainer, family_spec)


def _build_symbolic_torch_trainer(
    *, pipeline: Any, biases: Any, config: Dict[str, Any] | SymbolicTorchTrainerConfig | None = None
):
    family_spec = None
    if config is None:
        trainer_cfg = SymbolicTorchTrainerConfig()
        numericizer = None
        modality_encoders = None
        target_codecs = None
        target_codec = None
        categorical_unknown = None
        family_spec = legacy_symbolic_family_spec("symbolic_torch")
    elif isinstance(config, SymbolicTorchTrainerConfig):
        trainer_cfg = config
        numericizer = None
        modality_encoders = None
        target_codecs = None
        target_codec = None
        categorical_unknown = None
        family_spec = legacy_symbolic_family_spec(
            "symbolic_torch",
            trainer_params=trainer_cfg.__dict__,
        )
    elif isinstance(config, dict):
        cfg, numericizer, modality_encoders, target_codecs, target_codec, categorical_unknown = _split_numericizer_options(config)
        cfg = _normalize_symbolic_torch_like_dict(cfg)
        if "structure_engine_params" in cfg and cfg["structure_engine_params"] is not None:
            cfg["structure_engine_params"] = dict(cfg["structure_engine_params"])
        if "structure_engine" not in cfg or cfg["structure_engine"] is None:
            cfg["structure_engine"] = legacy_symbolic_family_spec(
                "symbolic_torch",
                trainer_params=cfg,
            ).structure_engine
        trainer_cfg = SymbolicTorchTrainerConfig(**cfg)
        family_spec = legacy_symbolic_family_spec(
            "symbolic_torch",
            trainer_params=cfg,
        )
    else:
        raise TypeError("trainer config must be dict, SymbolicTorchTrainerConfig, or None")

    trainer = SymbolicTorchSurrogateTrainer(
        config=trainer_cfg,
        pipeline=pipeline,
        biases=biases,
        numericizer=numericizer,
        modality_encoders=modality_encoders,
        target_codecs=target_codecs,
        target_codec=target_codec,
        categorical_unknown=("error" if categorical_unknown is None else str(categorical_unknown)),
    )
    return trainer if family_spec is None else _attach_symbolic_family(trainer, family_spec)




def _build_symbolic_stagewise_trainer(
    *, pipeline: Any, biases: Any, config: Dict[str, Any] | SymbolicStagewiseTrainerConfig | None = None
):
    family_spec = None
    if config is None:
        trainer_cfg = SymbolicStagewiseTrainerConfig()
        numericizer = None
        modality_encoders = None
        target_codecs = None
        target_codec = None
        categorical_unknown = None
        family_spec = legacy_symbolic_family_spec("symbolic_stagewise")
    elif isinstance(config, SymbolicStagewiseTrainerConfig):
        trainer_cfg = config
        numericizer = None
        modality_encoders = None
        target_codecs = None
        target_codec = None
        categorical_unknown = None
        family_spec = legacy_symbolic_family_spec("symbolic_stagewise")
    elif isinstance(config, dict):
        cfg, numericizer, modality_encoders, target_codecs, target_codec, categorical_unknown = _split_numericizer_options(config)
        cfg = _normalize_symbolic_stagewise_config_dict(cfg)
        if "search_hinge_quantiles" in cfg:
            cfg["search_hinge_quantiles"] = tuple(float(x) for x in cfg["search_hinge_quantiles"])
        if "search_unary_ops" in cfg:
            cfg["search_unary_ops"] = tuple(str(x) for x in cfg["search_unary_ops"])
        if "search_nested_unary_patterns" in cfg:
            cfg["search_nested_unary_patterns"] = tuple(str(x) for x in cfg["search_nested_unary_patterns"])
        if "search_auto_nested_allowed_ops" in cfg:
            cfg["search_auto_nested_allowed_ops"] = tuple(str(x) for x in cfg["search_auto_nested_allowed_ops"])
        if "structure_engine" not in cfg or cfg["structure_engine"] is None:
            cfg["structure_engine"] = legacy_symbolic_family_spec("symbolic_stagewise").structure_engine
        trainer_cfg = SymbolicStagewiseTrainerConfig(**cfg)
        family_spec = legacy_symbolic_family_spec("symbolic_stagewise")
    else:
        raise TypeError("trainer config must be dict, SymbolicStagewiseTrainerConfig, or None")

    trainer = SymbolicStagewiseSurrogateTrainer(
        config=trainer_cfg,
        pipeline=pipeline,
        biases=biases,
        numericizer=numericizer,
        modality_encoders=modality_encoders,
        target_codecs=target_codecs,
        target_codec=target_codec,
        categorical_unknown=("error" if categorical_unknown is None else str(categorical_unknown)),
    )
    return trainer if family_spec is None else _attach_symbolic_family(trainer, family_spec)


def _build_symbolic_orthogonal_trainer(
    *, pipeline: Any, biases: Any, config: Dict[str, Any] | SymbolicOrthogonalTrainerConfig | None = None
):
    family_spec = None
    if config is None:
        trainer_cfg = SymbolicOrthogonalTrainerConfig()
        numericizer = None
        modality_encoders = None
        target_codecs = None
        target_codec = None
        categorical_unknown = None
    elif isinstance(config, SymbolicOrthogonalTrainerConfig):
        trainer_cfg = config
        numericizer = None
        modality_encoders = None
        target_codecs = None
        target_codec = None
        categorical_unknown = None
    elif isinstance(config, dict):
        cfg, numericizer, modality_encoders, target_codecs, target_codec, categorical_unknown = _split_numericizer_options(config)
        work = dict(cfg)
        raw_family_spec = work.pop("family_spec", None)
        raw_structure_engine = work.pop("structure_engine", None)
        for field_name in ("l2_grid", "gate_quantiles"):
            if field_name in work:
                work[field_name] = tuple(float(value) for value in work[field_name])
        for field_name in ("gate_feature_names", "gate_families"):
            if field_name in work:
                work[field_name] = tuple(str(value) for value in work[field_name])
        trainer_cfg = SymbolicOrthogonalTrainerConfig(**work)
        if raw_family_spec is not None and isinstance(raw_family_spec, SymbolicTrainerFamilySpec):
            family_spec = raw_family_spec
        elif raw_family_spec is not None:
            family_spec = coerce_symbolic_family_spec(raw_family_spec, trainer_key="symbolic_orthogonal")
        elif raw_structure_engine is not None:
            family_spec = build_unified_symbolic_family_spec(
                trainer_key="symbolic_orthogonal",
                parameter_backend="ridge",
                task="point",
                trainer_state_enabled=True,
                supports_resume=True,
                supports_warm_start=True,
                supports_incremental=True,
                supports_piecewise_basis=bool(tuple(str(value) for value in trainer_cfg.gate_feature_names if str(value).strip())),
                metadata={
                    "preset_kind": "concrete_route",
                    "surface_status": "stable",
                    "canonical_preset": SYMBOLIC_FORMAL_PRESET_KEY,
                    "route_family": "symbolic",
                    "route_target": "symbolic_orthogonal",
                    "search_driver": "orthogonal_basis",
                    "supports_piecewise_basis": bool(tuple(str(value) for value in trainer_cfg.gate_feature_names if str(value).strip())),
                },
            )
            family_spec = type(family_spec)(
                trainer_key=family_spec.trainer_key,
                structure_engine=(
                    raw_structure_engine
                    if isinstance(raw_structure_engine, type(family_spec.structure_engine))
                    else type(family_spec.structure_engine)(**dict(raw_structure_engine))
                ),
                parameter_backend=family_spec.parameter_backend,
                task_head=family_spec.task_head,
                metadata=dict(family_spec.metadata),
            )
    else:
        raise TypeError("trainer config must be dict, SymbolicOrthogonalTrainerConfig, or None")

    if family_spec is None:
        family_payload = getattr(trainer_cfg, "__dict__", {})
        family_spec = build_unified_symbolic_family_spec(
            trainer_key="symbolic_orthogonal",
            parameter_backend="ridge",
            task="point",
            trainer_state_enabled=True,
            supports_resume=True,
            supports_warm_start=True,
            supports_incremental=True,
            supports_piecewise_basis=bool(tuple(str(value) for value in trainer_cfg.gate_feature_names if str(value).strip())),
            metadata={
                "preset_kind": "concrete_route",
                "surface_status": "stable",
                "canonical_preset": SYMBOLIC_FORMAL_PRESET_KEY,
                "route_family": "symbolic",
                "route_target": "symbolic_orthogonal",
                "search_driver": "orthogonal_basis",
                "supports_piecewise_basis": bool(tuple(str(value) for value in trainer_cfg.gate_feature_names if str(value).strip())),
                "orthogonal_basis_defaults": {
                    "candidate_limit": int(trainer_cfg.candidate_limit),
                    "group_count": int(trainer_cfg.group_count),
                    "selection_mode": str(trainer_cfg.selection_mode),
                    "gate_feature_names": [str(value) for value in tuple(trainer_cfg.gate_feature_names)],
                },
            },
        )
        family_spec = type(family_spec)(
            trainer_key=family_spec.trainer_key,
            structure_engine=type(family_spec.structure_engine)(
                structure_mode="orthogonal_basis_search",
                candidate_space=family_spec.structure_engine.candidate_space,
                grammar_source=family_spec.structure_engine.grammar_source,
                search_driver="orthogonal_basis",
                dynamic_pool_enabled=True,
                metadata={
                    **dict(family_spec.structure_engine.metadata),
                    "supports_piecewise_basis": bool(tuple(str(value) for value in trainer_cfg.gate_feature_names if str(value).strip())),
                },
            ),
            parameter_backend=family_spec.parameter_backend,
            task_head=family_spec.task_head,
            metadata=dict(family_spec.metadata),
        )

    trainer = SymbolicOrthogonalSurrogateTrainer(
        config=trainer_cfg,
        pipeline=pipeline,
        biases=biases,
        numericizer=numericizer,
        modality_encoders=modality_encoders,
        target_codecs=target_codecs,
        target_codec=target_codec,
        categorical_unknown=("error" if categorical_unknown is None else str(categorical_unknown)),
    )
    return trainer if family_spec is None else _attach_symbolic_family(trainer, family_spec)


def _build_symbolic_torch_interval_trainer(
    *, pipeline: Any, biases: Any, config: Dict[str, Any] | SymbolicTorchIntervalTrainerConfig | None = None
):
    family_spec = None
    if config is None:
        trainer_cfg = SymbolicTorchIntervalTrainerConfig()
        numericizer = None
        modality_encoders = None
        target_codecs = None
        target_codec = None
        categorical_unknown = None
        family_spec = legacy_symbolic_family_spec("symbolic_torch_interval")
    elif isinstance(config, SymbolicTorchIntervalTrainerConfig):
        trainer_cfg = config
        numericizer = None
        modality_encoders = None
        target_codecs = None
        target_codec = None
        categorical_unknown = None
        family_spec = legacy_symbolic_family_spec(
            "symbolic_torch_interval",
            trainer_params=trainer_cfg.__dict__,
        )
    elif isinstance(config, dict):
        cfg, numericizer, modality_encoders, target_codecs, target_codec, categorical_unknown = _split_numericizer_options(config)
        cfg = _normalize_symbolic_torch_like_dict(cfg)
        if "gate_feature_names" in cfg:
            cfg["gate_feature_names"] = tuple(str(x) for x in cfg["gate_feature_names"])
        if "stagewise_warmup_params" in cfg and cfg["stagewise_warmup_params"] is not None:
            cfg["stagewise_warmup_params"] = dict(cfg["stagewise_warmup_params"])
        if "structure_engine_params" in cfg and cfg["structure_engine_params"] is not None:
            cfg["structure_engine_params"] = dict(cfg["structure_engine_params"])
        if "structure_engine" not in cfg or cfg["structure_engine"] is None:
            cfg["structure_engine"] = legacy_symbolic_family_spec(
                "symbolic_torch_interval",
                trainer_params=cfg,
            ).structure_engine
        trainer_cfg = SymbolicTorchIntervalTrainerConfig(**cfg)
        family_spec = legacy_symbolic_family_spec(
            "symbolic_torch_interval",
            trainer_params=cfg,
        )
    else:
        raise TypeError("trainer config must be dict, SymbolicTorchIntervalTrainerConfig, or None")

    trainer = SymbolicTorchIntervalTrainer(
        config=trainer_cfg,
        pipeline=pipeline,
        biases=biases,
        numericizer=numericizer,
        modality_encoders=modality_encoders,
        target_codecs=target_codecs,
        target_codec=target_codec,
        categorical_unknown=("error" if categorical_unknown is None else str(categorical_unknown)),
    )
    return trainer if family_spec is None else _attach_symbolic_family(trainer, family_spec)


def _build_symbolic_family_trainer(
    *,
    pipeline: Any,
    biases: Any,
    config: Dict[str, Any] | None = None,
):
    family_spec, cfg = _resolve_symbolic_family_from_config(
        config,
        default_backend="ridge",
        default_task="point",
    )
    route_target = resolve_symbolic_router_target(
        family_spec,
        default_backend="ridge",
        default_task="point",
    )
    concrete_cfg = dict(cfg)
    concrete_cfg.setdefault("structure_engine", family_spec.structure_engine)
    if route_target == "symbolic_torch_interval" and "conformal_calibration" not in concrete_cfg:
        concrete_cfg["conformal_calibration"] = str(family_spec.task_head.calibration_mode).strip().lower() != "none"

    route_builders = {
        "symbolic_stagewise": _build_symbolic_stagewise_trainer,
        "symbolic_orthogonal": _build_symbolic_orthogonal_trainer,
        "symbolic_torch": _build_symbolic_torch_trainer,
        "symbolic_torch_interval": _build_symbolic_torch_interval_trainer,
    }
    route_builder = route_builders.get(route_target)
    if route_builder is None:
        raise ValueError(
            "unsupported symbolic route target "
            f"'{route_target}' resolved for trainer_key='{SYMBOLIC_FORMAL_PRESET_KEY}'"
        )

    trainer = route_builder(
        pipeline=pipeline,
        biases=biases,
        config=concrete_cfg,
    )
    return _attach_symbolic_family(trainer, family_spec)


def _build_xgboost_trainer(*, pipeline: Any, biases: Any, config: Dict[str, Any] | object | None = None):
    try:
        from core.trainers.xgboost_trainer import XGBoostSurrogateTrainer, XGBoostTrainerConfig
    except Exception as exc:
        raise ImportError(
            "xgboost trainer requires xgboost and scikit-learn. Install xgboost before using trainer_key='xgboost'."
        ) from exc

    if config is None:
        family_spec = build_xgboost_family_spec(trainer_key="xgboost")
        trainer_cfg = XGBoostTrainerConfig(
            n_estimators=int(family_spec.boosting.n_estimators),
            max_depth=int(family_spec.regularization.max_depth),
            learning_rate=float(family_spec.boosting.learning_rate),
            subsample=float(family_spec.sampling.subsample),
            colsample_bytree=float(family_spec.sampling.colsample_bytree),
            min_child_weight=float(family_spec.regularization.min_child_weight),
            gamma=float(family_spec.regularization.gamma),
            reg_lambda=float(family_spec.regularization.reg_lambda),
            reg_alpha=float(family_spec.regularization.reg_alpha),
            objective=str(family_spec.boosting.objective),
            tree_method=str(family_spec.boosting.tree_method),
            n_jobs=int(family_spec.execution.n_jobs),
            random_seed=int(family_spec.execution.random_seed),
            verbosity=int(family_spec.boosting.verbosity),
            family_spec=family_spec,
        )
        numericizer = None
        modality_encoders = None
        target_codecs = None
        target_codec = None
        categorical_unknown = None
    elif isinstance(config, XGBoostTrainerConfig):
        family_raw = getattr(config, "family_spec", None)
        if family_raw is None:
            family_spec = build_xgboost_family_spec(
                trainer_key="xgboost",
                n_estimators=int(config.n_estimators),
                max_depth=int(config.max_depth),
                learning_rate=float(config.learning_rate),
                subsample=float(config.subsample),
                colsample_bytree=float(config.colsample_bytree),
                min_child_weight=float(config.min_child_weight),
                gamma=float(config.gamma),
                reg_lambda=float(config.reg_lambda),
                reg_alpha=float(config.reg_alpha),
                objective=str(config.objective),
                tree_method=str(config.tree_method),
                n_jobs=int(config.n_jobs),
                random_seed=int(config.random_seed),
                verbosity=int(config.verbosity),
            )
        else:
            family_spec = coerce_tree_boosting_family_spec(family_raw, trainer_key="xgboost")
        trainer_cfg = XGBoostTrainerConfig(
            artifact_id=str(config.artifact_id),
            n_estimators=int(family_spec.boosting.n_estimators),
            max_depth=int(family_spec.regularization.max_depth),
            learning_rate=float(family_spec.boosting.learning_rate),
            subsample=float(family_spec.sampling.subsample),
            colsample_bytree=float(family_spec.sampling.colsample_bytree),
            min_child_weight=float(family_spec.regularization.min_child_weight),
            gamma=float(family_spec.regularization.gamma),
            reg_lambda=float(family_spec.regularization.reg_lambda),
            reg_alpha=float(family_spec.regularization.reg_alpha),
            objective=str(family_spec.boosting.objective),
            tree_method=str(family_spec.boosting.tree_method),
            n_jobs=int(family_spec.execution.n_jobs),
            random_seed=int(family_spec.execution.random_seed),
            verbosity=int(family_spec.boosting.verbosity),
            resume_training_from=config.resume_training_from,
            ood_z_threshold=float(config.ood_z_threshold),
            family_spec=family_spec,
            mechanisms=tuple(config.mechanisms),
        )
        numericizer = None
        modality_encoders = None
        target_codecs = None
        target_codec = None
        categorical_unknown = None
    elif isinstance(config, TreeBoostingTrainerFamilySpec):
        family_spec = config
        trainer_cfg = XGBoostTrainerConfig(
            n_estimators=int(family_spec.boosting.n_estimators),
            max_depth=int(family_spec.regularization.max_depth),
            learning_rate=float(family_spec.boosting.learning_rate),
            subsample=float(family_spec.sampling.subsample),
            colsample_bytree=float(family_spec.sampling.colsample_bytree),
            min_child_weight=float(family_spec.regularization.min_child_weight),
            gamma=float(family_spec.regularization.gamma),
            reg_lambda=float(family_spec.regularization.reg_lambda),
            reg_alpha=float(family_spec.regularization.reg_alpha),
            objective=str(family_spec.boosting.objective),
            tree_method=str(family_spec.boosting.tree_method),
            n_jobs=int(family_spec.execution.n_jobs),
            random_seed=int(family_spec.execution.random_seed),
            verbosity=int(family_spec.boosting.verbosity),
            family_spec=family_spec,
        )
        numericizer = None
        modality_encoders = None
        target_codecs = None
        target_codec = None
        categorical_unknown = None
    elif isinstance(config, dict):
        cfg, numericizer, modality_encoders, target_codecs, target_codec, categorical_unknown = _split_numericizer_options(config)
        family_spec, cfg = _resolve_tree_boosting_family_from_config(
            cfg,
            trainer_key="xgboost",
            default_builder=build_xgboost_family_spec,
        )
        trainer_cfg = XGBoostTrainerConfig(
            artifact_id=str(cfg.pop("artifact_id", "xgboost_surrogate_v1")),
            n_estimators=int(family_spec.boosting.n_estimators),
            max_depth=int(family_spec.regularization.max_depth),
            learning_rate=float(family_spec.boosting.learning_rate),
            subsample=float(family_spec.sampling.subsample),
            colsample_bytree=float(family_spec.sampling.colsample_bytree),
            min_child_weight=float(family_spec.regularization.min_child_weight),
            gamma=float(family_spec.regularization.gamma),
            reg_lambda=float(family_spec.regularization.reg_lambda),
            reg_alpha=float(family_spec.regularization.reg_alpha),
            objective=str(family_spec.boosting.objective),
            tree_method=str(family_spec.boosting.tree_method),
            n_jobs=int(family_spec.execution.n_jobs),
            random_seed=int(family_spec.execution.random_seed),
            verbosity=int(family_spec.boosting.verbosity),
            resume_training_from=cfg.pop("resume_training_from", None),
            ood_z_threshold=float(cfg.pop("ood_z_threshold", 4.0)),
            family_spec=family_spec,
            mechanisms=tuple(cfg.pop("mechanisms", XGBoostTrainerConfig().mechanisms)),
        )
        if cfg:
            unknown = ", ".join(sorted(str(k) for k in cfg.keys()))
            raise TypeError(f"unknown xgboost trainer config fields: {unknown}")
    else:
        raise TypeError(
            "trainer config must be dict, TreeBoostingTrainerFamilySpec, XGBoostTrainerConfig, or None"
        )

    trainer = XGBoostSurrogateTrainer(
        config=trainer_cfg,
        pipeline=pipeline,
        biases=biases,
        numericizer=numericizer,
        modality_encoders=modality_encoders,
        target_codecs=target_codecs,
        target_codec=target_codec,
        categorical_unknown=("error" if categorical_unknown is None else str(categorical_unknown)),
    )
    return _attach_tree_boosting_family(trainer, family_spec)


def _build_tree_boosting_family_trainer(
    *,
    pipeline: Any,
    biases: Any,
    config: Dict[str, Any] | object | None = None,
):
    family_spec, cfg = _resolve_tree_boosting_family_from_config(
        config,
        trainer_key=TREE_BOOSTING_FORMAL_PRESET_KEY,
        default_builder=build_unified_tree_boosting_family_spec,
    )
    route_target = resolve_tree_boosting_router_target(family_spec)
    concrete_cfg = dict(cfg)
    concrete_cfg["family_spec"] = family_spec
    route_builders = {
        "xgboost": _build_xgboost_trainer,
    }
    route_builder = route_builders.get(route_target)
    if route_builder is None:
        raise ValueError(
            "unsupported tree_boosting route target "
            f"'{route_target}' resolved for trainer_key='{TREE_BOOSTING_FORMAL_PRESET_KEY}'"
        )
    trainer = route_builder(
        pipeline=pipeline,
        biases=biases,
        config=concrete_cfg,
    )
    return _attach_tree_boosting_family(trainer, family_spec)


def _build_tree_ensemble_family_trainer(
    *,
    pipeline: Any,
    biases: Any,
    config: Dict[str, Any] | object | None = None,
):
    family_spec, cfg = _resolve_tree_family_from_config(
        config,
        trainer_key=TREE_ENSEMBLE_FORMAL_PRESET_KEY,
        default_builder=build_unified_tree_ensemble_family_spec,
    )
    route_target = resolve_tree_ensemble_router_target(family_spec)
    concrete_cfg = dict(cfg)
    concrete_cfg["family_spec"] = family_spec
    route_builders = {
        "random_forest": _build_random_forest_trainer,
        "extra_trees": _build_extra_trees_trainer,
        "bagging": _build_bagging_trainer,
        "adaboost": _build_adaboost_trainer,
    }
    route_builder = route_builders.get(route_target)
    if route_builder is None:
        raise ValueError(
            "unsupported tree_ensemble route target "
            f"'{route_target}' resolved for trainer_key='{TREE_ENSEMBLE_FORMAL_PRESET_KEY}'"
        )
    trainer = route_builder(
        pipeline=pipeline,
        biases=biases,
        config=concrete_cfg,
    )
    return _attach_tree_family(trainer, family_spec)


def _build_tree_family_trainer(
    *,
    trainer_key: str,
    trainer_module: str,
    trainer_class_name: str,
    trainer_config_name: str,
    default_family_builder: Any,
    default_artifact_id: str,
    pipeline: Any,
    biases: Any,
    config: Dict[str, Any] | object | None = None,
):
    try:
        module = __import__(trainer_module, fromlist=[trainer_class_name, trainer_config_name])
        trainer_cls = getattr(module, trainer_class_name)
        trainer_config_cls = getattr(module, trainer_config_name)
    except Exception as exc:
        raise ImportError(
            f"{trainer_key} trainer requires scikit-learn. Install scikit-learn before using trainer_key='{trainer_key}'."
        ) from exc

    if config is None:
        family_spec = default_family_builder(trainer_key=trainer_key)
        trainer_cfg = trainer_config_cls(family_spec=family_spec)
        numericizer = None
        modality_encoders = None
        target_codecs = None
        target_codec = None
        categorical_unknown = None
    elif isinstance(config, trainer_config_cls):
        trainer_cfg = config
        family_spec = coerce_tree_family_spec(getattr(config, "family_spec", None), trainer_key=trainer_key)
        numericizer = None
        modality_encoders = None
        target_codecs = None
        target_codec = None
        categorical_unknown = None
    elif isinstance(config, TreeTrainerFamilySpec):
        family_spec = config
        trainer_cfg = trainer_config_cls(family_spec=family_spec)
        numericizer = None
        modality_encoders = None
        target_codecs = None
        target_codec = None
        categorical_unknown = None
    elif isinstance(config, dict):
        cfg, numericizer, modality_encoders, target_codecs, target_codec, categorical_unknown = _split_numericizer_options(config)
        family_spec, cfg = _resolve_tree_family_from_config(
            cfg,
            trainer_key=trainer_key,
            default_builder=default_family_builder,
        )
        trainer_cfg = trainer_config_cls(
            artifact_id=str(cfg.pop("artifact_id", default_artifact_id)),
            family_spec=family_spec,
            resume_training_from=cfg.pop("resume_training_from", None),
            ood_z_threshold=float(cfg.pop("ood_z_threshold", 4.0)),
            mechanisms=tuple(cfg.pop("mechanisms", trainer_config_cls().mechanisms)),
        )
        if cfg:
            unknown = ", ".join(sorted(str(k) for k in cfg.keys()))
            raise TypeError(f"unknown {trainer_key} trainer config fields: {unknown}")
    else:
        raise TypeError(
            f"trainer config must be dict, TreeTrainerFamilySpec, {trainer_config_name}, or None"
        )

    trainer = trainer_cls(
        config=trainer_cfg,
        pipeline=pipeline,
        biases=biases,
        numericizer=numericizer,
        modality_encoders=modality_encoders,
        target_codecs=target_codecs,
        target_codec=target_codec,
        categorical_unknown=("error" if categorical_unknown is None else str(categorical_unknown)),
    )
    return _attach_tree_family(trainer, family_spec)


def _build_random_forest_trainer(*, pipeline: Any, biases: Any, config: Dict[str, Any] | object | None = None):
    return _build_tree_family_trainer(
        trainer_key="random_forest",
        trainer_module="core.trainers.random_forest_trainer",
        trainer_class_name="RandomForestSurrogateTrainer",
        trainer_config_name="RandomForestTrainerConfig",
        default_family_builder=build_random_forest_family_spec,
        default_artifact_id="random_forest_surrogate_v1",
        pipeline=pipeline,
        biases=biases,
        config=config,
    )


def _build_extra_trees_trainer(*, pipeline: Any, biases: Any, config: Dict[str, Any] | object | None = None):
    return _build_tree_family_trainer(
        trainer_key="extra_trees",
        trainer_module="core.trainers.extra_trees_trainer",
        trainer_class_name="ExtraTreesSurrogateTrainer",
        trainer_config_name="ExtraTreesTrainerConfig",
        default_family_builder=build_extra_trees_family_spec,
        default_artifact_id="extra_trees_surrogate_v1",
        pipeline=pipeline,
        biases=biases,
        config=config,
    )


def _build_bagging_trainer(*, pipeline: Any, biases: Any, config: Dict[str, Any] | object | None = None):
    return _build_tree_family_trainer(
        trainer_key="bagging",
        trainer_module="core.trainers.bagging_trainer",
        trainer_class_name="BaggingSurrogateTrainer",
        trainer_config_name="BaggingTrainerConfig",
        default_family_builder=build_bagging_family_spec,
        default_artifact_id="bagging_surrogate_v1",
        pipeline=pipeline,
        biases=biases,
        config=config,
    )


def _build_adaboost_trainer(*, pipeline: Any, biases: Any, config: Dict[str, Any] | object | None = None):
    return _build_tree_family_trainer(
        trainer_key="adaboost",
        trainer_module="core.trainers.adaboost_trainer",
        trainer_class_name="AdaBoostSurrogateTrainer",
        trainer_config_name="AdaBoostTrainerConfig",
        default_family_builder=build_adaboost_family_spec,
        default_artifact_id="adaboost_surrogate_v1",
        pipeline=pipeline,
        biases=biases,
        config=config,
    )


def create_default_config() -> MLBlackConfig:
    cfg = MLBlackConfig()

    # pipeline
    cfg.pipelines.register(
        "identity",
        lambda **_: IdentityPipeline(),
        metadata={
            "name": "identity",
            "kind": "pipeline",
            "purpose": "no-op transform",
        },
    )
    cfg.pipelines.register(
        "zscore",
        lambda **kwargs: ZScorePipeline(**kwargs),
        metadata={
            "name": "zscore",
            "kind": "pipeline",
            "purpose": "standardization",
        },
    )

    # bias
    cfg.biases.register(
        "noop",
        lambda **_: NoOpBias(),
        metadata={
            "name": "noop",
            "kind": "bias",
            "purpose": "no-op bias layer",
        },
    )
    cfg.biases.register(
        "l2_scale",
        lambda scale=1.0, **_: L2ScaleBias(scale=float(scale)),
        metadata={
            "name": "l2_scale",
            "kind": "bias",
            "purpose": "scale effective l2 strength",
            "params": {"scale": "float"},
        },
    )

    # numericizer
    cfg.numericizers.register(
        "default",
        lambda **kwargs: DefaultNumericizer(**kwargs),
        metadata={
            "name": "default",
            "kind": "numericizer",
            "purpose": "strong-typed sample->numeric encoder",
            "supports": {
                "processed_dataset_passthrough": True,
                "sample_dataset_encoding": True,
                "target_codec": True,
            },
        },
    )

    # capability
    cfg.capabilities.register(
        "noop",
        _build_noop_flow_capability,
        metadata={
            "name": "noop",
            "kind": "capability",
            "purpose": "flow lifecycle no-op capability",
            "params": {
                "name": "str",
                "priority": "int",
                "enabled": "bool",
                "is_algorithmic": "bool",
                "config": "dict",
                "context_requires": "sequence[str]",
                "context_provides": "sequence[str]",
                "context_mutates": "sequence[str]",
                "context_cache": "sequence[str]",
                "context_notes": "str|None",
            },
        },
    )
    cfg.capabilities.register(
        "metric_guard",
        _build_metric_guard_capability,
        metadata={
            "name": "metric_guard",
            "kind": "capability",
            "purpose": "post-eval metric threshold gate and report annotation",
            "params": {
                "rules": "sequence[{split,metric,op,threshold}]",
                "hard_fail": "bool",
                "report_key": "str",
                "name": "str",
                "priority": "int",
                "enabled": "bool",
                "is_algorithmic": "bool",
                "config": "dict",
            },
        },
    )
    cfg.capabilities.register(
        "experiment_tracker",
        build_experiment_tracker_capability,
        metadata={
            "name": "experiment_tracker",
            "kind": "capability",
            "purpose": "persist flow lifecycle events and metrics into sqlite for visualization",
            "params": {
                "db_path": "str",
                "namespace": "str",
                "tag": "str|None",
                "report_key": "str",
                "max_payload_chars": "int",
                "io_mode": "str(legacy|batched|safe)",
                "commit_interval": "int",
                "name": "str",
                "priority": "int",
                "enabled": "bool",
                "is_algorithmic": "bool",
                "config": "dict",
            },
        },
    )

    # trainer
    cfg.trainers.register(
        "linear",
        _build_linear_family_trainer,
        metadata={
            "name": "linear",
            "family": "linear",
            "backend": "family_router",
            "surface_status": "formal",
            "route_family": "linear",
            "route_registry": serialize_family_route_registry(linear_route_registry()),
            "nonlinear": False,
            "mechanism_bindings": serialize_family_bindings(build_linear_family_mechanism_bindings()),
            "supports": {
                "processed_dataset": True,
                "sample_dataset": True,
                "target_codec": True,
                "categorical_feature_one_hot": True,
                "family_spec": True,
                "parameter_backend": "closed_form",
                "runtime_backend": "numpy",
                "solver_kind": "ridge",
                "task_head": "point",
            },
            "runtime": {
                "note": "formal linear family entry that routes family specs to concrete linear presets",
            },
        },
    )
    cfg.trainers.register(
        "ridge",
        _build_ridge_trainer,
        metadata={
            "name": "ridge",
            "family": "linear",
            "backend": "numpy",
            "surface_status": "route_target",
            "route_family": "linear",
            "route_registry": serialize_family_route_registry(linear_route_registry()),
            "nonlinear": False,
            "mechanism_bindings": serialize_family_bindings(build_linear_family_mechanism_bindings()),
            "supports": {
                "processed_dataset": True,
                "sample_dataset": True,
                "target_codec": True,
                "categorical_feature_one_hot": True,
                "family_spec": True,
                "backend_component": "closed_form_numpy",
                "function_component": "affine_intercept",
                "regularization_component": "penalty|l2",
                "task_head_component": "point",
            },
        },
    )
    cfg.trainers.register(
        "neural",
        _build_neural_family_trainer,
        metadata={
            "name": "neural",
            "family": "neural",
            "backend": "family_router",
            "surface_status": "formal",
            "route_family": "neural",
            "route_registry": serialize_family_route_registry(neural_route_registry()),
            "nonlinear": True,
            "mechanism_bindings": serialize_family_bindings(build_neural_family_mechanism_bindings()),
            "supports": {
                "processed_dataset": True,
                "sample_dataset": True,
                "target_codec": True,
                "categorical_feature_one_hot": True,
                "family_spec": True,
                "parameter_backend": "pytorch|sklearn",
                "trainer_kind": "mlp",
                "task_head": "point",
            },
            "runtime": {
                "note": "formal neural family entry that routes backend/runtime variants to concrete neural presets",
            },
        },
    )
    cfg.trainers.register(
        "mlp_torch",
        _build_torch_mlp_trainer,
        metadata={
            "name": "mlp_torch",
            "family": "neural_network",
            "backend": "pytorch",
            "surface_status": "route_target",
            "route_family": "neural",
            "route_registry": serialize_family_route_registry(neural_route_registry()),
            "nonlinear": True,
            "mechanism_bindings": serialize_family_bindings(build_neural_family_mechanism_bindings()),
            "supports": {
                "processed_dataset": True,
                "sample_dataset": True,
                "target_codec": True,
                "categorical_feature_one_hot": True,
            },
            "runtime": {
                "requires_torch": True,
                "device": "auto|cpu|cuda|cuda:<index>",
            },
        },
    )
    cfg.trainers.register(
        "sklearn_mlp",
        _build_sklearn_mlp_trainer,
        metadata={
            "name": "sklearn_mlp",
            "family": "neural_network",
            "backend": "scikit-learn",
            "surface_status": "route_target",
            "route_family": "neural",
            "route_registry": serialize_family_route_registry(neural_route_registry()),
            "nonlinear": True,
            "mechanism_bindings": serialize_family_bindings(build_neural_family_mechanism_bindings()),
            "supports": {
                "processed_dataset": True,
                "sample_dataset": True,
                "sample_weight": False,
                "target_codec": True,
                "categorical_feature_one_hot": True,
                "family_spec": True,
            },
            "runtime": {
                "requires": "sklearn",
            },
        },
    )
    cfg.trainers.register(
        "symbolic",
        _build_symbolic_family_trainer,
        metadata={
            "name": "symbolic",
            "family": "symbolic_unified",
            "route_family": "symbolic",
            "backend": "family_router",
            "surface_status": "formal",
            "canonical_preset": SYMBOLIC_FORMAL_PRESET_KEY,
            "legacy_facades": ("symbolic_stagewise", "symbolic_torch", "symbolic_torch_interval"),
            "nonlinear": True,
            "route_registry": serialize_family_route_registry(
                tuple(route.as_family_route_spec() for route in symbolic_route_registry())
            ),
            "mechanism_bindings": serialize_family_bindings(build_symbolic_family_mechanism_bindings()),
            "search_mechanism_contracts": serialize_symbolic_search_mechanism_contracts(
                build_symbolic_search_mechanism_contracts()
            ),
            "supports": {
                "processed_dataset": True,
                "sample_dataset": True,
                "target_codec": True,
                "categorical_feature_one_hot": True,
                "family_spec": True,
                "structure_engine": "stagewise_search|orthogonal_basis_search|seed_library|explicit_genome",
                "parameter_backend": "ridge|torch",
                "task_head": "point|interval",
            },
            "runtime": {
                "note": "formal symbolic family entry that routes backend/task/head combinations to concrete symbolic trainers",
            },
        },
    )
    cfg.trainers.register(
        "symbolic_torch",
        _build_symbolic_torch_trainer,
        metadata={
            "name": "symbolic_torch",
            "family": "symbolic_hybrid",
            "backend": "pytorch",
            "surface_status": "deprecated",
            "canonical_preset": canonical_symbolic_preset_key("symbolic_torch"),
            "deprecated_replacement": SYMBOLIC_FORMAL_PRESET_KEY,
            "nonlinear": True,
            "mechanism_bindings": serialize_family_bindings(build_symbolic_family_mechanism_bindings()),
            "search_mechanism_contracts": serialize_symbolic_search_mechanism_contracts(
                build_symbolic_search_mechanism_contracts()
            ),
            "supports": {
                "processed_dataset": True,
                "sample_dataset": True,
                "sample_weight": True,
                "target_codec": True,
                "categorical_feature_one_hot": True,
                "dynamic_function_genome": True,
                "v2_interaction_terms": True,
                "v2_piecewise_hinge_terms": True,
            },
            "runtime": {
                "requires": "torch",
                "device": "auto|cpu|cuda|cuda:<index>",
                "note": "deprecated symbolic facade; prefer trainer_key='symbolic' with parameter_backend='torch' and task='point'",
            },
        },
    )
    cfg.trainers.register(
        "symbolic_orthogonal",
        _build_symbolic_orthogonal_trainer,
        metadata={
            "name": "symbolic_orthogonal",
            "family": "symbolic_unified",
            "route_family": "symbolic",
            "route_target": "symbolic_orthogonal",
            "backend": "ridge",
            "surface_status": "stable",
            "canonical_preset": SYMBOLIC_FORMAL_PRESET_KEY,
            "nonlinear": True,
            "route_registry": serialize_family_route_registry(
                tuple(route.as_family_route_spec() for route in symbolic_route_registry())
            ),
            "mechanism_bindings": serialize_family_bindings(build_symbolic_family_mechanism_bindings()),
            "search_mechanism_contracts": serialize_symbolic_search_mechanism_contracts(
                build_symbolic_search_mechanism_contracts()
            ),
            "supports": {
                "processed_dataset": True,
                "sample_dataset": True,
                "target_codec": True,
                "categorical_feature_one_hot": True,
                "family_spec": True,
                "structure_engine": "orthogonal_basis_search",
                "parameter_backend": "ridge",
                "task_head": "point",
                "piecewise_gate_basis": True,
                "trainer_state": True,
            },
            "runtime": {
                "note": "concrete symbolic orthogonal route that discovers relative orthogonal basis groups before the final small-budget symbolic assembly",
            },
        },
    )
    cfg.trainers.register(
        "symbolic_stagewise",
        _build_symbolic_stagewise_trainer,
        metadata={
            "name": "symbolic_stagewise",
            "family": "symbolic_stagewise",
            "backend": "numpy",
            "surface_status": "deprecated",
            "canonical_preset": canonical_symbolic_preset_key("symbolic_stagewise"),
            "deprecated_replacement": SYMBOLIC_FORMAL_PRESET_KEY,
            "nonlinear": True,
            "mechanism_bindings": serialize_family_bindings(build_symbolic_family_mechanism_bindings()),
            "search_mechanism_contracts": serialize_symbolic_search_mechanism_contracts(
                build_symbolic_search_mechanism_contracts()
            ),
            "supports": {
                "processed_dataset": True,
                "sample_dataset": True,
                "target_codec": True,
                "categorical_feature_one_hot": True,
                "linear_floor": True,
                "residual_increment_search": True,
                "path_memory": True,
                "add_and_drop_terms": True,
                "tabu_prior_rerank": True,
                "expression_export": True,
            },
            "runtime": {
                "requires": "numpy",
                "note": "deprecated symbolic facade; prefer trainer_key='symbolic' with parameter_backend='ridge' and task='point'",
            },
        },
    )
    cfg.trainers.register(
        "symbolic_torch_interval",
        _build_symbolic_torch_interval_trainer,
        metadata={
            "name": "symbolic_torch_interval",
            "family": "symbolic_interval",
            "backend": "pytorch",
            "surface_status": "deprecated",
            "canonical_preset": canonical_symbolic_preset_key("symbolic_torch_interval"),
            "deprecated_replacement": SYMBOLIC_FORMAL_PRESET_KEY,
            "nonlinear": True,
            "mechanism_bindings": serialize_family_bindings(build_symbolic_family_mechanism_bindings()),
            "search_mechanism_contracts": serialize_symbolic_search_mechanism_contracts(
                build_symbolic_search_mechanism_contracts()
            ),
            "supports": {
                "processed_dataset": True,
                "sample_dataset": True,
                "sample_weight": True,
                "target_codec": True,
                "categorical_feature_one_hot": True,
                "dynamic_function_genome": True,
                "v2_interaction_terms": True,
                "v2_piecewise_hinge_terms": True,
                "interval_output": True,
            },
            "runtime": {
                "requires": "torch",
                "device": "auto|cpu|cuda|cuda:<index>",
                "note": "deprecated symbolic facade; prefer trainer_key='symbolic' with parameter_backend='torch' and task='interval'",
            },
        },
    )
    cfg.trainers.register(
        "tree_boosting",
        _build_tree_boosting_family_trainer,
        metadata={
            "name": "tree_boosting",
            "family": "tree_boosting",
            "backend": "family_router",
            "surface_status": "formal",
            "route_family": "tree_boosting",
            "route_registry": serialize_family_route_registry(tree_boosting_route_registry()),
            "nonlinear": True,
            "mechanism_bindings": serialize_family_bindings(build_tree_boosting_family_mechanism_bindings()),
            "supports": {
                "processed_dataset": True,
                "sample_dataset": True,
                "target_codec": True,
                "categorical_feature_one_hot": True,
                "family_spec": True,
                "backend": "xgboost",
                "booster": "gbtree",
                "task_head": "point",
            },
            "runtime": {
                "note": "formal tree_boosting family entry that routes backend variants to concrete boosting presets",
            },
        },
    )
    cfg.trainers.register(
        "tree_ensemble",
        _build_tree_ensemble_family_trainer,
        metadata={
            "name": "tree_ensemble",
            "family": "tree_ensemble",
            "backend": "family_router",
            "surface_status": "formal",
            "route_family": "tree_ensemble",
            "route_registry": serialize_family_route_registry(tree_ensemble_route_registry()),
            "nonlinear": True,
            "mechanism_bindings": serialize_family_bindings(build_random_forest_mechanism_bindings()),
            "supports": {
                "processed_dataset": True,
                "sample_dataset": True,
                "target_codec": True,
                "categorical_feature_one_hot": True,
                "family_spec": True,
                "ensemble_kind": "random_forest|extra_trees|bagging|adaboost",
                "runtime_backend": "scikit-learn",
                "task_head": "point",
            },
            "runtime": {
                "note": "formal tree_ensemble family entry that routes ensemble-style variants to concrete tree presets",
            },
        },
    )
    cfg.trainers.register(
        "xgboost",
        _build_xgboost_trainer,
        metadata={
            "name": "xgboost",
            "family": "tree_boosting",
            "backend": "xgboost",
            "surface_status": "route_target",
            "route_family": "tree_boosting",
            "route_registry": serialize_family_route_registry(tree_boosting_route_registry()),
            "nonlinear": True,
            "mechanism_bindings": serialize_family_bindings(build_tree_boosting_family_mechanism_bindings()),
            "supports": {
                "processed_dataset": True,
                "sample_dataset": True,
                "sample_weight": True,
                "target_codec": True,
                "categorical_feature_one_hot": True,
                "family_spec": True,
                "backend_component": "xgboost_gbtree",
                "boosting_component": "n_estimators|learning_rate|objective|tree_method",
                "sampling_component": "subsample|colsample_bytree",
                "regularization_component": "max_depth|min_child_weight|gamma|reg_lambda|reg_alpha",
            },
            "runtime": {
                "requires": ["xgboost", "sklearn"],
                "tree_method": "hist|approx|exact|gpu_hist",
                "continuation_via": "xgb_model",
            },
        },
    )
    cfg.trainers.register(
        "random_forest",
        _build_random_forest_trainer,
        metadata={
            "name": "random_forest",
            "family": "tree_ensemble",
            "backend": "scikit-learn",
            "surface_status": "route_target",
            "route_family": "tree_ensemble",
            "route_registry": serialize_family_route_registry(tree_ensemble_route_registry()),
            "nonlinear": True,
            "mechanism_bindings": serialize_family_bindings(build_random_forest_mechanism_bindings()),
            "supports": {
                "processed_dataset": True,
                "sample_dataset": True,
                "sample_weight": True,
                "target_codec": True,
                "categorical_feature_one_hot": True,
                "family_spec": True,
                "ensemble_component": "random_forest",
                "sampling_component": "bootstrap|max_samples|max_features",
                "splitter_component": "criterion|min_impurity_decrease",
                "regularization_component": "max_depth|min_samples_split|min_samples_leaf|max_leaf_nodes|ccp_alpha",
            },
            "runtime": {
                "requires": "sklearn",
                "continuation_via": "warm_start_append",
                "task_head": "point",
            },
        },
    )
    cfg.trainers.register(
        "extra_trees",
        _build_extra_trees_trainer,
        metadata={
            "name": "extra_trees",
            "family": "tree_ensemble",
            "backend": "scikit-learn",
            "surface_status": "route_target",
            "route_family": "tree_ensemble",
            "route_registry": serialize_family_route_registry(tree_ensemble_route_registry()),
            "nonlinear": True,
            "mechanism_bindings": serialize_family_bindings(build_extra_trees_mechanism_bindings()),
            "supports": {
                "processed_dataset": True,
                "sample_dataset": True,
                "sample_weight": True,
                "target_codec": True,
                "categorical_feature_one_hot": True,
                "family_spec": True,
                "ensemble_component": "extra_trees",
                "sampling_component": "max_features|optional_bootstrap|max_samples",
                "splitter_component": "criterion|random_split|min_impurity_decrease",
                "regularization_component": "max_depth|min_samples_split|min_samples_leaf|max_leaf_nodes|ccp_alpha",
            },
            "runtime": {
                "requires": "sklearn",
                "continuation_via": "warm_start_append",
                "task_head": "point",
            },
        },
    )
    cfg.trainers.register(
        "bagging",
        _build_bagging_trainer,
        metadata={
            "name": "bagging",
            "family": "tree_ensemble",
            "backend": "scikit-learn",
            "surface_status": "route_target",
            "route_family": "tree_ensemble",
            "route_registry": serialize_family_route_registry(tree_ensemble_route_registry()),
            "nonlinear": True,
            "mechanism_bindings": serialize_family_bindings(build_bagging_mechanism_bindings()),
            "supports": {
                "processed_dataset": True,
                "sample_dataset": True,
                "sample_weight": True,
                "target_codec": True,
                "categorical_feature_one_hot": True,
                "family_spec": True,
                "ensemble_component": "bagging",
                "sampling_component": "bootstrap|bootstrap_features|max_samples|max_features",
                "splitter_component": "criterion|splitter",
                "regularization_component": "max_depth|min_samples_split|min_samples_leaf|max_leaf_nodes|ccp_alpha",
            },
            "runtime": {
                "requires": "sklearn",
                "continuation_via": "warm_start_append",
                "task_head": "point",
            },
        },
    )
    cfg.trainers.register(
        "adaboost",
        _build_adaboost_trainer,
        metadata={
            "name": "adaboost",
            "family": "tree_ensemble",
            "backend": "scikit-learn",
            "surface_status": "route_target",
            "route_family": "tree_ensemble",
            "route_registry": serialize_family_route_registry(tree_ensemble_route_registry()),
            "nonlinear": True,
            "mechanism_bindings": serialize_family_bindings(build_adaboost_mechanism_bindings()),
            "supports": {
                "processed_dataset": True,
                "sample_dataset": True,
                "sample_weight": True,
                "target_codec": True,
                "categorical_feature_one_hot": True,
                "family_spec": True,
                "ensemble_component": "adaboost",
                "sample_weighting_component": "defining_round_reweighting",
                "aggregation_component": "weighted_additive",
                "base_estimator_component": "decision_tree",
            },
            "runtime": {
                "requires": "sklearn",
                "continuation_via": "fresh_only",
                "task_head": "point",
            },
        },
    )

    return cfg













