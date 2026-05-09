from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

NSGABLACK_ROOT = ROOT.parent / "nsgablack"
if str(NSGABLACK_ROOT) not in sys.path:
    sys.path.append(str(NSGABLACK_ROOT))

from bias import (
    BranchPolicyConfig,
    DynamicPoolPolicyConfig,
    ObjectivePolicyConfig,
    build_dynamic_activation_policy,
    resolve_branch_policy,
    resolve_branch_workers,
)
from conditional import (
    AutoThresholdBindingSpec,
    BinaryGateBinding,
    ConditionalComposerConfig,
    ConditionalConfig,
    build_auto_threshold_bindings,
    build_feature_role_config,
)
from core.symbolic.expression_graph_cache import ExpressionGraphCache
from examples.work_ci_reader import WorkCiIntervalReader
from nowcasting_work_ci.mlblack_side.config import build_output_root
from nowcasting_work_ci.mlblack_side.problem.domain_router import (
    build_work_ci_branch_policy,
    build_work_ci_conditional_router_policy,
)
from pipeline import FeatureSpaceBuilderConfig, FeatureSpaceBuildInput, build_feature_space
from pipeline.feature_space import (
    CandidatePoolConfig,
    DynamicActivationConfig,
    FeatureEngineeringConfig,
    RegimeFeaturePackConfig,
    TemporalFeaturePackConfig,
    parse_float_list_csv,
    parse_int_list_csv,
)

from .config import RuntimeCliConfig


def _build_dynamic_activation_spec(cfg: DynamicPoolPolicyConfig) -> DynamicActivationConfig:
    return DynamicActivationConfig(
        unary_top_k=int(max(1, cfg.unary_top_k)),
        pair_top_k=int(max(1, cfg.pair_top_k)),
        gate_top_k=int(max(1, cfg.gate_top_k)),
        recursive_depth=int(max(1, cfg.recursive_depth)),
        recursive_seed_top_k=int(max(1, cfg.recursive_seed_top_k)),
        recursive_pair_seed_top_k=int(max(1, cfg.recursive_pair_seed_top_k)),
        recursive_max_complexity=float(max(3.0, cfg.recursive_max_complexity)),
        allow_trig=bool(cfg.allow_trig),
        allow_safe_exp=bool(cfg.allow_safe_exp),
        allow_safe_log=bool(cfg.allow_safe_log),
        allow_safe_ratio=bool(cfg.allow_safe_ratio),
        family_budget_csv=str(cfg.family_budget_csv),
    )


def reg_data(args: RuntimeCliConfig) -> dict[str, Any]:
    out_root = build_output_root(ROOT, seed=int(args.seed))
    reader = WorkCiIntervalReader(
        csv_path=str(args.csv_path),
        target_col=str(args.target_col),
        test_fold_col=str(args.test_fold_col),
    )
    bundle = reader.read()
    tr = bundle.train
    te = bundle.test
    if te is None:
        raise ValueError("no test split in reader output")
    return {
        "out_root": out_root,
        "train": tr,
        "test": te,
    }


def reg_dynamic_pool(args: RuntimeCliConfig) -> dict[str, Any]:
    policy = DynamicPoolPolicyConfig(
        enabled=bool(int(args.dynamic_pool_enabled)),
        epochs=int(max(1, args.dynamic_pool_epochs)),
        init_minimal=bool(int(args.dynamic_init_minimal)),
        expand_max_new=int(max(1, args.dynamic_expand_max_new)),
        focus_top_features=int(max(2, args.dynamic_focus_top_features)),
        partner_topk=int(max(2, args.dynamic_partner_topk)),
        top_cache_use=int(max(5, args.dynamic_top_cache_use)),
        max_pool_size=int(max(32, args.dynamic_max_pool_size)),
        unary_top_k=int(max(1, args.dynamic_unary_top_k)),
        pair_top_k=int(max(1, args.dynamic_pair_top_k)),
        gate_top_k=int(max(1, args.dynamic_gate_top_k)),
        recursive_depth=int(max(1, args.dynamic_recursive_depth)),
        recursive_seed_top_k=int(max(1, args.dynamic_recursive_seed_top_k)),
        recursive_pair_seed_top_k=int(max(1, args.dynamic_recursive_pair_seed_top_k)),
        recursive_max_complexity=float(max(3.0, args.dynamic_recursive_max_complexity)),
        allow_trig=bool(int(args.dynamic_allow_trig)),
        allow_safe_exp=bool(int(args.dynamic_allow_safe_exp)),
        allow_safe_log=bool(int(args.dynamic_allow_safe_log)),
        allow_safe_ratio=bool(int(args.dynamic_allow_safe_ratio)),
        family_budget_csv=str(args.dynamic_family_budget_csv),
    )
    activation_spec = _build_dynamic_activation_spec(policy)
    activation_cfg = build_dynamic_activation_policy(policy)
    return {
        "policy": policy,
        "enabled": bool(policy.enabled),
        "activation_spec": activation_spec,
        "activation_cfg": activation_cfg,
    }


def reg_feature_space(
    args: RuntimeCliConfig,
    *,
    train: Any,
    test: Any,
    dynamic_runtime: dict[str, Any],
) -> dict[str, Any]:
    lag_orders = parse_int_list_csv(str(args.lag_orders), default=(1, 2, 3))
    lag_sources = [s.strip() for s in str(args.lag_sources).split(",") if s.strip()]
    lag_source_set = {s for s in set(lag_sources) if s in {"ci", "total_flow", "avg_speed", "avg_occ"}}
    lag_cross_q = [
        float(np.clip(v, 0.01, 0.99))
        for v in parse_float_list_csv(str(args.lag_cross_quantiles), default=(0.25, 0.5, 0.75))
    ]
    temporal_pack_cross_q = [
        float(np.clip(v, 0.01, 0.99))
        for v in parse_float_list_csv(str(args.temporal_pack_cross_quantiles), default=(0.5,))
    ]
    temporal_pack_ratio_eps = float(max(1e-8, args.temporal_pack_ratio_eps))
    regime_pack_shock_q = [
        float(np.clip(v, 0.5, 0.999))
        for v in parse_float_list_csv(str(args.regime_pack_shock_quantiles), default=(0.8, 0.9))
    ]
    regime_pack_ci_q = [
        float(np.clip(v, 0.01, 0.99))
        for v in parse_float_list_csv(str(args.regime_pack_ci_quantiles), default=(0.33, 0.66))
    ]
    regime_pack_eps = float(max(1e-10, args.regime_pack_eps))

    feature_space = build_feature_space(
        inputs=FeatureSpaceBuildInput(
            X_train=np.asarray(train.X_train, dtype=float),
            y_train=np.asarray(train.y_train, dtype=float).reshape(-1, 1),
            X_test=np.asarray(test.X_train, dtype=float),
            y_test=np.asarray(test.y_train, dtype=float).reshape(-1, 1),
            feature_names=tuple(str(v) for v in train.feature_names),
        ),
        cfg=FeatureSpaceBuilderConfig(
            feature_engineering=FeatureEngineeringConfig(
                lag_feature_enabled=bool(int(args.lag_feature_enabled)),
                lag_orders_csv=str(args.lag_orders),
                lag_sources_csv=str(args.lag_sources),
                lag_cross_enabled=bool(int(args.lag_cross_enabled)),
                lag_cross_quantiles_csv=str(args.lag_cross_quantiles),
                drop_same_day_flow_speed_occ=bool(int(args.drop_same_day_flow_speed_occ)),
                drop_feature_list_csv=str(args.drop_feature_list),
            ),
            temporal_pack=TemporalFeaturePackConfig(
                enabled=bool(int(args.temporal_pack_enabled)),
                rolling_enabled=bool(int(args.temporal_pack_rolling_enabled)),
                momentum_enabled=bool(int(args.temporal_pack_momentum_enabled)),
                cross_enabled=bool(int(args.temporal_pack_cross_enabled)),
                ratio_enabled=bool(int(args.temporal_pack_ratio_enabled)),
                cross_quantiles=tuple(temporal_pack_cross_q),
                safe_ratio_eps=float(temporal_pack_ratio_eps),
            ),
            regime_pack=RegimeFeaturePackConfig(
                enabled=bool(int(args.regime_pack_enabled)),
                volatility_enabled=bool(int(args.regime_pack_volatility_enabled)),
                shock_enabled=bool(int(args.regime_pack_shock_enabled)),
                ci_regime_enabled=bool(int(args.regime_pack_ci_regime_enabled)),
                shock_quantiles=tuple(regime_pack_shock_q),
                ci_regime_quantiles=tuple(regime_pack_ci_q),
                eps=float(regime_pack_eps),
            ),
            candidate_pool=CandidatePoolConfig(
                dynamic_pool_enabled=bool(dynamic_runtime["enabled"]),
                dynamic_init_minimal=bool(dynamic_runtime["policy"].init_minimal),
                safe_log1p_abs=bool(int(args.safe_log1p_abs)),
                safe_exp_clip=bool(int(args.safe_exp_clip)),
                safe_reciprocal=bool(int(args.safe_reciprocal)),
                safe_exp_clip_k=float(max(1.0, args.safe_exp_clip_k)),
                safe_reciprocal_eps=float(max(1e-8, args.safe_reciprocal_eps)),
                dynamic_activation=dynamic_runtime["activation_spec"],
            ),
            build_full_candidate_pool=bool(not dynamic_runtime["enabled"]),
        ),
    )

    return {
        "feature_space": feature_space,
        "feature_bundle": feature_space.feature_bundle,
        "temporal_pack": feature_space.temporal_pack_result,
        "regime_pack": feature_space.regime_pack_result,
        "candidates": list(feature_space.candidates),
        "lag_orders": lag_orders,
        "lag_source_set": lag_source_set,
        "lag_cross_q": lag_cross_q,
        "temporal_pack_cross_q": temporal_pack_cross_q,
        "temporal_pack_ratio_eps": temporal_pack_ratio_eps,
        "regime_pack_shock_q": regime_pack_shock_q,
        "regime_pack_ci_q": regime_pack_ci_q,
        "regime_pack_eps": regime_pack_eps,
    }


def reg_branch(args: RuntimeCliConfig, *, feature_names: tuple[str, ...]) -> dict[str, Any]:
    policy = build_work_ci_branch_policy(
        enabled=bool(args.strict4_branch_mode),
        min_branch_train=int(max(8, args.strict4_min_branch_train)),
        parallel_workers=int(max(1, args.strict4_branch_parallel_workers)),
    )
    resolution = resolve_branch_policy(feature_names, cfg=policy)
    return {
        "policy": policy,
        "resolution": resolution,
        "strict4_enabled": bool(resolution.enabled),
    }


def _match_threshold_multiplier_feature(feature_name: str, feature_names: tuple[str, ...]) -> str | None:
    name = str(feature_name)
    feature_set = set(str(v) for v in feature_names)
    if name.startswith("ci_lag"):
        suffix = name[len("ci_lag") :]
        candidate = f"avg_speed_lag{suffix}"
        if candidate in feature_set:
            return candidate
    return None


def _build_default_threshold_specs(feature_names: tuple[str, ...]) -> dict[str, AutoThresholdBindingSpec]:
    out: dict[str, AutoThresholdBindingSpec] = {}
    for feature_name in feature_names:
        name = str(feature_name)
        multiplier = _match_threshold_multiplier_feature(name, feature_names)
        if name.startswith("ci_lag"):
            out[name] = AutoThresholdBindingSpec(
                primitive_family="hinge",
                quantiles=(0.33, 0.66),
                directions=("positive", "negative"),
                multiplier_feature=multiplier,
                min_unique_values=6,
                min_cut_separation_ratio=0.08,
            )
        elif name.startswith("shock_"):
            out[name] = AutoThresholdBindingSpec(
                primitive_family="hinge",
                quantiles=(0.80, 0.90),
                directions=("positive",),
                min_unique_values=6,
                min_cut_separation_ratio=0.08,
            )
        elif name.startswith("volatility_"):
            out[name] = AutoThresholdBindingSpec(
                primitive_family="hinge",
                quantiles=(0.75, 0.90),
                directions=("positive",),
                min_unique_values=6,
                min_cut_separation_ratio=0.08,
            )
    return out


def reg_conditional(
    args: RuntimeCliConfig,
    *,
    feature_names: tuple[str, ...],
    branch_runtime: dict[str, Any],
    X_train: np.ndarray | None = None,
) -> dict[str, Any]:
    feature_names_norm = tuple(str(v) for v in feature_names)
    router_policy = build_work_ci_conditional_router_policy() if bool(branch_runtime["strict4_enabled"]) else None
    router_features = (
        tuple(str(v) for v in router_policy.gate_names if str(v) in feature_names_norm)
        if router_policy is not None
        else tuple()
    )
    threshold_features = tuple(
        name
        for name in feature_names_norm
        if name.startswith("ci_lag") or name.startswith("shock_") or name.startswith("volatility_")
    )
    feature_roles = build_feature_role_config(
        feature_names_norm,
        router_features=router_features,
        threshold_features=threshold_features,
        metadata={
            "source": "nowcasting_work_ci_runtime_auto_roles",
            "strict4_enabled": bool(branch_runtime["strict4_enabled"]),
        },
    )
    threshold_primitives = (
        build_auto_threshold_bindings(
            np.asarray(X_train, dtype=float),
            feature_names_norm,
            threshold_features=threshold_features,
            per_feature_specs=_build_default_threshold_specs(feature_names_norm),
        )
        if X_train is not None and threshold_features
        else tuple()
    )
    cfg = ConditionalConfig(
        enabled=True,
        feature_roles=feature_roles,
        router_policy=router_policy,
        binary_gates=tuple(BinaryGateBinding(feature_name=name) for name in router_features),
        threshold_primitives=tuple(threshold_primitives),
        composer=ConditionalComposerConfig(
            mode="route_then_formula" if router_policy is not None else "formula_only_primitives",
        ),
        metadata={
            "source": "nowcasting_work_ci_runtime_defaults",
            "router_feature_count": int(len(router_features)),
            "threshold_feature_count": int(len(threshold_features)),
            "binary_gate_count": int(len(router_features)),
            "threshold_binding_count": int(len(threshold_primitives)),
        },
    )
    return {
        "config": cfg,
        "feature_roles": feature_roles,
        "role_index": feature_roles.role_index(),
        "composer_spec": cfg.composer_spec(),
        "router_policy": cfg.adapted_router_policy,
    }


def reg_budget(args: RuntimeCliConfig, *, branch_runtime: dict[str, Any]) -> dict[str, Any]:
    batched_eval_enabled = bool(int(args.batched_eval))
    reinvest_enabled = bool(int(args.reinvest_search))
    effective_pop_size = int(max(4, int(args.pop_size)))
    effective_generations = int(max(1, int(args.generations)))
    effective_strict4_workers = resolve_branch_workers(
        branch_runtime["policy"],
        reinvest_enabled=False,
        reinvest_mult=1.0,
    )
    if batched_eval_enabled and reinvest_enabled:
        effective_pop_size = int(max(effective_pop_size, round(effective_pop_size * float(max(1.0, args.reinvest_pop_mult)))))
        effective_generations = int(
            max(effective_generations, round(effective_generations * float(max(1.0, args.reinvest_gen_mult))))
        )
        if bool(branch_runtime["strict4_enabled"]):
            effective_strict4_workers = resolve_branch_workers(
                branch_runtime["policy"],
                reinvest_enabled=True,
                reinvest_mult=float(max(1.0, args.reinvest_strict4_workers_mult)),
            )
    effective_vns_batch_size = int(max(4, int(args.vns_batch_size), effective_pop_size))
    return {
        "batched_eval_enabled": batched_eval_enabled,
        "reinvest_enabled": reinvest_enabled,
        "effective_pop_size": effective_pop_size,
        "effective_generations": effective_generations,
        "effective_strict4_workers": effective_strict4_workers,
        "effective_vns_batch_size": effective_vns_batch_size,
    }


def reg_objective(args: RuntimeCliConfig) -> ObjectivePolicyConfig:
    return ObjectivePolicyConfig(
        coverage_error_threshold=float(max(0.0, args.selection_coverage_error_threshold)),
    )


def reg_graph_cache(args: RuntimeCliConfig) -> dict[str, Any]:
    enabled = bool(int(args.graph_cache_enabled))
    backend = str(args.graph_cache_backend).strip().lower()
    db_path = str(args.graph_cache_db_path).strip()
    if enabled and backend == "sqlite" and not db_path:
        db_path = str((ROOT / ".mlblack_cache" / "work_ci_subset_expression_graph_cache.sqlite3"))
    graph_cache = ExpressionGraphCache(
        enabled=bool(enabled),
        backend=str(backend),
        db_path=str(db_path),
        namespace=str(args.graph_cache_namespace),
        persist_values=bool(int(args.graph_cache_persist_values)),
    )
    return {
        "enabled": enabled,
        "backend": backend,
        "db_path": db_path,
        "graph_cache": graph_cache,
    }


def assemble_runtime_context(
    args: RuntimeCliConfig,
    *,
    data_runtime: dict[str, Any],
    feature_runtime: dict[str, Any],
    branch_runtime: dict[str, Any],
    conditional_runtime: dict[str, Any],
    budget_runtime: dict[str, Any],
    dynamic_runtime: dict[str, Any],
    objective_policy: ObjectivePolicyConfig,
    graph_cache_runtime: dict[str, Any],
) -> dict[str, Any]:
    feature_bundle = feature_runtime["feature_bundle"]
    temporal_pack = feature_runtime["temporal_pack"]
    regime_pack = feature_runtime["regime_pack"]
    return {
        "out_root": data_runtime["out_root"],
        "X_train": np.asarray(feature_bundle.X_train, dtype=float),
        "y_train": np.asarray(feature_bundle.y_train, dtype=float),
        "X_test": np.asarray(feature_bundle.X_test, dtype=float),
        "y_test": np.asarray(feature_bundle.y_test, dtype=float),
        "feature_names": tuple(str(v) for v in feature_bundle.feature_names),
        "n_features_raw": int(feature_bundle.n_features_raw),
        "feature_names_raw": tuple(str(v) for v in feature_bundle.feature_names_raw),
        "lag_added_features": [str(v) for v in feature_bundle.lag_added_features],
        "lag_cross_added_features": [str(v) for v in feature_bundle.lag_cross_added_features],
        "temporal_pack_added_features": [str(v) for v in temporal_pack.added_features],
        "temporal_pack_rolling_added": [str(v) for v in temporal_pack.rolling_added],
        "temporal_pack_momentum_added": [str(v) for v in temporal_pack.momentum_added],
        "temporal_pack_cross_added": [str(v) for v in temporal_pack.cross_added],
        "temporal_pack_ratio_added": [str(v) for v in temporal_pack.ratio_added],
        "regime_pack_added_features": [str(v) for v in regime_pack.added_features],
        "regime_pack_volatility_added": [str(v) for v in regime_pack.volatility_added],
        "regime_pack_shock_added": [str(v) for v in regime_pack.shock_added],
        "regime_pack_ci_regime_added": [str(v) for v in regime_pack.ci_regime_added],
        "lag_enabled": bool(int(args.lag_feature_enabled)),
        "lag_orders": feature_runtime["lag_orders"],
        "lag_source_set": feature_runtime["lag_source_set"],
        "lag_cross_enabled": bool(int(args.lag_cross_enabled)),
        "lag_cross_q": feature_runtime["lag_cross_q"],
        "dropped_features": [str(v) for v in feature_bundle.dropped_features],
        "temporal_pack_enabled": bool(int(args.temporal_pack_enabled)),
        "temporal_pack_cross_q": feature_runtime["temporal_pack_cross_q"],
        "temporal_pack_ratio_eps": feature_runtime["temporal_pack_ratio_eps"],
        "regime_pack_enabled": bool(int(args.regime_pack_enabled)),
        "regime_pack_shock_q": feature_runtime["regime_pack_shock_q"],
        "regime_pack_ci_q": feature_runtime["regime_pack_ci_q"],
        "regime_pack_eps": feature_runtime["regime_pack_eps"],
        "branch_policy": branch_runtime["policy"],
        "branch_resolution": branch_runtime["resolution"],
        "strict4_enabled": branch_runtime["strict4_enabled"],
        "conditional_config": conditional_runtime["config"],
        "conditional_feature_roles": conditional_runtime["feature_roles"],
        "conditional_feature_role_index": conditional_runtime["role_index"],
        "conditional_composer_spec": conditional_runtime["composer_spec"],
        "conditional_router_policy": conditional_runtime["router_policy"],
        "batched_eval_enabled": budget_runtime["batched_eval_enabled"],
        "reinvest_enabled": budget_runtime["reinvest_enabled"],
        "effective_pop_size": budget_runtime["effective_pop_size"],
        "effective_generations": budget_runtime["effective_generations"],
        "effective_strict4_workers": budget_runtime["effective_strict4_workers"],
        "effective_vns_batch_size": budget_runtime["effective_vns_batch_size"],
        "dynamic_pool_policy": dynamic_runtime["policy"],
        "dynamic_pool_enabled": dynamic_runtime["enabled"],
        "dynamic_pool_epochs": int(dynamic_runtime["policy"].epochs),
        "dynamic_init_minimal": bool(dynamic_runtime["policy"].init_minimal),
        "dynamic_expand_max_new": int(dynamic_runtime["policy"].expand_max_new),
        "dynamic_focus_top_features": int(dynamic_runtime["policy"].focus_top_features),
        "dynamic_partner_topk": int(dynamic_runtime["policy"].partner_topk),
        "dynamic_top_cache_use": int(dynamic_runtime["policy"].top_cache_use),
        "dynamic_max_pool_size": int(dynamic_runtime["policy"].max_pool_size),
        "dynamic_activation_cfg": dynamic_runtime["activation_cfg"],
        "objective_policy": objective_policy,
        "graph_cache_enabled": graph_cache_runtime["enabled"],
        "graph_cache_backend": graph_cache_runtime["backend"],
        "graph_cache_db_path": graph_cache_runtime["db_path"],
        "graph_cache": graph_cache_runtime["graph_cache"],
        "safe_log1p_abs_enabled": bool(int(args.safe_log1p_abs)),
        "safe_exp_clip_enabled": bool(int(args.safe_exp_clip)),
        "safe_reciprocal_enabled": bool(int(args.safe_reciprocal)),
        "safe_exp_clip_k": float(max(1.0, args.safe_exp_clip_k)),
        "safe_reciprocal_eps": float(max(1e-8, args.safe_reciprocal_eps)),
        "candidates": list(feature_runtime["candidates"]),
    }


__all__ = [
    "assemble_runtime_context",
    "reg_branch",
    "reg_budget",
    "reg_data",
    "reg_dynamic_pool",
    "reg_feature_space",
    "reg_graph_cache",
    "reg_objective",
]
