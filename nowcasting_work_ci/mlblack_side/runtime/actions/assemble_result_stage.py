from __future__ import annotations

from datetime import datetime
from typing import Any, Mapping

import numpy as np

from model.interval_fit import _jsonable

from ..config import RuntimeCliConfig
from ..contracts import (
    ComparisonStageResult,
    FinalStageResult,
    ResultSummaryPayload,
    RuntimeContextKey,
    SearchStageResult,
    SummaryReportPayload,
    ctx_require,
    ctx_set,
)


def _build_config_section(
    args: RuntimeCliConfig,
    prepared: Mapping[str, Any],
    *,
    interval_alpha: float,
    interval_method: str,
) -> dict[str, Any]:
    return {
        "csv_path": str(args.csv_path),
        "target_col": str(args.target_col),
        "test_fold_col": str(args.test_fold_col),
        "pop_size": int(args.pop_size),
        "generations": int(args.generations),
        "effective_pop_size": int(prepared["effective_pop_size"]),
        "effective_generations": int(prepared["effective_generations"]),
        "rolling_folds": int(args.rolling_folds),
        "rolling_val_ratio": float(args.rolling_val_ratio),
        "max_terms": int(args.max_terms),
        "ridge_l2": float(args.ridge_l2),
        "strict4_branch_mode_requested": bool(args.strict4_branch_mode),
        "strict4_branch_mode_enabled": bool(prepared["strict4_enabled"]),
        "strict4_min_branch_train": int(args.strict4_min_branch_train),
        "strict4_branch_parallel_workers": int(args.strict4_branch_parallel_workers),
        "effective_strict4_branch_parallel_workers": int(prepared["effective_strict4_workers"]),
        "seed": int(args.seed),
        "outer_strategy": str(args.outer_strategy),
        "portfolio_phases": str(args.portfolio_phases),
        "portfolio_phase_weights": str(args.portfolio_phase_weights),
        "moead_neighborhood_size": int(args.moead_neighborhood_size),
        "moead_delta": float(args.moead_delta),
        "moead_nr": int(args.moead_nr),
        "vns_k_max": int(args.vns_k_max),
        "vns_batch_size": int(args.vns_batch_size),
        "effective_vns_batch_size": int(prepared["effective_vns_batch_size"]),
        "batched_eval_enabled": bool(prepared["batched_eval_enabled"]),
        "reinvest_search": bool(prepared["reinvest_enabled"]),
        "reinvest_pop_mult": float(args.reinvest_pop_mult),
        "reinvest_gen_mult": float(args.reinvest_gen_mult),
        "reinvest_strict4_workers_mult": float(args.reinvest_strict4_workers_mult),
        "dynamic_pool_enabled": bool(prepared["dynamic_pool_enabled"]),
        "dynamic_pool_epochs": int(prepared["dynamic_pool_epochs"]),
        "dynamic_init_minimal": bool(prepared["dynamic_init_minimal"]),
        "dynamic_expand_max_new": int(prepared["dynamic_expand_max_new"]),
        "dynamic_focus_top_features": int(prepared["dynamic_focus_top_features"]),
        "dynamic_partner_topk": int(prepared["dynamic_partner_topk"]),
        "dynamic_top_cache_use": int(prepared["dynamic_top_cache_use"]),
        "dynamic_max_pool_size": int(prepared["dynamic_max_pool_size"]),
        "dynamic_unary_top_k": int(prepared["dynamic_activation_cfg"]["unary_top_k"]),
        "dynamic_pair_top_k": int(prepared["dynamic_activation_cfg"]["pair_top_k"]),
        "dynamic_gate_top_k": int(prepared["dynamic_activation_cfg"]["gate_top_k"]),
        "dynamic_recursive_depth": int(prepared["dynamic_activation_cfg"]["recursive_depth"]),
        "dynamic_recursive_seed_top_k": int(prepared["dynamic_activation_cfg"]["recursive_seed_top_k"]),
        "dynamic_recursive_pair_seed_top_k": int(prepared["dynamic_activation_cfg"]["recursive_pair_seed_top_k"]),
        "dynamic_recursive_max_complexity": float(prepared["dynamic_activation_cfg"]["recursive_max_complexity"]),
        "dynamic_allow_trig": bool(prepared["dynamic_activation_cfg"]["allow_trig"]),
        "dynamic_allow_safe_exp": bool(prepared["dynamic_activation_cfg"]["allow_safe_exp"]),
        "dynamic_allow_safe_log": bool(prepared["dynamic_activation_cfg"]["allow_safe_log"]),
        "dynamic_allow_safe_ratio": bool(prepared["dynamic_activation_cfg"]["allow_safe_ratio"]),
        "dynamic_family_budget": dict(prepared["dynamic_activation_cfg"]["family_budget"]),
        "graph_cache_enabled": bool(prepared["graph_cache_enabled"]),
        "graph_cache_backend": str(prepared["graph_cache_backend"]),
        "graph_cache_db_path": str(prepared["graph_cache_db_path"]),
        "graph_cache_namespace": str(args.graph_cache_namespace),
        "graph_cache_persist_values": bool(int(args.graph_cache_persist_values)),
        "interval_alpha": float(interval_alpha),
        "interval_method": str(interval_method),
        "interval_calib_ratio": float(args.interval_calib_ratio),
        "interval_quantile_l2": float(args.interval_quantile_l2),
        "selection_coverage_error_threshold": float(max(0.0, args.selection_coverage_error_threshold)),
        "safe_log1p_abs": bool(prepared["safe_log1p_abs_enabled"]),
        "safe_exp_clip": bool(prepared["safe_exp_clip_enabled"]),
        "safe_reciprocal": bool(prepared["safe_reciprocal_enabled"]),
        "safe_exp_clip_k": float(prepared["safe_exp_clip_k"]),
        "safe_reciprocal_eps": float(prepared["safe_reciprocal_eps"]),
        "lag_feature_enabled": bool(prepared["lag_enabled"]),
        "lag_orders": [int(v) for v in prepared["lag_orders"]],
        "lag_sources": sorted([str(v) for v in prepared["lag_source_set"]]),
        "lag_added_features": list(prepared["lag_added_features"]),
        "lag_cross_enabled": bool(prepared["lag_cross_enabled"]),
        "lag_cross_quantiles": [float(v) for v in prepared["lag_cross_q"]],
        "lag_cross_added_features": list(prepared["lag_cross_added_features"]),
        "drop_same_day_flow_speed_occ": bool(int(args.drop_same_day_flow_speed_occ)),
        "drop_feature_list": [s.strip() for s in str(args.drop_feature_list).split(",") if s.strip()],
        "dropped_features": list(prepared["dropped_features"]),
        "temporal_pack_enabled": bool(prepared["temporal_pack_enabled"]),
        "temporal_pack_rolling_enabled": bool(int(args.temporal_pack_rolling_enabled)),
        "temporal_pack_momentum_enabled": bool(int(args.temporal_pack_momentum_enabled)),
        "temporal_pack_cross_enabled": bool(int(args.temporal_pack_cross_enabled)),
        "temporal_pack_ratio_enabled": bool(int(args.temporal_pack_ratio_enabled)),
        "temporal_pack_cross_quantiles": [float(v) for v in prepared["temporal_pack_cross_q"]],
        "temporal_pack_ratio_eps": float(prepared["temporal_pack_ratio_eps"]),
        "temporal_pack_added_features": list(prepared["temporal_pack_added_features"]),
        "temporal_pack_rolling_added": list(prepared["temporal_pack_rolling_added"]),
        "temporal_pack_momentum_added": list(prepared["temporal_pack_momentum_added"]),
        "temporal_pack_cross_added": list(prepared["temporal_pack_cross_added"]),
        "temporal_pack_ratio_added": list(prepared["temporal_pack_ratio_added"]),
        "regime_pack_enabled": bool(prepared["regime_pack_enabled"]),
        "regime_pack_volatility_enabled": bool(int(args.regime_pack_volatility_enabled)),
        "regime_pack_shock_enabled": bool(int(args.regime_pack_shock_enabled)),
        "regime_pack_ci_regime_enabled": bool(int(args.regime_pack_ci_regime_enabled)),
        "regime_pack_shock_quantiles": [float(v) for v in prepared["regime_pack_shock_q"]],
        "regime_pack_ci_quantiles": [float(v) for v in prepared["regime_pack_ci_q"]],
        "regime_pack_eps": float(prepared["regime_pack_eps"]),
        "regime_pack_added_features": list(prepared["regime_pack_added_features"]),
        "regime_pack_volatility_added": list(prepared["regime_pack_volatility_added"]),
        "regime_pack_shock_added": list(prepared["regime_pack_shock_added"]),
        "regime_pack_ci_regime_added": list(prepared["regime_pack_ci_regime_added"]),
        "inner_opt_enabled": bool(int(args.inner_opt_enabled)),
        "inner_opt_adam_steps": int(args.inner_opt_adam_steps),
        "inner_opt_adam_lr": float(args.inner_opt_adam_lr),
        "inner_opt_lbfgs_steps": int(args.inner_opt_lbfgs_steps),
        "inner_opt_lbfgs_lr": float(args.inner_opt_lbfgs_lr),
        "inner_opt_accept_rmse_tol": float(args.inner_opt_accept_rmse_tol),
        "inner_opt_accept_rel_tol": float(args.inner_opt_accept_rel_tol),
        "inner_opt_guard_patience": int(args.inner_opt_guard_patience),
        "inner_opt_guard_check_interval": int(args.inner_opt_guard_check_interval),
        "inner_opt_alt_freeze_readout": bool(int(args.inner_opt_alt_freeze_readout)),
        "inner_opt_grad_clip_norm": float(args.inner_opt_grad_clip_norm),
        "inner_opt_residual_clip_q": float(args.inner_opt_residual_clip_q),
        "outer_decision_encoding": "expanded_structure_plus_hyperparams_v1",
        "nesting_architecture": "outer_structure_search -> middle_param_optimizer -> inner_rolling_eval",
    }


def _build_dataset_section(prepared: Mapping[str, Any], X_train: np.ndarray, X_test: np.ndarray) -> dict[str, Any]:
    feature_names = tuple(str(v) for v in prepared["feature_names"])
    feature_names_raw = tuple(str(v) for v in prepared["feature_names_raw"])
    return {
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "n_features": int(X_train.shape[1]),
        "n_features_raw": int(prepared["n_features_raw"]),
        "feature_names_raw": list(feature_names_raw),
        "feature_names": list(feature_names),
        "dropped_features": list(prepared["dropped_features"]),
        "lag_added_features": list(prepared["lag_added_features"]),
        "lag_cross_added_features": list(prepared["lag_cross_added_features"]),
        "temporal_pack_added_features": list(prepared["temporal_pack_added_features"]),
        "temporal_pack_rolling_added": list(prepared["temporal_pack_rolling_added"]),
        "temporal_pack_momentum_added": list(prepared["temporal_pack_momentum_added"]),
        "temporal_pack_cross_added": list(prepared["temporal_pack_cross_added"]),
        "temporal_pack_ratio_added": list(prepared["temporal_pack_ratio_added"]),
        "regime_pack_added_features": list(prepared["regime_pack_added_features"]),
        "regime_pack_volatility_added": list(prepared["regime_pack_volatility_added"]),
        "regime_pack_shock_added": list(prepared["regime_pack_shock_added"]),
        "regime_pack_ci_regime_added": list(prepared["regime_pack_ci_regime_added"]),
    }


def _build_outer_search_section(prepared: Mapping[str, Any], search: SearchStageResult) -> dict[str, Any]:
    problem = search.problem
    dynamic_epoch_logs = list(search.dynamic_epoch_logs)
    return {
        "duration_sec": float(search.outer_sec),
        "resource_budget": _jsonable(dict(search.resource_budget)),
        "objective_schema": ["coverage_error", "pinaw", "interval_score"],
        "outer_meta": _jsonable({**dict(search.outer_meta), "dynamic_pool_epochs": list(dynamic_epoch_logs)}),
        "n_candidates": int(len(search.candidates)),
        "n_families": int(len(problem.families)),
        "families": [str(v) for v in problem.families],
        "run_result": _jsonable(search.run),
        "n_cached_evals": int(search.n_cached_evals),
        "top_cache": list(search.top_cache)[:20],
        "graph_cache": _jsonable(prepared["graph_cache"].snapshot()),
    }


def _build_best_solution_section(search: SearchStageResult, comparison: ComparisonStageResult) -> dict[str, Any]:
    best_row = dict(search.best_row)
    best_subset_idx = [int(v) for v in search.best_subset_idx]
    best_k = int(search.best_k)
    best_decode_meta = dict(search.best_decode_meta)
    fit_final = comparison.fit_final
    return {
        "objective_schema": ["coverage_error", "pinaw", "interval_score"],
        "k_decoded": int(best_k),
        "subset_size": int(len(best_subset_idx)),
        "subset_idx": [int(i) for i in best_subset_idx],
        "subset_names": _jsonable(best_row.get("subset_names", [])),
        "subset_families": _jsonable(best_row.get("subset_families", [])),
        "decode_meta": _jsonable(best_decode_meta),
        "obj_coverage_error": float(best_row.get("obj_coverage_error", float("inf"))),
        "obj_pinaw": float(best_row.get("obj_pinaw", float("inf"))),
        "obj_interval_score": float(best_row.get("obj_interval_score", float("inf"))),
        "inner_opt_info": _jsonable(fit_final.get("inner_opt_info", {})),
    }


def _build_test_compare_section(comparison: ComparisonStageResult) -> dict[str, Any]:
    sym_rmse = float(comparison.sym_rmse)
    sym_mae = float(comparison.sym_mae)
    xgb_rmse = float(comparison.xgb_rmse)
    xgb_mae = float(comparison.xgb_mae)
    sym_interval_info = dict(comparison.sym_interval_info)
    sym_interval = comparison.sym_interval
    xgb_calib_q = float(comparison.xgb_calib_q)
    xgb_interval = comparison.xgb_interval
    return {
        "symbolic_subset_rmse": float(sym_rmse),
        "symbolic_subset_mae": float(sym_mae),
        "xgboost_rmse": float(xgb_rmse),
        "xgboost_mae": float(xgb_mae),
        "delta_symbolic_minus_xgb": float(sym_rmse - xgb_rmse),
        "interval_metrics": {
            "symbolic": {
                **_jsonable(sym_interval_info),
                **_jsonable(sym_interval),
            },
            "xgboost": {
                "calib_abs_residual_q": float(xgb_calib_q),
                **_jsonable(xgb_interval),
            },
        },
    }


def assemble_result(
    args: RuntimeCliConfig,
    prepared: Mapping[str, Any],
    search: SearchStageResult,
    comparison: ComparisonStageResult,
) -> FinalStageResult:
    X_train = np.asarray(prepared["X_train"], dtype=float)
    X_test = np.asarray(prepared["X_test"], dtype=float)
    best_row = dict(search.best_row)
    sym_rmse = float(comparison.sym_rmse)
    xgb_rmse = float(comparison.xgb_rmse)
    interval_alpha = float(comparison.interval_alpha)
    interval_method = str(comparison.interval_method)
    sym_interval = comparison.sym_interval
    xgb_interval = comparison.xgb_interval

    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "out_root": str(prepared["out_root"]),
        "config": _build_config_section(
            args,
            prepared,
            interval_alpha=interval_alpha,
            interval_method=interval_method,
        ),
        "dataset": _build_dataset_section(prepared, X_train, X_test),
        "outer_search": _build_outer_search_section(prepared, search),
        "best_solution": _build_best_solution_section(search, comparison),
        "test_compare": _build_test_compare_section(comparison),
    }
    payload = SummaryReportPayload(
        report=report,
        out_root=str(prepared["out_root"]),
        graph_cache_snapshot=_jsonable(prepared["graph_cache"].snapshot()),
        sym_rmse=float(sym_rmse),
        xgb_rmse=float(xgb_rmse),
        sym_interval=_jsonable(sym_interval),
        xgb_interval=_jsonable(xgb_interval),
        interval_alpha=float(interval_alpha),
    )
    result_summary = ResultSummaryPayload(
        out_root=str(prepared["out_root"]),
        best_obj_coverage_error=float(best_row.get("obj_coverage_error", float("inf"))),
        best_obj_pinaw=float(best_row.get("obj_pinaw", float("inf"))),
        best_obj_interval_score=float(best_row.get("obj_interval_score", float("inf"))),
        symbolic_test_rmse=float(sym_rmse),
        xgb_test_rmse=float(xgb_rmse),
        symbolic_test_pinaw=float(sym_interval.get("pinaw", float("inf"))),
        symbolic_test_interval_score=float(sym_interval.get("interval_score", float("inf"))),
    )
    return FinalStageResult(
        status="completed",
        report_payload=payload,
        result_summary=result_summary,
    )


def run_assemble_result_stage(context: dict[str, Any]) -> FinalStageResult:
    args = ctx_require(context, RuntimeContextKey.ARGS)
    prepared = ctx_require(context, RuntimeContextKey.PREPARED)
    search = ctx_require(context, RuntimeContextKey.SEARCH)
    comparison = ctx_require(context, RuntimeContextKey.COMPARISON)
    result = assemble_result(args, prepared, search, comparison)
    ctx_set(context, RuntimeContextKey.FINAL_RESULT, result)
    return result


__all__ = ["assemble_result", "run_assemble_result_stage"]
