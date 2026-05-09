from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Sequence

from bias import DynamicPoolPolicyConfig
from examples.path_defaults import default_work_ci_csv
from nowcasting_work_ci.mlblack_side.config import MlblackRuntimeConfig


@dataclass(frozen=True)
class RuntimeCliConfig:
    csv_path: str = default_work_ci_csv()
    target_col: str = "ci"
    test_fold_col: str = "test_fold_10"
    pop_size: int = 64
    generations: int = 50
    rolling_folds: int = 3
    rolling_val_ratio: float = 0.18
    max_terms: int = 12
    ridge_l2: float = 1e-4
    strict4_branch_mode: bool = False
    strict4_min_branch_train: int = 64
    strict4_branch_parallel_workers: int = 4
    seed: int = 42
    outer_strategy: str = "portfolio"
    portfolio_phases: str = "nsga2,moead,vns"
    portfolio_phase_weights: str = "2,1,1"
    moead_neighborhood_size: int = 12
    moead_delta: float = 0.9
    moead_nr: int = 2
    vns_k_max: int = 5
    vns_batch_size: int = 32
    inner_opt_enabled: int = 1
    inner_opt_adam_steps: int = 80
    inner_opt_adam_lr: float = 1e-2
    inner_opt_lbfgs_steps: int = 25
    inner_opt_lbfgs_lr: float = 0.8
    inner_opt_accept_rmse_tol: float = 0.0
    inner_opt_accept_rel_tol: float = 0.01
    inner_opt_guard_patience: int = 3
    inner_opt_guard_check_interval: int = 10
    inner_opt_alt_freeze_readout: int = 1
    inner_opt_grad_clip_norm: float = 1.0
    inner_opt_residual_clip_q: float = 0.98
    batched_eval: int = 1
    reinvest_search: int = 1
    reinvest_pop_mult: float = 1.5
    reinvest_gen_mult: float = 1.5
    reinvest_strict4_workers_mult: float = 1.5
    dynamic_pool_enabled: int = 1
    dynamic_pool_epochs: int = 5
    dynamic_init_minimal: int = 1
    dynamic_expand_max_new: int = 64
    dynamic_focus_top_features: int = 8
    dynamic_partner_topk: int = 6
    dynamic_top_cache_use: int = 32
    dynamic_max_pool_size: int = 640
    dynamic_unary_top_k: int = 6
    dynamic_pair_top_k: int = 8
    dynamic_gate_top_k: int = 6
    dynamic_recursive_depth: int = 2
    dynamic_recursive_seed_top_k: int = 3
    dynamic_recursive_pair_seed_top_k: int = 2
    dynamic_recursive_max_complexity: float = 9.5
    dynamic_allow_trig: int = 1
    dynamic_allow_safe_exp: int = 1
    dynamic_allow_safe_log: int = 1
    dynamic_allow_safe_ratio: int = 1
    dynamic_family_budget_csv: str = DynamicPoolPolicyConfig().family_budget_csv
    graph_cache_enabled: int = 1
    graph_cache_backend: str = "sqlite"
    graph_cache_db_path: str = ""
    graph_cache_namespace: str = MlblackRuntimeConfig().graph_cache_namespace
    graph_cache_persist_values: int = 0
    interval_alpha: float = 0.1
    interval_method: str = "native_quantile_cqr"
    interval_calib_ratio: float = 0.2
    interval_quantile_l2: float = 1e-4
    selection_coverage_error_threshold: float = 0.03
    safe_log1p_abs: int = 1
    safe_exp_clip: int = 1
    safe_reciprocal: int = 1
    safe_exp_clip_k: float = 8.0
    safe_reciprocal_eps: float = 1e-3
    lag_feature_enabled: int = 1
    lag_orders: str = "1,2,3"
    lag_sources: str = "ci,total_flow,avg_speed,avg_occ"
    lag_cross_enabled: int = 1
    lag_cross_quantiles: str = "0.25,0.5,0.75"
    drop_same_day_flow_speed_occ: int = 1
    drop_feature_list: str = "total_flow,avg_speed,avg_occ"
    temporal_pack_enabled: int = 1
    temporal_pack_rolling_enabled: int = 1
    temporal_pack_momentum_enabled: int = 1
    temporal_pack_cross_enabled: int = 1
    temporal_pack_ratio_enabled: int = 1
    temporal_pack_cross_quantiles: str = "0.5"
    temporal_pack_ratio_eps: float = 1e-3
    regime_pack_enabled: int = 1
    regime_pack_volatility_enabled: int = 1
    regime_pack_shock_enabled: int = 1
    regime_pack_ci_regime_enabled: int = 1
    regime_pack_shock_quantiles: str = "0.8,0.9"
    regime_pack_ci_quantiles: str = "0.33,0.66"
    regime_pack_eps: float = 1e-6


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "NOWCASTING package: NSGABLACK outer (subset optimization) + MLBLACK inner "
            "(symbolic ridge eval) on Work-CI (same-day state estimation, not strict t+1 forecasting)."
        )
    )
    parser.add_argument("--csv-path", type=str, default=default_work_ci_csv())
    parser.add_argument("--target-col", type=str, default="ci")
    parser.add_argument("--test-fold-col", type=str, default="test_fold_10")
    parser.add_argument("--pop-size", type=int, default=64)
    parser.add_argument("--generations", type=int, default=50)
    parser.add_argument("--rolling-folds", type=int, default=3)
    parser.add_argument("--rolling-val-ratio", type=float, default=0.18)
    parser.add_argument("--max-terms", type=int, default=12)
    parser.add_argument("--ridge-l2", type=float, default=1e-4)
    parser.add_argument("--strict4-branch-mode", action="store_true")
    parser.add_argument("--strict4-min-branch-train", type=int, default=64)
    parser.add_argument("--strict4-branch-parallel-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outer-strategy", type=str, default="portfolio", choices=["nsga2", "moead", "vns", "portfolio"])
    parser.add_argument("--portfolio-phases", type=str, default="nsga2,moead,vns")
    parser.add_argument("--portfolio-phase-weights", type=str, default="2,1,1")
    parser.add_argument("--moead-neighborhood-size", type=int, default=12)
    parser.add_argument("--moead-delta", type=float, default=0.9)
    parser.add_argument("--moead-nr", type=int, default=2)
    parser.add_argument("--vns-k-max", type=int, default=5)
    parser.add_argument("--vns-batch-size", type=int, default=32)
    parser.add_argument("--inner-opt-enabled", type=int, default=1)
    parser.add_argument("--inner-opt-adam-steps", type=int, default=80)
    parser.add_argument("--inner-opt-adam-lr", type=float, default=1e-2)
    parser.add_argument("--inner-opt-lbfgs-steps", type=int, default=25)
    parser.add_argument("--inner-opt-lbfgs-lr", type=float, default=0.8)
    parser.add_argument("--inner-opt-accept-rmse-tol", type=float, default=0.0)
    parser.add_argument("--inner-opt-accept-rel-tol", type=float, default=0.01)
    parser.add_argument("--inner-opt-guard-patience", type=int, default=3)
    parser.add_argument("--inner-opt-guard-check-interval", type=int, default=10)
    parser.add_argument("--inner-opt-alt-freeze-readout", type=int, default=1)
    parser.add_argument("--inner-opt-grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--inner-opt-residual-clip-q", type=float, default=0.98)
    parser.add_argument("--batched-eval", type=int, default=1)
    parser.add_argument("--reinvest-search", type=int, default=1)
    parser.add_argument("--reinvest-pop-mult", type=float, default=1.5)
    parser.add_argument("--reinvest-gen-mult", type=float, default=1.5)
    parser.add_argument("--reinvest-strict4-workers-mult", type=float, default=1.5)
    parser.add_argument("--dynamic-pool-enabled", type=int, default=1)
    parser.add_argument("--dynamic-pool-epochs", type=int, default=5)
    parser.add_argument("--dynamic-init-minimal", type=int, default=1)
    parser.add_argument("--dynamic-expand-max-new", type=int, default=64)
    parser.add_argument("--dynamic-focus-top-features", type=int, default=8)
    parser.add_argument("--dynamic-partner-topk", type=int, default=6)
    parser.add_argument("--dynamic-top-cache-use", type=int, default=32)
    parser.add_argument("--dynamic-max-pool-size", type=int, default=640)
    parser.add_argument("--dynamic-unary-top-k", type=int, default=6)
    parser.add_argument("--dynamic-pair-top-k", type=int, default=8)
    parser.add_argument("--dynamic-gate-top-k", type=int, default=6)
    parser.add_argument("--dynamic-recursive-depth", type=int, default=2)
    parser.add_argument("--dynamic-recursive-seed-top-k", type=int, default=3)
    parser.add_argument("--dynamic-recursive-pair-seed-top-k", type=int, default=2)
    parser.add_argument("--dynamic-recursive-max-complexity", type=float, default=9.5)
    parser.add_argument("--dynamic-allow-trig", type=int, default=1)
    parser.add_argument("--dynamic-allow-safe-exp", type=int, default=1)
    parser.add_argument("--dynamic-allow-safe-log", type=int, default=1)
    parser.add_argument("--dynamic-allow-safe-ratio", type=int, default=1)
    parser.add_argument("--dynamic-family-budget-csv", type=str, default=DynamicPoolPolicyConfig().family_budget_csv)
    parser.add_argument("--graph-cache-enabled", type=int, default=1)
    parser.add_argument("--graph-cache-backend", type=str, default="sqlite", choices=["memory", "sqlite"])
    parser.add_argument("--graph-cache-db-path", type=str, default="")
    parser.add_argument("--graph-cache-namespace", type=str, default=MlblackRuntimeConfig().graph_cache_namespace)
    parser.add_argument("--graph-cache-persist-values", type=int, default=0)
    parser.add_argument("--interval-alpha", type=float, default=0.1, help="Interval alpha (target coverage=1-alpha).")
    parser.add_argument("--interval-method", type=str, default="native_quantile_cqr", choices=["native_quantile_cqr", "symmetric_residual"], help="Interval construction method for symbolic model.")
    parser.add_argument("--interval-calib-ratio", type=float, default=0.2, help="Calibration split ratio for conformal quantile interval.")
    parser.add_argument("--interval-quantile-l2", type=float, default=1e-4, help="L2 regularization for QuantileRegressor heads.")
    parser.add_argument("--selection-coverage-error-threshold", type=float, default=0.03, help="Final selection tolerance on coverage_error before prioritizing PINAW/Interval Score.")
    parser.add_argument("--safe-log1p-abs", type=int, default=1, help="Enable safe basis log(1+abs(x)).")
    parser.add_argument("--safe-exp-clip", type=int, default=1, help="Enable safe basis exp(clip-scaled x).")
    parser.add_argument("--safe-reciprocal", type=int, default=1, help="Enable safe basis 1/(abs(x)+eps).")
    parser.add_argument("--safe-exp-clip-k", type=float, default=8.0, help="Scale k for exp_clip: exp(x/k).")
    parser.add_argument("--safe-reciprocal-eps", type=float, default=1e-3, help="Epsilon for reciprocal_safe.")
    parser.add_argument("--lag-feature-enabled", type=int, default=1, help="If 1, generate lag features for selected sources.")
    parser.add_argument("--lag-orders", type=str, default="1,2,3", help="Comma-separated lag orders, e.g. 1,2,3")
    parser.add_argument("--lag-sources", type=str, default="ci,total_flow,avg_speed,avg_occ", help="Comma-separated lag sources from {ci,total_flow,avg_speed,avg_occ}")
    parser.add_argument("--lag-cross-enabled", type=int, default=1, help="If 1, add hinge(ci_lag1,c)*avg_speed_lag1 cross-lag terms.")
    parser.add_argument("--lag-cross-quantiles", type=str, default="0.25,0.5,0.75", help="Quantiles for c in hinge(ci_lag1,c)*avg_speed_lag1")
    parser.add_argument("--drop-same-day-flow-speed-occ", type=int, default=1, help="If 1, drop leak-prone same-day features: total_flow/avg_speed/avg_occ.")
    parser.add_argument("--drop-feature-list", type=str, default="total_flow,avg_speed,avg_occ", help="Comma-separated feature names to drop from train/test feature matrices.")
    parser.add_argument("--temporal-pack-enabled", type=int, default=1, help="Enable compact temporal feature strengthening pack.")
    parser.add_argument("--temporal-pack-rolling-enabled", type=int, default=1, help="Enable rolling mean/std features from lag terms.")
    parser.add_argument("--temporal-pack-momentum-enabled", type=int, default=1, help="Enable lag-delta and slope momentum features.")
    parser.add_argument("--temporal-pack-cross-enabled", type=int, default=1, help="Enable targeted ci_lag1 gated cross-lag interactions.")
    parser.add_argument("--temporal-pack-ratio-enabled", type=int, default=1, help="Enable safe ratio lag features.")
    parser.add_argument("--temporal-pack-cross-quantiles", type=str, default="0.5", help="Comma-separated quantiles for targeted ci_lag1 hinge crosses in temporal pack.")
    parser.add_argument("--temporal-pack-ratio-eps", type=float, default=1e-3, help="Epsilon for safe ratios in temporal pack.")
    parser.add_argument("--regime-pack-enabled", type=int, default=1, help="Enable changepoint/regime state feature pack.")
    parser.add_argument("--regime-pack-volatility-enabled", type=int, default=1, help="Enable local volatility/acceleration features from lag deltas.")
    parser.add_argument("--regime-pack-shock-enabled", type=int, default=1, help="Enable shock z-score and shock flags.")
    parser.add_argument("--regime-pack-ci-regime-enabled", type=int, default=1, help="Enable ci_lag1 regime state indicators.")
    parser.add_argument("--regime-pack-shock-quantiles", type=str, default="0.8,0.9", help="Comma-separated quantiles for shock flags on lag z-score.")
    parser.add_argument("--regime-pack-ci-quantiles", type=str, default="0.33,0.66", help="Comma-separated quantiles for ci regime bucket boundaries.")
    parser.add_argument("--regime-pack-eps", type=float, default=1e-6, help="Numerical epsilon for regime feature pack.")
    return parser


def parse_runtime_config(argv: Sequence[str] | None = None) -> RuntimeCliConfig:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    return RuntimeCliConfig(
        csv_path=str(args.csv_path),
        target_col=str(args.target_col),
        test_fold_col=str(args.test_fold_col),
        pop_size=int(args.pop_size),
        generations=int(args.generations),
        rolling_folds=int(args.rolling_folds),
        rolling_val_ratio=float(args.rolling_val_ratio),
        max_terms=int(args.max_terms),
        ridge_l2=float(args.ridge_l2),
        strict4_branch_mode=bool(args.strict4_branch_mode),
        strict4_min_branch_train=int(args.strict4_min_branch_train),
        strict4_branch_parallel_workers=int(args.strict4_branch_parallel_workers),
        seed=int(args.seed),
        outer_strategy=str(args.outer_strategy),
        portfolio_phases=str(args.portfolio_phases),
        portfolio_phase_weights=str(args.portfolio_phase_weights),
        moead_neighborhood_size=int(args.moead_neighborhood_size),
        moead_delta=float(args.moead_delta),
        moead_nr=int(args.moead_nr),
        vns_k_max=int(args.vns_k_max),
        vns_batch_size=int(args.vns_batch_size),
        inner_opt_enabled=int(args.inner_opt_enabled),
        inner_opt_adam_steps=int(args.inner_opt_adam_steps),
        inner_opt_adam_lr=float(args.inner_opt_adam_lr),
        inner_opt_lbfgs_steps=int(args.inner_opt_lbfgs_steps),
        inner_opt_lbfgs_lr=float(args.inner_opt_lbfgs_lr),
        inner_opt_accept_rmse_tol=float(args.inner_opt_accept_rmse_tol),
        inner_opt_accept_rel_tol=float(args.inner_opt_accept_rel_tol),
        inner_opt_guard_patience=int(args.inner_opt_guard_patience),
        inner_opt_guard_check_interval=int(args.inner_opt_guard_check_interval),
        inner_opt_alt_freeze_readout=int(args.inner_opt_alt_freeze_readout),
        inner_opt_grad_clip_norm=float(args.inner_opt_grad_clip_norm),
        inner_opt_residual_clip_q=float(args.inner_opt_residual_clip_q),
        batched_eval=int(args.batched_eval),
        reinvest_search=int(args.reinvest_search),
        reinvest_pop_mult=float(args.reinvest_pop_mult),
        reinvest_gen_mult=float(args.reinvest_gen_mult),
        reinvest_strict4_workers_mult=float(args.reinvest_strict4_workers_mult),
        dynamic_pool_enabled=int(args.dynamic_pool_enabled),
        dynamic_pool_epochs=int(args.dynamic_pool_epochs),
        dynamic_init_minimal=int(args.dynamic_init_minimal),
        dynamic_expand_max_new=int(args.dynamic_expand_max_new),
        dynamic_focus_top_features=int(args.dynamic_focus_top_features),
        dynamic_partner_topk=int(args.dynamic_partner_topk),
        dynamic_top_cache_use=int(args.dynamic_top_cache_use),
        dynamic_max_pool_size=int(args.dynamic_max_pool_size),
        dynamic_unary_top_k=int(args.dynamic_unary_top_k),
        dynamic_pair_top_k=int(args.dynamic_pair_top_k),
        dynamic_gate_top_k=int(args.dynamic_gate_top_k),
        dynamic_recursive_depth=int(args.dynamic_recursive_depth),
        dynamic_recursive_seed_top_k=int(args.dynamic_recursive_seed_top_k),
        dynamic_recursive_pair_seed_top_k=int(args.dynamic_recursive_pair_seed_top_k),
        dynamic_recursive_max_complexity=float(args.dynamic_recursive_max_complexity),
        dynamic_allow_trig=int(args.dynamic_allow_trig),
        dynamic_allow_safe_exp=int(args.dynamic_allow_safe_exp),
        dynamic_allow_safe_log=int(args.dynamic_allow_safe_log),
        dynamic_allow_safe_ratio=int(args.dynamic_allow_safe_ratio),
        dynamic_family_budget_csv=str(args.dynamic_family_budget_csv),
        graph_cache_enabled=int(args.graph_cache_enabled),
        graph_cache_backend=str(args.graph_cache_backend),
        graph_cache_db_path=str(args.graph_cache_db_path),
        graph_cache_namespace=str(args.graph_cache_namespace),
        graph_cache_persist_values=int(args.graph_cache_persist_values),
        interval_alpha=float(args.interval_alpha),
        interval_method=str(args.interval_method),
        interval_calib_ratio=float(args.interval_calib_ratio),
        interval_quantile_l2=float(args.interval_quantile_l2),
        selection_coverage_error_threshold=float(args.selection_coverage_error_threshold),
        safe_log1p_abs=int(args.safe_log1p_abs),
        safe_exp_clip=int(args.safe_exp_clip),
        safe_reciprocal=int(args.safe_reciprocal),
        safe_exp_clip_k=float(args.safe_exp_clip_k),
        safe_reciprocal_eps=float(args.safe_reciprocal_eps),
        lag_feature_enabled=int(args.lag_feature_enabled),
        lag_orders=str(args.lag_orders),
        lag_sources=str(args.lag_sources),
        lag_cross_enabled=int(args.lag_cross_enabled),
        lag_cross_quantiles=str(args.lag_cross_quantiles),
        drop_same_day_flow_speed_occ=int(args.drop_same_day_flow_speed_occ),
        drop_feature_list=str(args.drop_feature_list),
        temporal_pack_enabled=int(args.temporal_pack_enabled),
        temporal_pack_rolling_enabled=int(args.temporal_pack_rolling_enabled),
        temporal_pack_momentum_enabled=int(args.temporal_pack_momentum_enabled),
        temporal_pack_cross_enabled=int(args.temporal_pack_cross_enabled),
        temporal_pack_ratio_enabled=int(args.temporal_pack_ratio_enabled),
        temporal_pack_cross_quantiles=str(args.temporal_pack_cross_quantiles),
        temporal_pack_ratio_eps=float(args.temporal_pack_ratio_eps),
        regime_pack_enabled=int(args.regime_pack_enabled),
        regime_pack_volatility_enabled=int(args.regime_pack_volatility_enabled),
        regime_pack_shock_enabled=int(args.regime_pack_shock_enabled),
        regime_pack_ci_regime_enabled=int(args.regime_pack_ci_regime_enabled),
        regime_pack_shock_quantiles=str(args.regime_pack_shock_quantiles),
        regime_pack_ci_quantiles=str(args.regime_pack_ci_quantiles),
        regime_pack_eps=float(args.regime_pack_eps),
    )


def parse_runtime_args(argv: Sequence[str] | None = None) -> RuntimeCliConfig:
    return parse_runtime_config(argv)


__all__ = ["RuntimeCliConfig", "build_parser", "parse_runtime_args", "parse_runtime_config"]
