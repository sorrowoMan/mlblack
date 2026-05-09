from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from bias import BranchPolicyConfig, BranchPolicyResolution, ObjectivePolicyConfig
from core.symbolic.expression_graph_cache import ExpressionGraphCache
from pipeline.feature_space import CandidateTerm
from nowcasting_work_ci.mlblack_side.problem.problem_model import SymbolicSubsetSelectionProblem
from nowcasting_work_ci.mlblack_side.problem.domain_router import default_work_ci_branch_policy


@dataclass(frozen=True)
class ProblemConfig:
    max_terms: int = 12
    ridge_l2: float = 1e-4
    rolling_folds: int = 3
    rolling_val_ratio: float = 0.18
    min_train_ratio: float = 0.4
    random_seed: int | None = 42
    inner_opt_enabled: bool = True
    inner_opt_adam_steps: int = 80
    inner_opt_adam_lr: float = 1e-2
    inner_opt_lbfgs_steps: int = 25
    inner_opt_lbfgs_lr: float = 0.8
    inner_opt_accept_rmse_tol: float = 0.0
    inner_opt_accept_rel_tol: float = 0.01
    inner_opt_guard_patience: int = 3
    inner_opt_guard_check_interval: int = 10
    inner_opt_alt_freeze_readout: bool = True
    inner_opt_grad_clip_norm: float = 1.0
    inner_opt_residual_clip_q: float = 0.98
    interval_alpha: float = 0.1
    interval_method: str = "native_quantile_cqr"
    interval_calib_ratio: float = 0.2
    interval_quantile_l2: float = 1e-4
    branch_policy: BranchPolicyConfig = field(default_factory=default_work_ci_branch_policy)
    objective_policy: ObjectivePolicyConfig = field(default_factory=ObjectivePolicyConfig)

def build_problem(
    *,
    X_fit: np.ndarray,
    y_fit: np.ndarray,
    candidates: Sequence[CandidateTerm],
    cfg: ProblemConfig,
    branch_resolution: BranchPolicyResolution,
    strict4_workers: int,
    graph_cache: ExpressionGraphCache | None,
) -> SymbolicSubsetSelectionProblem:
    x = np.asarray(X_fit, dtype=float)
    y = np.asarray(y_fit, dtype=float).reshape(-1, 1)
    return SymbolicSubsetSelectionProblem(
        X_fit=x,
        y_fit=y,
        candidates=list(candidates),
        max_terms=int(max(2, cfg.max_terms)),
        ridge_l2=float(max(0.0, cfg.ridge_l2)),
        rolling_folds=int(max(1, cfg.rolling_folds)),
        rolling_val_ratio=float(np.clip(cfg.rolling_val_ratio, 0.05, 0.45)),
        min_train=max(256, int(round(float(max(0.05, cfg.min_train_ratio)) * x.shape[0]))),
        random_seed=None if cfg.random_seed is None else int(cfg.random_seed),
        regime_branch_mode=bool(branch_resolution.enabled),
        regime_gate_idx=branch_resolution.gate_idx,
        regime_min_branch_train=int(max(8, cfg.branch_policy.min_branch_train)),
        regime_branch_parallel_workers=int(max(1, strict4_workers)),
        regime_policy=branch_resolution.regime_policy,
        inner_opt_enabled=bool(cfg.inner_opt_enabled),
        inner_opt_adam_steps=int(max(0, cfg.inner_opt_adam_steps)),
        inner_opt_adam_lr=float(max(1e-8, cfg.inner_opt_adam_lr)),
        inner_opt_lbfgs_steps=int(max(0, cfg.inner_opt_lbfgs_steps)),
        inner_opt_lbfgs_lr=float(max(1e-8, cfg.inner_opt_lbfgs_lr)),
        inner_opt_accept_rmse_tol=float(max(0.0, cfg.inner_opt_accept_rmse_tol)),
        inner_opt_accept_rel_tol=float(max(0.0, cfg.inner_opt_accept_rel_tol)),
        inner_opt_guard_patience=int(max(1, cfg.inner_opt_guard_patience)),
        inner_opt_guard_check_interval=int(max(1, cfg.inner_opt_guard_check_interval)),
        inner_opt_alt_freeze_readout=bool(cfg.inner_opt_alt_freeze_readout),
        inner_opt_grad_clip_norm=float(max(0.0, cfg.inner_opt_grad_clip_norm)),
        inner_opt_residual_clip_q=float(np.clip(cfg.inner_opt_residual_clip_q, 0.70, 0.999)),
        interval_alpha=float(np.clip(cfg.interval_alpha, 1e-6, 0.99)),
        interval_method=str(cfg.interval_method),
        interval_calib_ratio=float(np.clip(cfg.interval_calib_ratio, 0.05, 0.4)),
        interval_quantile_l2=float(max(0.0, cfg.interval_quantile_l2)),
        selection_coverage_error_threshold=float(max(0.0, cfg.objective_policy.coverage_error_threshold)),
        graph_cache=graph_cache,
    )


__all__ = ["ProblemConfig", "BranchPolicyResolution", "build_problem"]
