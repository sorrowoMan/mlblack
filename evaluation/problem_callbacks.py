from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

import numpy as np

from core.execution import (
    EXECUTION_RESOURCE_GRANT_KEY,
    ExecutionResourceGrant,
    ExecutionResourceRequest,
    coerce_execution_resource_grant,
    sum_execution_resource_requests,
)
from model.interval_fit import (
    _build_native_quantile_interval,
    _build_symmetric_interval,
    _interval_metrics,
    _three_layer_fit_predict,
)
from pipeline.feature_space import BranchEvaluationConfig, as_2d, coverage_error
from training.inner_runtime import InnerRuntimeDispatcher
from .config import FitPredictCallbackConfig, IntervalCallbackConfig

JsonableFn = Callable[[Any], Any]
RmseFn = Callable[[np.ndarray, np.ndarray], float]


class ProblemEvaluationCallbacks:
    def __init__(
        self,
        *,
        interval_config: IntervalCallbackConfig,
        fit_predict_config: FitPredictCallbackConfig,
        jsonable_fn: JsonableFn,
        rmse_fn: RmseFn,
        inner_runtime_dispatcher: InnerRuntimeDispatcher | None = None,
        inner_runtime_context: Mapping[str, Any] | None = None,
    ) -> None:
        self.interval_config = interval_config
        self.fit_predict_config = fit_predict_config
        self._jsonable = jsonable_fn
        self._rmse = rmse_fn
        self.inner_runtime_dispatcher = inner_runtime_dispatcher
        self.inner_runtime_context = {} if inner_runtime_context is None else dict(inner_runtime_context)

    def build_inner_runtime_context(self, extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
        merged = dict(self.inner_runtime_context)
        if extra:
            for key, value in dict(extra).items():
                if value is not None:
                    merged[str(key)] = value
        return merged

    def execution_resource_grant(self) -> ExecutionResourceGrant | None:
        raw = self.inner_runtime_context.get(EXECUTION_RESOURCE_GRANT_KEY)
        if raw is None:
            return None
        return coerce_execution_resource_grant(raw, phase="problem_evaluation")

    def set_execution_resource_grant(
        self,
        grant: ExecutionResourceGrant | ExecutionResourceRequest | Mapping[str, Any] | None,
    ) -> ExecutionResourceGrant | None:
        if grant is None:
            self.inner_runtime_context.pop(EXECUTION_RESOURCE_GRANT_KEY, None)
            return None
        resolved = coerce_execution_resource_grant(grant, phase="problem_evaluation")
        self.inner_runtime_context[EXECUTION_RESOURCE_GRANT_KEY] = resolved.as_dict()
        return resolved

    def _effective_regime_branch_parallel_workers(self) -> int:
        requested = int(max(1, self.interval_config.regime_branch_parallel_workers))
        grant = self.execution_resource_grant()
        if grant is None:
            return requested
        extra_parallel_budget = max(0, int(grant.threads) - 1)
        if extra_parallel_budget <= 0:
            return 1
        return int(max(1, min(requested, extra_parallel_budget)))

    def branch_eval_config(self) -> BranchEvaluationConfig:
        return BranchEvaluationConfig(
            regime_branch_mode=bool(self.interval_config.regime_branch_mode),
            regime_gate_idx=self.interval_config.regime_gate_idx,
            base_regime_min_branch_train=int(self.interval_config.base_regime_min_branch_train),
            regime_branch_parallel_workers=int(self._effective_regime_branch_parallel_workers()),
            interval_alpha=float(self.interval_config.interval_alpha),
            regime_policy=self.interval_config.regime_policy,
        )

    def execution_resource_requests(
        self,
        *,
        rolling_folds: int | None = None,
        label: str = "problem_evaluation",
    ) -> tuple[ExecutionResourceRequest, ...]:
        folds = None if rolling_folds is None else int(max(1, rolling_folds))
        fold_request = ExecutionResourceRequest(
            threads=1,
            backend="serial",
            label=f"{label}:fold",
            metadata={
                "rolling_folds": folds,
                "interval_method": str(self.interval_config.interval_method),
                "inner_opt_enabled": bool(self.fit_predict_config.inner_opt_enabled),
            },
        )

        if not bool(self.interval_config.regime_branch_mode) or self.interval_config.regime_gate_idx is None:
            return (fold_request,)

        branch_workers = int(max(1, self._effective_regime_branch_parallel_workers()))
        if branch_workers <= 1:
            return (fold_request,)
        branch_request = ExecutionResourceRequest(
            threads=int(branch_workers),
            backend="thread",
            label=f"{label}:branch_workers",
            metadata={
                "rolling_folds": folds,
                "regime_branch_mode": True,
                "regime_branch_parallel_workers": int(branch_workers),
                "regime_policy": type(self.interval_config.regime_policy).__name__,
            },
        )
        return (fold_request, branch_request)

    def execution_resource_request(
        self,
        *,
        rolling_folds: int | None = None,
        label: str = "problem_evaluation",
    ) -> ExecutionResourceRequest:
        components = tuple(self.execution_resource_requests(rolling_folds=rolling_folds, label=label))
        total = sum_execution_resource_requests(components, label=label)
        metadata = dict(total.metadata)
        metadata.update(
            {
                "rolling_folds": None if rolling_folds is None else int(max(1, rolling_folds)),
                "interval_method": str(self.interval_config.interval_method),
                "regime_branch_mode": bool(self.interval_config.regime_branch_mode),
                "inner_opt_enabled": bool(self.fit_predict_config.inner_opt_enabled),
                "execution_resource_grant": (
                    None if self.execution_resource_grant() is None else self.execution_resource_grant().as_dict()
                ),
            }
        )
        return ExecutionResourceRequest(
            threads=int(total.threads),
            backend=("thread" if len(components) > 1 else "serial"),
            label=str(label),
            device_tokens=tuple(total.device_tokens),
            metadata=metadata,
        )

    def build_interval_bounds(
        self,
        *,
        genome: Sequence[Mapping[str, Any]],
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_eval: np.ndarray,
        pred_train: np.ndarray,
        pred_eval: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        if self.interval_config.interval_method == "native_quantile_cqr":
            lower, upper, info = _build_native_quantile_interval(
                genome=genome,
                X_train=np.asarray(X_train, dtype=float),
                y_train=np.asarray(y_train, dtype=float),
                X_eval=np.asarray(X_eval, dtype=float),
                alpha=float(self.interval_config.interval_alpha),
                calib_ratio=float(self.interval_config.interval_calib_ratio),
                quantile_l2=float(self.interval_config.interval_quantile_l2),
            )
            return as_2d(np.asarray(lower, dtype=float)), as_2d(np.asarray(upper, dtype=float)), dict(info)

        lower, upper, qhat = _build_symmetric_interval(
            y_train=np.asarray(y_train, dtype=float),
            pred_train=np.asarray(pred_train, dtype=float),
            pred_eval=np.asarray(pred_eval, dtype=float),
            alpha=float(self.interval_config.interval_alpha),
        )
        return (
            as_2d(np.asarray(lower, dtype=float)),
            as_2d(np.asarray(upper, dtype=float)),
            {"method": "symmetric_residual", "conformal_qhat": float(qhat)},
        )

    def summarize_fold(
        self,
        *,
        y_true: np.ndarray,
        pred_eval: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        mode: str,
        branch_detail: Mapping[str, Any] | None = None,
        interval_info: Mapping[str, Any] | None = None,
        precomputed_interval_metrics: Mapping[str, Any] | None = None,
        precomputed_rmse: float | None = None,
    ) -> dict[str, Any]:
        if precomputed_interval_metrics is None:
            interval = _interval_metrics(
                y_true=np.asarray(y_true, dtype=float),
                lower=np.asarray(lower, dtype=float),
                upper=np.asarray(upper, dtype=float),
                alpha=float(self.interval_config.interval_alpha),
            )
        else:
            interval = dict(precomputed_interval_metrics)
        rmse_value = (
            float(self._rmse(y_true, pred_eval)) if precomputed_rmse is None else float(precomputed_rmse)
        )
        return {
            "coverage_error": float(interval["coverage_error"])
            if "coverage_error" in interval
            else coverage_error(
                picp=float(interval["picp"]),
                coverage_target=float(interval["coverage_target"]),
            ),
            "pinaw": float(interval["pinaw"]),
            "interval_score": float(interval["interval_score"]),
            "picp": float(interval["picp"]),
            "coverage_target": float(interval["coverage_target"]),
            "mean_width": float(interval["mean_width"]),
            "rmse": float(rmse_value),
            "mode": str(mode),
            "branch_detail": self._jsonable(dict(branch_detail or {})),
            "interval_info": self._jsonable(dict(interval_info or {})),
        }

    def fit_predict(
        self,
        *,
        genome: Sequence[Mapping[str, Any]],
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_eval: np.ndarray,
        y_eval: np.ndarray | None,
        l2: float,
    ) -> Mapping[str, Any]:
        cfg = self.fit_predict_config
        return _three_layer_fit_predict(
            genome=genome,
            X_train=np.asarray(X_train, dtype=float),
            y_train=np.asarray(y_train, dtype=float),
            X_eval=np.asarray(X_eval, dtype=float),
            y_eval=None if y_eval is None else np.asarray(y_eval, dtype=float),
            l2=float(max(0.0, l2)),
            random_seed=None if cfg.random_seed is None else int(cfg.random_seed),
            inner_opt_enabled=bool(cfg.inner_opt_enabled),
            inner_opt_adam_steps=int(cfg.inner_opt_adam_steps),
            inner_opt_adam_lr=float(cfg.inner_opt_adam_lr),
            inner_opt_lbfgs_steps=int(cfg.inner_opt_lbfgs_steps),
            inner_opt_lbfgs_lr=float(cfg.inner_opt_lbfgs_lr),
            inner_opt_accept_rmse_tol=float(cfg.inner_opt_accept_rmse_tol),
            inner_opt_accept_rel_tol=float(cfg.inner_opt_accept_rel_tol),
            inner_opt_guard_patience=int(cfg.inner_opt_guard_patience),
            inner_opt_guard_check_interval=int(cfg.inner_opt_guard_check_interval),
            inner_opt_alt_freeze_readout=bool(cfg.inner_opt_alt_freeze_readout),
            inner_opt_grad_clip_norm=float(cfg.inner_opt_grad_clip_norm),
            inner_opt_residual_clip_q=float(cfg.inner_opt_residual_clip_q),
        )


__all__ = [
    "IntervalCallbackConfig",
    "FitPredictCallbackConfig",
    "ProblemEvaluationCallbacks",
]
