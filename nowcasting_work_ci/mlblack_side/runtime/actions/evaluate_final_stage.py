from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from core.common.contracts import ProcessedDataset
from core.trainers.xgboost_trainer import XGBoostSurrogateTrainer, XGBoostTrainerConfig
from model.interval_fit import (
    _as_2d,
    _build_native_quantile_interval,
    _build_symmetric_interval,
    _interval_metrics,
    _mae,
    _rmse,
    _three_layer_fit_predict,
)

from ..config import RuntimeCliConfig
from ..contracts import ComparisonStageResult, RuntimeContextKey, SearchStageResult, ctx_require, ctx_set


def evaluate_final_models(
    args: RuntimeCliConfig,
    prepared: Mapping[str, Any],
    search: SearchStageResult,
) -> ComparisonStageResult:
    X_train = np.asarray(prepared["X_train"], dtype=float)
    y_train = np.asarray(prepared["y_train"], dtype=float)
    X_test = np.asarray(prepared["X_test"], dtype=float)
    y_test = np.asarray(prepared["y_test"], dtype=float)
    feature_names = tuple(str(v) for v in prepared["feature_names"])
    best_genome = list(search.best_genome)
    best_decode_meta = dict(search.best_decode_meta)

    fit_final = _three_layer_fit_predict(
        genome=best_genome,
        X_train=X_train,
        y_train=y_train,
        X_eval=X_test,
        y_eval=y_test,
        l2=float(max(0.0, best_decode_meta["tuned_l2"])),
        inner_opt_enabled=bool(int(args.inner_opt_enabled)),
        inner_opt_adam_steps=int(max(0, args.inner_opt_adam_steps)),
        inner_opt_adam_lr=float(max(1e-8, args.inner_opt_adam_lr)),
        inner_opt_lbfgs_steps=int(max(0, args.inner_opt_lbfgs_steps)),
        inner_opt_lbfgs_lr=float(max(1e-8, args.inner_opt_lbfgs_lr)),
        inner_opt_accept_rmse_tol=float(max(0.0, args.inner_opt_accept_rmse_tol)),
        inner_opt_accept_rel_tol=float(max(0.0, args.inner_opt_accept_rel_tol)),
        inner_opt_guard_patience=int(max(1, args.inner_opt_guard_patience)),
        inner_opt_guard_check_interval=int(max(1, args.inner_opt_guard_check_interval)),
        inner_opt_alt_freeze_readout=bool(int(args.inner_opt_alt_freeze_readout)),
        inner_opt_grad_clip_norm=float(max(0.0, args.inner_opt_grad_clip_norm)),
        inner_opt_residual_clip_q=float(np.clip(args.inner_opt_residual_clip_q, 0.70, 0.999)),
        random_seed=int(args.seed),
    )
    sym_pred_train = _as_2d(np.asarray(fit_final.get("pred_train"), dtype=float))
    sym_pred_test = _as_2d(np.asarray(fit_final.get("pred_eval"), dtype=float))
    sym_rmse = float(_rmse(y_test, sym_pred_test))
    sym_mae = float(_mae(y_test, sym_pred_test))
    interval_alpha = float(np.clip(args.interval_alpha, 1e-6, 0.99))
    interval_method = str(args.interval_method)
    sym_interval_info: dict[str, Any] = {"method": str(interval_method)}
    if interval_method == "native_quantile_cqr":
        sym_lo, sym_hi, sym_interval_info = _build_native_quantile_interval(
            genome=best_genome,
            X_train=X_train,
            y_train=y_train,
            X_eval=X_test,
            alpha=interval_alpha,
            calib_ratio=float(np.clip(args.interval_calib_ratio, 0.05, 0.4)),
            quantile_l2=float(max(0.0, args.interval_quantile_l2)),
        )
    else:
        sym_lo, sym_hi, q = _build_symmetric_interval(
            y_train=y_train,
            pred_train=sym_pred_train,
            pred_eval=sym_pred_test,
            alpha=interval_alpha,
        )
        sym_interval_info = {"method": "symmetric_residual", "conformal_qhat": float(q)}
    sym_interval = _interval_metrics(
        y_true=y_test,
        lower=sym_lo,
        upper=sym_hi,
        alpha=interval_alpha,
    )

    xgb = XGBoostSurrogateTrainer(
        config=XGBoostTrainerConfig(
            artifact_id="subset_bridge_xgb_baseline",
            n_estimators=360,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            tree_method="hist",
            random_seed=int(args.seed),
        )
    )
    xgb_art = xgb.fit(
        ProcessedDataset(
            X_train=X_train,
            y_train=y_train,
            feature_names=feature_names,
            target_names=(str(args.target_col),),
        )
    )
    xgb_pred = np.asarray(xgb_art.predict(X_test), dtype=float).reshape(-1, 1)
    xgb_rmse = _rmse(y_test, xgb_pred)
    xgb_mae = _mae(y_test, xgb_pred)
    xgb_train_pred = np.asarray(xgb_art.predict(X_train), dtype=float).reshape(-1, 1)
    xgb_lo, xgb_hi, xgb_calib_q = _build_symmetric_interval(
        y_train=y_train,
        pred_train=xgb_train_pred,
        pred_eval=xgb_pred,
        alpha=interval_alpha,
    )
    xgb_interval = _interval_metrics(
        y_true=y_test,
        lower=xgb_lo,
        upper=xgb_hi,
        alpha=interval_alpha,
    )
    return ComparisonStageResult(
        fit_final=dict(fit_final),
        sym_rmse=float(sym_rmse),
        sym_mae=float(sym_mae),
        interval_alpha=float(interval_alpha),
        interval_method=str(interval_method),
        sym_interval_info=dict(sym_interval_info),
        sym_interval=dict(sym_interval),
        xgb_rmse=float(xgb_rmse),
        xgb_mae=float(xgb_mae),
        xgb_calib_q=float(xgb_calib_q),
        xgb_interval=dict(xgb_interval),
    )


def run_evaluate_final_stage(context: dict[str, Any]) -> ComparisonStageResult:
    args = ctx_require(context, RuntimeContextKey.ARGS)
    prepared = ctx_require(context, RuntimeContextKey.PREPARED)
    search = ctx_require(context, RuntimeContextKey.SEARCH)
    comparison = evaluate_final_models(args, prepared, search)
    ctx_set(context, RuntimeContextKey.COMPARISON, comparison)
    return comparison


__all__ = ["evaluate_final_models", "run_evaluate_final_stage"]
