from __future__ import annotations

import concurrent.futures
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from core.execution import (
    EXECUTION_RESOURCE_GRANT_KEY,
    EXECUTION_USAGE_REPORTS_KEY,
    ExecutionResourceGrant,
    ExecutionResourceRequest,
    assert_phase_resource_budget,
    build_execution_usage_report,
    coerce_execution_resource_grant,
    constrain_execution_offer_to_grant,
    detect_local_execution_offer,
)
from core.execution.resources import resolve_phase_worker_count
from core.symbolic.feature_space.regime_router import (
    RegimePolicy,
    Strict4RouterSpec,
    build_regime_index,
    holiday_union_indices,
    regime_keys_from_X,
    resolve_branch_train_selection,
)
from training.inner_runtime import (
    InnerRuntimeDispatcher,
    InnerRuntimeErrorPayload,
    InnerRuntimeFinishPayload,
    InnerRuntimeRoundPayload,
    InnerRuntimeStartPayload,
)

FitPredictFn = Callable[..., Mapping[str, Any]]
BuildIntervalBoundsFn = Callable[..., tuple[np.ndarray, np.ndarray, dict[str, Any]]]
SummarizeFoldFn = Callable[..., dict[str, Any]]
BatchedPredictFn = Callable[..., tuple[np.ndarray, np.ndarray]]
SymmetricIntervalBatchFn = Callable[..., tuple[np.ndarray, np.ndarray, np.ndarray]]
IntervalMetricsBatchFn = Callable[..., Mapping[str, np.ndarray]]
RmseFn = Callable[[np.ndarray, np.ndarray], float]


def _build_inner_runtime_context(
    base_context: Mapping[str, Any] | None,
    *,
    runtime_key: str,
    trainer_name: str = "branch_evaluator",
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    merged = {} if base_context is None else dict(base_context)
    run_id = str(merged.get("run_id", merged.get("task_id", runtime_key)))
    suffix = str(extra.get("run_suffix")) if extra and extra.get("run_suffix") is not None else ""
    merged["run_id"] = f"{run_id}:{suffix}" if suffix else run_id
    merged.setdefault("task_id", run_id)
    merged["runtime_key"] = str(runtime_key)
    merged["trainer_name"] = str(merged.get("trainer_name", trainer_name))
    if extra:
        for key, value in dict(extra).items():
            if value is not None and str(key) != "run_suffix":
                merged[str(key)] = value
    return merged


def _resolve_execution_resource_grant(
    context: Mapping[str, Any] | None,
) -> ExecutionResourceGrant | None:
    if context is None:
        return None
    raw = dict(context).get(EXECUTION_RESOURCE_GRANT_KEY)
    if raw is None:
        return None
    return coerce_execution_resource_grant(raw, phase="branch_evaluation")


def _resolve_effective_resource_offer(
    context: Mapping[str, Any] | None,
):
    grant = _resolve_execution_resource_grant(context)
    local_offer = detect_local_execution_offer()
    effective_offer = constrain_execution_offer_to_grant(local_offer, grant)
    return effective_offer, grant


def _append_usage_report(
    context: Mapping[str, Any] | None,
    report_payload: Mapping[str, Any] | None,
) -> None:
    if context is None or report_payload is None:
        return
    if not isinstance(context, dict):
        return
    reports = list(context.get(EXECUTION_USAGE_REPORTS_KEY, ()))
    reports.append(dict(report_payload))
    context[EXECUTION_USAGE_REPORTS_KEY] = reports


def _emit_inner_start(
    dispatcher: InnerRuntimeDispatcher | None,
    *,
    context: Mapping[str, Any],
    total_rounds: int,
    input_shape: tuple[int, int],
    seed_terms: int,
    metadata: Mapping[str, Any],
) -> None:
    if dispatcher is None or not dispatcher.enabled:
        return
    dispatcher.emit_start(
        InnerRuntimeStartPayload(
            run_id=str(context.get("run_id", context.get("task_id", "inner_runtime"))),
            runtime_key=str(context.get("runtime_key", "branch_evaluation")),
            trainer_name=str(context.get("trainer_name", "branch_evaluator")),
            total_rounds=int(total_rounds),
            input_shape=(int(input_shape[0]), int(input_shape[1])),
            seed_terms=int(seed_terms),
            context=context,
            metadata=metadata,
        )
    )


def _emit_inner_round(
    dispatcher: InnerRuntimeDispatcher | None,
    *,
    context: Mapping[str, Any],
    round_index: int,
    total_rounds: int,
    genome_size: int,
    history_entry: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> None:
    if dispatcher is None or not dispatcher.enabled:
        return
    dispatcher.emit_round_end(
        InnerRuntimeRoundPayload(
            run_id=str(context.get("run_id", context.get("task_id", "inner_runtime"))),
            runtime_key=str(context.get("runtime_key", "branch_evaluation")),
            trainer_name=str(context.get("trainer_name", "branch_evaluator")),
            round_index=int(round_index),
            total_rounds=int(total_rounds),
            genome_size=int(genome_size),
            history_entry=dict(history_entry),
            context=context,
            metadata=metadata,
        )
    )


def _emit_inner_finish(
    dispatcher: InnerRuntimeDispatcher | None,
    *,
    context: Mapping[str, Any],
    total_rounds: int,
    completed_rounds: int,
    genome_size: int,
    final_metrics: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> None:
    if dispatcher is None or not dispatcher.enabled:
        return
    dispatcher.emit_finish(
        InnerRuntimeFinishPayload(
            run_id=str(context.get("run_id", context.get("task_id", "inner_runtime"))),
            runtime_key=str(context.get("runtime_key", "branch_evaluation")),
            trainer_name=str(context.get("trainer_name", "branch_evaluator")),
            total_rounds=int(total_rounds),
            completed_rounds=int(completed_rounds),
            genome_size=int(genome_size),
            final_metrics=dict(final_metrics),
            context=context,
            metadata=metadata,
        )
    )


def _emit_inner_error(
    dispatcher: InnerRuntimeDispatcher | None,
    *,
    context: Mapping[str, Any],
    error: Exception,
    round_index: int | None,
    metadata: Mapping[str, Any],
) -> None:
    if dispatcher is None or not dispatcher.enabled:
        return
    dispatcher.emit_error(
        InnerRuntimeErrorPayload(
            run_id=str(context.get("run_id", context.get("task_id", "inner_runtime"))),
            runtime_key=str(context.get("runtime_key", "branch_evaluation")),
            trainer_name=str(context.get("trainer_name", "branch_evaluator")),
            error=f"{type(error).__name__}: {error}",
            round_index=(None if round_index is None else int(round_index)),
            context=context,
            metadata=metadata,
        )
    )


@dataclass(frozen=True)
class BranchEvaluationConfig:
    regime_branch_mode: bool
    regime_gate_idx: tuple[int, int, int, int] | None
    base_regime_min_branch_train: int
    regime_branch_parallel_workers: int
    interval_alpha: float
    regime_policy: RegimePolicy = field(default_factory=Strict4RouterSpec)

    @property
    def strict4_branch_mode(self) -> bool:
        return self.regime_branch_mode

    @property
    def strict4_gate_idx(self) -> tuple[int, int, int, int] | None:
        return self.regime_gate_idx

    @property
    def base_strict4_min_branch_train(self) -> int:
        return self.base_regime_min_branch_train

    @property
    def strict4_branch_parallel_workers(self) -> int:
        return self.regime_branch_parallel_workers

    @property
    def strict4_regime_policy(self) -> RegimePolicy:
        return self.regime_policy

    @property
    def strict4_router_spec(self) -> RegimePolicy:
        return self.regime_policy


def evaluate_global_fold(
    *,
    genome: Sequence[Mapping[str, Any]],
    X_fit: np.ndarray,
    y_fit: np.ndarray,
    tr_idx: np.ndarray,
    va_idx: np.ndarray,
    l2: float,
    fit_predict_fn: FitPredictFn,
    build_interval_bounds_fn: BuildIntervalBoundsFn,
    summarize_fold_fn: SummarizeFoldFn,
    inner_runtime_dispatcher: InnerRuntimeDispatcher | None = None,
    inner_runtime_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    runtime_context = _build_inner_runtime_context(
        inner_runtime_context,
        runtime_key="branch_evaluation.global_fold",
        extra={
            "fold_kind": "global",
            "train_size": int(np.asarray(tr_idx, dtype=int).size),
            "val_size": int(np.asarray(va_idx, dtype=int).size),
        },
    )
    runtime_metadata = {
        "mode": "global",
        "l2": float(max(0.0, l2)),
    }
    try:
        resource_offer, grant = _resolve_effective_resource_offer(runtime_context)
        runtime_metadata["resource_budget"] = {
            "fold": assert_phase_resource_budget(
                "fold",
                (
                    ExecutionResourceRequest(
                        threads=1,
                        backend="serial",
                        label="fold:global",
                    ),
                ),
                offer=resource_offer,
            )
        }
        if grant is not None:
            runtime_metadata["execution_resource_grant"] = grant.as_dict()
        _emit_inner_start(
            inner_runtime_dispatcher,
            context=runtime_context,
            total_rounds=1,
            input_shape=(int(np.asarray(tr_idx, dtype=int).size), int(np.asarray(X_fit, dtype=float).shape[1])),
            seed_terms=int(len(tuple(genome))),
            metadata=runtime_metadata,
        )
        xtr = np.asarray(X_fit[tr_idx], dtype=float)
        ytr = np.asarray(y_fit[tr_idx], dtype=float)
        xva = np.asarray(X_fit[va_idx], dtype=float)
        yva = np.asarray(y_fit[va_idx], dtype=float)
        fit = fit_predict_fn(
            genome=genome,
            X_train=xtr,
            y_train=ytr,
            X_eval=xva,
            y_eval=yva,
            l2=float(max(0.0, l2)),
        )
        pred_train = np.asarray(fit.get("pred_train"), dtype=float)
        pred_eval = np.asarray(fit.get("pred_eval"), dtype=float)
        lower, upper, interval_info = build_interval_bounds_fn(
            genome=genome,
            X_train=xtr,
            y_train=ytr,
            X_eval=xva,
            pred_train=pred_train,
            pred_eval=pred_eval,
        )
        result = summarize_fold_fn(
            y_true=yva,
            pred_eval=pred_eval,
            lower=lower,
            upper=upper,
            mode="global",
            branch_detail={"inner_opt_info": fit.get("inner_opt_info", {})},
            interval_info=interval_info,
        )
        _emit_inner_round(
            inner_runtime_dispatcher,
            context=runtime_context,
            round_index=1,
            total_rounds=1,
            genome_size=int(len(tuple(genome))),
            history_entry={
                "mode": "global",
                "train_size": int(xtr.shape[0]),
                "val_size": int(xva.shape[0]),
                "rmse": float(result.get("rmse", float("nan"))),
                "interval_score": float(result.get("interval_score", float("nan"))),
            },
            metadata=runtime_metadata,
        )
        _emit_inner_finish(
            inner_runtime_dispatcher,
            context=runtime_context,
            total_rounds=1,
            completed_rounds=1,
            genome_size=int(len(tuple(genome))),
            final_metrics={
                "rmse": float(result.get("rmse", float("nan"))),
                "interval_score": float(result.get("interval_score", float("nan"))),
                "coverage_error": float(result.get("coverage_error", float("nan"))),
            },
            metadata=(
                dict(runtime_metadata)
                if grant is None
                else {
                    **dict(runtime_metadata),
                    "usage_report": build_execution_usage_report(
                        grant,
                        label="branch_evaluation.global_fold",
                        peak_threads=1,
                        used_threads=1,
                        backend="serial",
                    ).as_dict(),
                }
            ),
        )
        if grant is not None:
            _append_usage_report(
                runtime_context,
                build_execution_usage_report(
                    grant,
                    label="branch_evaluation.global_fold",
                    peak_threads=1,
                    used_threads=1,
                    backend="serial",
                ).as_dict(),
            )
        return result
    except Exception as exc:
        _emit_inner_error(
            inner_runtime_dispatcher,
            context=runtime_context,
            error=exc,
            round_index=None,
            metadata=runtime_metadata,
        )
        raise


def evaluate_regime_fold(
    *,
    genome: Sequence[Mapping[str, Any]],
    X_fit: np.ndarray,
    y_fit: np.ndarray,
    tr_idx: np.ndarray,
    va_idx: np.ndarray,
    l2: float,
    regime_min_branch_train: int,
    config: BranchEvaluationConfig,
    fit_predict_fn: FitPredictFn,
    build_interval_bounds_fn: BuildIntervalBoundsFn,
    summarize_fold_fn: SummarizeFoldFn,
    inner_runtime_dispatcher: InnerRuntimeDispatcher | None = None,
    inner_runtime_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not config.regime_branch_mode or config.regime_gate_idx is None:
        return evaluate_global_fold(
            genome=genome,
            X_fit=X_fit,
            y_fit=y_fit,
            tr_idx=tr_idx,
            va_idx=va_idx,
            l2=l2,
            fit_predict_fn=fit_predict_fn,
            build_interval_bounds_fn=build_interval_bounds_fn,
            summarize_fold_fn=summarize_fold_fn,
            inner_runtime_dispatcher=inner_runtime_dispatcher,
            inner_runtime_context=_build_inner_runtime_context(
                inner_runtime_context,
                runtime_key="branch_evaluation.global_fold",
                extra={"fallback_from": "regime_fold"},
            ),
        )

    runtime_context = _build_inner_runtime_context(
        inner_runtime_context,
        runtime_key="branch_evaluation.regime_fold",
        extra={
            "fold_kind": "regime",
            "train_size": int(np.asarray(tr_idx, dtype=int).size),
            "val_size": int(np.asarray(va_idx, dtype=int).size),
        },
    )
    regime_policy = config.regime_policy
    regime_order = tuple(regime_policy.regime_order)
    resource_offer, grant = _resolve_effective_resource_offer(runtime_context)
    requested_parallel_workers = int(max(1, config.regime_branch_parallel_workers))
    branch_parallel_budget = max(0, int(resource_offer.threads) - 1)
    if requested_parallel_workers <= 1 or branch_parallel_budget <= 0:
        n_workers = 1
    else:
        n_workers = int(
            min(
                resolve_phase_worker_count(requested_parallel_workers, n_tasks=len(regime_order)),
                max(1, branch_parallel_budget),
            )
        )
    runtime_metadata = {
        "mode": "regime_branch",
        "l2": float(max(0.0, l2)),
        "parallel_workers": int(n_workers),
        "parallel_workers_requested": int(requested_parallel_workers),
    }
    try:
        fold_request = ExecutionResourceRequest(
            threads=1,
            backend="serial",
            label="fold:regime",
        )
        branch_requests = (
            tuple(
                ExecutionResourceRequest(
                    threads=1,
                    backend="thread",
                    label=f"branch_worker:{idx + 1}",
                )
                for idx in range(int(n_workers))
            )
            if int(n_workers) > 1
            else tuple()
        )
        runtime_metadata["resource_budget"] = {
            "fold": assert_phase_resource_budget(
                "fold",
                (fold_request,),
                offer=resource_offer,
            ),
            "branch": assert_phase_resource_budget(
                "branch",
                (fold_request, *branch_requests),
                offer=resource_offer,
            ),
        }
        if grant is not None:
            runtime_metadata["execution_resource_grant"] = grant.as_dict()
        _emit_inner_start(
            inner_runtime_dispatcher,
            context=runtime_context,
            total_rounds=int(len(regime_order)),
            input_shape=(int(np.asarray(tr_idx, dtype=int).size), int(np.asarray(X_fit, dtype=float).shape[1])),
            seed_terms=int(len(tuple(genome))),
            metadata=runtime_metadata,
        )
        xtr = np.asarray(X_fit[tr_idx], dtype=float)
        ytr = np.asarray(y_fit[tr_idx], dtype=float)
        xva = np.asarray(X_fit[va_idx], dtype=float)
        yva = np.asarray(y_fit[va_idx], dtype=float)

        holiday_keys = tuple(regime_policy.holiday_keys)

        keys_tr = regime_keys_from_X(xtr, config.regime_gate_idx, regime_policy=regime_policy)
        keys_va = regime_keys_from_X(xva, config.regime_gate_idx, regime_policy=regime_policy)
        idx_tr_by_key = build_regime_index(keys_tr, regime_order, router_spec=regime_policy)
        idx_va_by_key = build_regime_index(keys_va, regime_order, router_spec=regime_policy)

        fit_global = fit_predict_fn(
            genome=genome,
            X_train=xtr,
            y_train=ytr,
            X_eval=xva,
            y_eval=yva,
            l2=float(max(0.0, l2)),
        )
        pred_global = np.asarray(fit_global.get("pred_eval"), dtype=float).reshape(-1, 1)
        pred_train_global = np.asarray(fit_global.get("pred_train"), dtype=float).reshape(-1, 1)
        lower_global, upper_global, global_interval_info = build_interval_bounds_fn(
            genome=genome,
            X_train=xtr,
            y_train=ytr,
            X_eval=xva,
            pred_train=pred_train_global,
            pred_eval=pred_global,
        )

        pred_va = np.asarray(pred_global, dtype=float).copy()
        lower_va = np.asarray(lower_global, dtype=float).copy()
        upper_va = np.asarray(upper_global, dtype=float).copy()
        branch_rmse: dict[str, float] = {}
        branch_used_train: dict[str, int] = {}
        branch_used_fallback: dict[str, bool] = {}
        branch_train_source: dict[str, str] = {}

        def _fit_branch(
            regime: tuple[int, int, int, int],
        ) -> tuple[tuple[int, int, int, int], np.ndarray | None, np.ndarray | None, np.ndarray | None, bool, str, int]:
            va_local = np.asarray(idx_va_by_key[regime], dtype=int)
            if int(va_local.size) <= 0:
                train_size = int(np.asarray(idx_tr_by_key[regime], dtype=int).size)
                return regime, None, None, None, True, "self", train_size
            branch_sel = resolve_branch_train_selection(
                regime=regime,
                idx_tr_by_key=idx_tr_by_key,
                regime_min_branch_train=int(regime_min_branch_train),
                base_regime_min_branch_train=int(config.base_regime_min_branch_train),
                total_train_size=int(xtr.shape[0]),
                holiday_keys=holiday_keys,
                router_spec=regime_policy,
            )
            train_used = np.asarray(branch_sel.train_used, dtype=int)
            train_source = str(branch_sel.train_source)
            if not bool(branch_sel.use_branch):
                return regime, None, None, None, True, train_source, int(train_used.size)
            fit = fit_predict_fn(
                genome=genome,
                X_train=xtr[train_used],
                y_train=ytr[train_used],
                X_eval=xva[va_local],
                y_eval=yva[va_local],
                l2=float(max(0.0, l2)),
            )
            pred = np.asarray(fit.get("pred_eval"), dtype=float).reshape(-1, 1)
            pred_train = np.asarray(fit.get("pred_train"), dtype=float).reshape(-1, 1)
            lower, upper, _interval_info = build_interval_bounds_fn(
                genome=genome,
                X_train=xtr[train_used],
                y_train=ytr[train_used],
                X_eval=xva[va_local],
                pred_train=pred_train,
                pred_eval=pred,
            )
            return regime, pred, lower, upper, False, train_source, int(train_used.size)

        if n_workers <= 1:
            branch_results = [_fit_branch(regime) for regime in regime_order]
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
                futures = [executor.submit(_fit_branch, regime) for regime in regime_order]
                branch_results = [future.result() for future in futures]

        completed_rounds = 0
        for completed_rounds, (regime, pred_k, lower_k, upper_k, fallback, train_source, train_size) in enumerate(
            branch_results,
            start=1,
        ):
            va_local = np.asarray(idx_va_by_key[regime], dtype=int)
            if int(va_local.size) > 0 and pred_k is not None and not bool(fallback):
                pred_va[va_local] = np.asarray(pred_k, dtype=float)
                lower_va[va_local] = np.asarray(lower_k, dtype=float)
                upper_va[va_local] = np.asarray(upper_k, dtype=float)
            if int(va_local.size) > 0:
                yk = np.asarray(yva[va_local], dtype=float).reshape(-1)
                pk = np.asarray(pred_va[va_local], dtype=float).reshape(-1)
                branch_rmse[str(regime)] = float(np.sqrt(np.mean((pk - yk) ** 2)))
            branch_used_train[str(regime)] = int(train_size)
            branch_used_fallback[str(regime)] = bool(fallback)
            branch_train_source[str(regime)] = str(train_source)
            _emit_inner_round(
                inner_runtime_dispatcher,
                context=runtime_context,
                round_index=int(completed_rounds),
                total_rounds=int(len(regime_order)),
                genome_size=int(len(tuple(genome))),
                history_entry={
                    "regime": tuple(int(v) for v in regime),
                    "val_size": int(va_local.size),
                    "train_size": int(train_size),
                    "fallback": bool(fallback),
                    "train_source": str(train_source),
                    "rmse": None if str(regime) not in branch_rmse else float(branch_rmse[str(regime)]),
                },
                metadata=runtime_metadata,
            )

        result = summarize_fold_fn(
            y_true=yva,
            pred_eval=pred_va,
            lower=lower_va,
            upper=upper_va,
            mode="strict4_branch",
            branch_detail={
                "branch_rmse": dict(branch_rmse),
                "branch_train_size": dict(branch_used_train),
                "branch_fallback": dict(branch_used_fallback),
                "branch_train_source": dict(branch_train_source),
            },
            interval_info=global_interval_info,
        )
        _emit_inner_finish(
            inner_runtime_dispatcher,
            context=runtime_context,
            total_rounds=int(len(regime_order)),
            completed_rounds=int(completed_rounds),
            genome_size=int(len(tuple(genome))),
            final_metrics={
                "rmse": float(result.get("rmse", float("nan"))),
                "interval_score": float(result.get("interval_score", float("nan"))),
                "coverage_error": float(result.get("coverage_error", float("nan"))),
            },
            metadata=(
                dict(runtime_metadata)
                if grant is None
                else {
                    **dict(runtime_metadata),
                    "usage_report": build_execution_usage_report(
                        grant,
                        label="branch_evaluation.regime_fold",
                        peak_threads=(1 if int(n_workers) <= 1 else int(1 + n_workers)),
                        used_threads=(1 if int(n_workers) <= 1 else int(1 + n_workers)),
                        backend=("serial" if int(n_workers) <= 1 else "thread"),
                    ).as_dict(),
                }
            ),
        )
        if grant is not None:
            _append_usage_report(
                runtime_context,
                build_execution_usage_report(
                    grant,
                    label="branch_evaluation.regime_fold",
                    peak_threads=(1 if int(n_workers) <= 1 else int(1 + n_workers)),
                    used_threads=(1 if int(n_workers) <= 1 else int(1 + n_workers)),
                    backend=("serial" if int(n_workers) <= 1 else "thread"),
                ).as_dict(),
            )
        return result
    except Exception as exc:
        _emit_inner_error(
            inner_runtime_dispatcher,
            context=runtime_context,
            error=exc,
            round_index=None,
            metadata=runtime_metadata,
        )
        raise


def evaluate_strict4_fold(
    *,
    genome: Sequence[Mapping[str, Any]],
    X_fit: np.ndarray,
    y_fit: np.ndarray,
    tr_idx: np.ndarray,
    va_idx: np.ndarray,
    l2: float,
    strict4_min_branch_train: int,
    config: BranchEvaluationConfig,
    fit_predict_fn: FitPredictFn,
    build_interval_bounds_fn: BuildIntervalBoundsFn,
    summarize_fold_fn: SummarizeFoldFn,
) -> dict[str, Any]:
    return evaluate_regime_fold(
        genome=genome,
        X_fit=X_fit,
        y_fit=y_fit,
        tr_idx=tr_idx,
        va_idx=va_idx,
        l2=l2,
        regime_min_branch_train=strict4_min_branch_train,
        config=config,
        fit_predict_fn=fit_predict_fn,
        build_interval_bounds_fn=build_interval_bounds_fn,
        summarize_fold_fn=summarize_fold_fn,
    )


def evaluate_symmetric_residual_fold_batch(
    *,
    genomes: Sequence[Sequence[Mapping[str, Any]]],
    metas: Sequence[Mapping[str, Any]],
    X_fit: np.ndarray,
    y_fit: np.ndarray,
    tr_idx: np.ndarray,
    va_idx: np.ndarray,
    base_ridge_l2: float,
    config: BranchEvaluationConfig,
    batched_predict_fn: BatchedPredictFn,
    symmetric_interval_batch_fn: SymmetricIntervalBatchFn,
    interval_metrics_batch_fn: IntervalMetricsBatchFn,
    summarize_fold_fn: SummarizeFoldFn,
    rmse_fn: RmseFn,
    graph_cache: Any = None,
    batch_key_prefix: str = "",
    inner_runtime_dispatcher: InnerRuntimeDispatcher | None = None,
    inner_runtime_context: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    batch_size = int(len(genomes))
    if batch_size <= 0:
        return []

    runtime_context = _build_inner_runtime_context(
        inner_runtime_context,
        runtime_key="branch_evaluation.fold_batch",
        extra={
            "fold_kind": "batch_interval",
            "batch_key_prefix": str(batch_key_prefix),
            "batch_size": int(batch_size),
            "train_size": int(np.asarray(tr_idx, dtype=int).size),
            "val_size": int(np.asarray(va_idx, dtype=int).size),
        },
    )
    x_fit_arr = np.asarray(X_fit, dtype=float)
    total_rounds = (
        1
        if (not config.regime_branch_mode or config.regime_gate_idx is None)
        else int(len(tuple(config.regime_policy.regime_order)))
    )
    runtime_metadata = {
        "mode": (
            "global"
            if (not config.regime_branch_mode or config.regime_gate_idx is None)
            else "strict4_branch"
        ),
        "base_ridge_l2": float(base_ridge_l2),
        "batch_key_prefix": str(batch_key_prefix),
    }
    try:
        resource_offer, grant = _resolve_effective_resource_offer(runtime_context)
        runtime_metadata["resource_budget"] = {
            "fold": assert_phase_resource_budget(
                "fold",
                (
                    ExecutionResourceRequest(
                        threads=1,
                        backend="serial",
                        label="fold:batch_interval",
                    ),
                ),
                offer=resource_offer,
            )
        }
        if grant is not None:
            runtime_metadata["execution_resource_grant"] = grant.as_dict()
        _emit_inner_start(
            inner_runtime_dispatcher,
            context=runtime_context,
            total_rounds=int(total_rounds),
            input_shape=(int(np.asarray(tr_idx, dtype=int).size), int(x_fit_arr.shape[1])),
            seed_terms=int(batch_size),
            metadata=runtime_metadata,
        )
        xtr = np.asarray(X_fit[tr_idx], dtype=float)
        ytr = np.asarray(y_fit[tr_idx], dtype=float)
        xva = np.asarray(X_fit[va_idx], dtype=float)
        yva = np.asarray(y_fit[va_idx], dtype=float)

        l2s = [float(max(0.0, meta.get("tuned_l2", base_ridge_l2))) for meta in metas]
        pred_global, pred_train_global = batched_predict_fn(
            genomes=genomes,
            X_train=xtr,
            y_train=ytr,
            X_eval=xva,
            l2_values=l2s,
            graph_cache=graph_cache,
            batch_key_train=f"{batch_key_prefix}|global|tr",
            batch_key_eval=f"{batch_key_prefix}|global|va",
        )
        lower_global, upper_global, q_global = symmetric_interval_batch_fn(
            y_train=ytr,
            pred_train=pred_train_global,
            pred_eval=pred_global,
            alpha=float(config.interval_alpha),
        )

        if not config.regime_branch_mode or config.regime_gate_idx is None:
            metrics_all = interval_metrics_batch_fn(
                y_true=yva,
                lower=lower_global,
                upper=upper_global,
                alpha=float(config.interval_alpha),
            )
            results: list[dict[str, Any]] = []
            for idx in range(batch_size):
                results.append(
                    summarize_fold_fn(
                        y_true=yva,
                        pred_eval=pred_global[idx],
                        lower=lower_global[idx],
                        upper=upper_global[idx],
                        mode="global",
                        branch_detail={},
                        interval_info={"method": "symmetric_residual", "conformal_qhat": float(q_global[idx])},
                        precomputed_interval_metrics={
                            "coverage_error": float(metrics_all["coverage_error"][idx]),
                            "picp": float(metrics_all["picp"][idx]),
                            "pinaw": float(metrics_all["pinaw"][idx]),
                            "interval_score": float(metrics_all["interval_score"][idx]),
                            "mean_width": float(metrics_all["mean_width"][idx]),
                            "coverage_target": float(metrics_all["coverage_target"][idx]),
                        },
                        precomputed_rmse=float(rmse_fn(yva, pred_global[idx])),
                    )
                )
            _emit_inner_round(
                inner_runtime_dispatcher,
                context=runtime_context,
                round_index=1,
                total_rounds=1,
                genome_size=int(batch_size),
                history_entry={
                    "mode": "global",
                    "batch_size": int(batch_size),
                    "val_size": int(xva.shape[0]),
                },
                metadata=runtime_metadata,
            )
            _emit_inner_finish(
                inner_runtime_dispatcher,
                context=runtime_context,
                total_rounds=1,
                completed_rounds=1,
                genome_size=int(batch_size),
                final_metrics={
                    "batch_size": int(batch_size),
                    "mean_rmse": float(np.mean([float(row.get("rmse", 0.0)) for row in results])),
                },
                metadata=(
                    dict(runtime_metadata)
                    if grant is None
                    else {
                        **dict(runtime_metadata),
                        "usage_report": build_execution_usage_report(
                            grant,
                            label="branch_evaluation.fold_batch",
                            peak_threads=1,
                            used_threads=1,
                            backend="serial",
                        ).as_dict(),
                    }
                ),
            )
            if grant is not None:
                _append_usage_report(
                    runtime_context,
                    build_execution_usage_report(
                        grant,
                        label="branch_evaluation.fold_batch",
                        peak_threads=1,
                        used_threads=1,
                        backend="serial",
                    ).as_dict(),
                )
            return results

        regime_policy = config.regime_policy
        regime_order = tuple(regime_policy.regime_order)
        holiday_keys = tuple(regime_policy.holiday_keys)
        keys_tr = regime_keys_from_X(xtr, config.regime_gate_idx, regime_policy=regime_policy)
        keys_va = regime_keys_from_X(xva, config.regime_gate_idx, regime_policy=regime_policy)
        idx_tr_by_key = build_regime_index(keys_tr, regime_order, router_spec=regime_policy)
        idx_va_by_key = build_regime_index(keys_va, regime_order, router_spec=regime_policy)
        holiday_union_tr = holiday_union_indices(
            idx_by_key=idx_tr_by_key,
            holiday_keys=holiday_keys,
            router_spec=regime_policy,
        )

        pred_va = np.asarray(pred_global, dtype=float).copy()
        lower_va = np.asarray(lower_global, dtype=float).copy()
        upper_va = np.asarray(upper_global, dtype=float).copy()
        branch_detail_all: list[dict[str, Any]] = [
            {"branch_rmse": {}, "branch_train_size": {}, "branch_fallback": {}, "branch_train_source": {}}
            for _ in range(batch_size)
        ]

        completed_rounds = 0
        for completed_rounds, regime in enumerate(regime_order, start=1):
            va_local = np.asarray(idx_va_by_key[regime], dtype=int)
            if int(va_local.size) <= 0:
                _emit_inner_round(
                    inner_runtime_dispatcher,
                    context=runtime_context,
                    round_index=int(completed_rounds),
                    total_rounds=int(len(regime_order)),
                    genome_size=int(batch_size),
                    history_entry={
                        "regime": tuple(int(v) for v in regime),
                        "val_size": 0,
                        "active_self": 0,
                        "active_union": 0,
                    },
                    metadata=runtime_metadata,
                )
                continue

            active_self: list[int] = []
            active_union: list[int] = []
            for batch_idx, meta in enumerate(metas):
                regime_min_train_ratio = float(np.clip(meta.get("strict4_min_train_ratio", 0.08), 0.01, 0.30))
                min_train = int(max(config.base_regime_min_branch_train, round(regime_min_train_ratio * float(tr_idx.size))))
                branch_sel = resolve_branch_train_selection(
                    regime=regime,
                    idx_tr_by_key=idx_tr_by_key,
                    regime_min_branch_train=int(min_train),
                    base_regime_min_branch_train=int(config.base_regime_min_branch_train),
                    total_train_size=int(tr_idx.size),
                    holiday_keys=holiday_keys,
                    router_spec=regime_policy,
                )
                train_used = np.asarray(branch_sel.train_used, dtype=int)
                train_source = str(branch_sel.train_source)
                use_branch = bool(branch_sel.use_branch)
                branch_detail_all[batch_idx]["branch_train_size"][str(regime)] = int(train_used.size)
                branch_detail_all[batch_idx]["branch_fallback"][str(regime)] = bool(not use_branch)
                branch_detail_all[batch_idx]["branch_train_source"][str(regime)] = train_source
                if use_branch:
                    if train_source == "holiday_union":
                        active_union.append(int(batch_idx))
                    else:
                        active_self.append(int(batch_idx))

            if active_self:
                genomes_act = [genomes[idx] for idx in active_self]
                l2s_act = [l2s[idx] for idx in active_self]
                tr_local = np.asarray(idx_tr_by_key[regime], dtype=int)
                pred_loc, pred_train_loc = batched_predict_fn(
                    genomes=genomes_act,
                    X_train=xtr[tr_local],
                    y_train=ytr[tr_local],
                    X_eval=xva[va_local],
                    l2_values=l2s_act,
                    graph_cache=graph_cache,
                    batch_key_train=f"{batch_key_prefix}|{str(regime)}|tr",
                    batch_key_eval=f"{batch_key_prefix}|{str(regime)}|va",
                )
                lo_loc, hi_loc, _ = symmetric_interval_batch_fn(
                    y_train=ytr[tr_local],
                    pred_train=pred_train_loc,
                    pred_eval=pred_loc,
                    alpha=float(config.interval_alpha),
                )
                for loc, batch_idx in enumerate(active_self):
                    pred_va[batch_idx, va_local, :] = pred_loc[loc]
                    lower_va[batch_idx, va_local, :] = lo_loc[loc]
                    upper_va[batch_idx, va_local, :] = hi_loc[loc]

            if active_union and int(holiday_union_tr.size) > 0:
                genomes_act = [genomes[idx] for idx in active_union]
                l2s_act = [l2s[idx] for idx in active_union]
                pred_loc, pred_train_loc = batched_predict_fn(
                    genomes=genomes_act,
                    X_train=xtr[holiday_union_tr],
                    y_train=ytr[holiday_union_tr],
                    X_eval=xva[va_local],
                    l2_values=l2s_act,
                    graph_cache=graph_cache,
                    batch_key_train=f"{batch_key_prefix}|{str(regime)}|holiday_union|tr",
                    batch_key_eval=f"{batch_key_prefix}|{str(regime)}|holiday_union|va",
                )
                lo_loc, hi_loc, _ = symmetric_interval_batch_fn(
                    y_train=ytr[holiday_union_tr],
                    pred_train=pred_train_loc,
                    pred_eval=pred_loc,
                    alpha=float(config.interval_alpha),
                )
                for loc, batch_idx in enumerate(active_union):
                    pred_va[batch_idx, va_local, :] = pred_loc[loc]
                    lower_va[batch_idx, va_local, :] = lo_loc[loc]
                    upper_va[batch_idx, va_local, :] = hi_loc[loc]

            for batch_idx in range(batch_size):
                branch_detail_all[batch_idx]["branch_rmse"][str(regime)] = float(
                    rmse_fn(yva[va_local], pred_va[batch_idx, va_local, :])
                )

            _emit_inner_round(
                inner_runtime_dispatcher,
                context=runtime_context,
                round_index=int(completed_rounds),
                total_rounds=int(len(regime_order)),
                genome_size=int(batch_size),
                history_entry={
                    "regime": tuple(int(v) for v in regime),
                    "val_size": int(va_local.size),
                    "active_self": int(len(active_self)),
                    "active_union": int(len(active_union)),
                },
                metadata=runtime_metadata,
            )

        metrics_all = interval_metrics_batch_fn(
            y_true=yva,
            lower=lower_va,
            upper=upper_va,
            alpha=float(config.interval_alpha),
        )
        results = []
        for idx in range(batch_size):
            results.append(
                summarize_fold_fn(
                    y_true=yva,
                    pred_eval=pred_va[idx],
                    lower=lower_va[idx],
                    upper=upper_va[idx],
                    mode="strict4_branch",
                    branch_detail=dict(branch_detail_all[idx]),
                    interval_info={"method": "symmetric_residual", "conformal_qhat": float(q_global[idx])},
                    precomputed_interval_metrics={
                        "coverage_error": float(metrics_all["coverage_error"][idx]),
                        "picp": float(metrics_all["picp"][idx]),
                        "pinaw": float(metrics_all["pinaw"][idx]),
                        "interval_score": float(metrics_all["interval_score"][idx]),
                        "mean_width": float(metrics_all["mean_width"][idx]),
                        "coverage_target": float(metrics_all["coverage_target"][idx]),
                    },
                    precomputed_rmse=float(rmse_fn(yva, pred_va[idx])),
                )
            )
        _emit_inner_finish(
            inner_runtime_dispatcher,
            context=runtime_context,
            total_rounds=int(len(regime_order)),
            completed_rounds=int(completed_rounds),
            genome_size=int(batch_size),
            final_metrics={
                "batch_size": int(batch_size),
                "mean_rmse": float(np.mean([float(row.get("rmse", 0.0)) for row in results])),
            },
            metadata=(
                dict(runtime_metadata)
                if grant is None
                else {
                    **dict(runtime_metadata),
                    "usage_report": build_execution_usage_report(
                        grant,
                        label="branch_evaluation.fold_batch",
                        peak_threads=1,
                        used_threads=1,
                        backend="serial",
                    ).as_dict(),
                }
            ),
        )
        if grant is not None:
            _append_usage_report(
                runtime_context,
                build_execution_usage_report(
                    grant,
                    label="branch_evaluation.fold_batch",
                    peak_threads=1,
                    used_threads=1,
                    backend="serial",
                ).as_dict(),
            )
        return results
    except Exception as exc:
        _emit_inner_error(
            inner_runtime_dispatcher,
            context=runtime_context,
            error=exc,
            round_index=None,
            metadata=runtime_metadata,
        )
        raise


__all__ = [
    "BranchEvaluationConfig",
    "evaluate_global_fold",
    "evaluate_strict4_fold",
    "evaluate_symmetric_residual_fold_batch",
]
