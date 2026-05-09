from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from core.execution import ExecutionBudgetError, ExecutionResourceOffer, issue_execution_resource_grant
from core.symbolic.feature_space.branch_evaluator import (
    BranchEvaluationConfig,
    evaluate_regime_fold,
    evaluate_symmetric_residual_fold_batch,
)
from training import (
    InnerRuntimeDispatcher,
    InnerRuntimeErrorPayload,
    InnerRuntimeFinishPayload,
    InnerRuntimeRoundPayload,
    InnerRuntimeStartPayload,
)


def _make_regime_dataset() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    regimes = (
        (1, 1, 0, 0),
        (1, 0, 1, 0),
        (0, 0, 0, 1),
        (0, 0, 0, 0),
    )
    rows: list[np.ndarray] = []
    targets: list[list[float]] = []
    train_idx: list[int] = []
    val_idx: list[int] = []
    idx = 0
    for regime_offset, regime in enumerate(regimes):
        for sample_idx in range(5):
            cont0 = float(regime_offset) + 0.1 * float(sample_idx)
            cont1 = 0.2 * float(sample_idx)
            row = np.asarray([*regime, cont0, cont1], dtype=float)
            target = 0.6 * cont0 - 0.25 * cont1 + 0.15 * float(sum(regime))
            rows.append(row)
            targets.append([target])
            if sample_idx < 3:
                train_idx.append(idx)
            else:
                val_idx.append(idx)
            idx += 1
    return (
        np.asarray(rows, dtype=float),
        np.asarray(targets, dtype=float),
        np.asarray(train_idx, dtype=int),
        np.asarray(val_idx, dtype=int),
    )


def _fit_predict_stub(
    *,
    genome,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    l2: float,
):
    _ = (genome, y_eval, l2)
    coef = np.asarray([[0.6], [-0.25]], dtype=float)
    pred_train = np.asarray(X_train[:, -2:], dtype=float) @ coef
    pred_eval = np.asarray(X_eval[:, -2:], dtype=float) @ coef
    return {
        "pred_train": np.asarray(pred_train, dtype=float),
        "pred_eval": np.asarray(pred_eval, dtype=float),
        "inner_opt_info": {"kind": "stub"},
    }


def _build_interval_bounds_stub(
    *,
    genome,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    pred_train: np.ndarray,
    pred_eval: np.ndarray,
):
    _ = (genome, X_train, y_train, X_eval, pred_train)
    lower = np.asarray(pred_eval, dtype=float) - 0.25
    upper = np.asarray(pred_eval, dtype=float) + 0.25
    return lower, upper, {"method": "stub_interval"}


def _summarize_fold_stub(
    *,
    y_true: np.ndarray,
    pred_eval: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    mode: str,
    branch_detail,
    interval_info,
    precomputed_interval_metrics=None,
    precomputed_rmse=None,
):
    rmse = (
        float(precomputed_rmse)
        if precomputed_rmse is not None
        else float(np.sqrt(np.mean((np.asarray(pred_eval, dtype=float) - np.asarray(y_true, dtype=float)) ** 2)))
    )
    if precomputed_interval_metrics is None:
        mean_width = float(np.mean(np.asarray(upper, dtype=float) - np.asarray(lower, dtype=float)))
        coverage_target = 0.9
        coverage = float(
            np.mean(
                (np.asarray(y_true, dtype=float) >= np.asarray(lower, dtype=float))
                & (np.asarray(y_true, dtype=float) <= np.asarray(upper, dtype=float))
            )
        )
        coverage_error = abs(coverage - coverage_target)
        interval_score = mean_width + coverage_error
    else:
        mean_width = float(precomputed_interval_metrics["mean_width"])
        coverage_target = float(precomputed_interval_metrics["coverage_target"])
        coverage_error = float(precomputed_interval_metrics["coverage_error"])
        interval_score = float(precomputed_interval_metrics["interval_score"])
    return {
        "mode": str(mode),
        "rmse": float(rmse),
        "interval_score": float(interval_score),
        "coverage_error": float(coverage_error),
        "mean_width": float(mean_width),
        "coverage_target": float(coverage_target),
        "branch_detail": dict(branch_detail),
        "interval_info": dict(interval_info),
    }


def _batched_predict_stub(
    *,
    genomes,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    l2_values,
    graph_cache,
    batch_key_train: str,
    batch_key_eval: str,
):
    _ = (genomes, y_train, l2_values, graph_cache, batch_key_train, batch_key_eval)
    batch_size = int(len(l2_values))
    coef = np.asarray([[0.6], [-0.25]], dtype=float)
    pred_train_single = np.asarray(X_train[:, -2:], dtype=float) @ coef
    pred_eval_single = np.asarray(X_eval[:, -2:], dtype=float) @ coef
    pred_train = np.stack([pred_train_single for _ in range(batch_size)], axis=0)
    pred_eval = np.stack([pred_eval_single for _ in range(batch_size)], axis=0)
    return pred_eval, pred_train


def _symmetric_interval_batch_stub(
    *,
    y_train: np.ndarray,
    pred_train: np.ndarray,
    pred_eval: np.ndarray,
    alpha: float,
):
    _ = (y_train, pred_train, alpha)
    lower = np.asarray(pred_eval, dtype=float) - 0.3
    upper = np.asarray(pred_eval, dtype=float) + 0.3
    qhat = np.asarray([0.3] * int(np.asarray(pred_eval).shape[0]), dtype=float)
    return lower, upper, qhat


def _interval_metrics_batch_stub(
    *,
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    alpha: float,
):
    _ = alpha
    batch_size = int(np.asarray(lower).shape[0])
    lower_arr = np.asarray(lower, dtype=float)
    upper_arr = np.asarray(upper, dtype=float)
    y_true_arr = np.asarray(y_true, dtype=float)
    mean_width = np.mean(upper_arr - lower_arr, axis=(1, 2))
    coverage = np.mean((y_true_arr[None, :, :] >= lower_arr) & (y_true_arr[None, :, :] <= upper_arr), axis=(1, 2))
    coverage_target = np.full((batch_size,), 0.9, dtype=float)
    coverage_error = np.abs(coverage - coverage_target)
    interval_score = mean_width + coverage_error
    pinaw = np.asarray(mean_width, dtype=float)
    return {
        "coverage_error": np.asarray(coverage_error, dtype=float),
        "picp": np.asarray(coverage, dtype=float),
        "pinaw": np.asarray(pinaw, dtype=float),
        "interval_score": np.asarray(interval_score, dtype=float),
        "mean_width": np.asarray(mean_width, dtype=float),
        "coverage_target": np.asarray(coverage_target, dtype=float),
    }


def _rmse_stub(y_true: np.ndarray, pred_eval: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(pred_eval, dtype=float) - np.asarray(y_true, dtype=float)) ** 2)))


class _CollectingHook:
    def __init__(self) -> None:
        self.events: list[tuple[str, object]] = []

    def on_inner_run_start(self, payload: InnerRuntimeStartPayload) -> None:
        self.events.append(("start", payload))

    def on_inner_round_end(self, payload: InnerRuntimeRoundPayload) -> None:
        self.events.append(("round", payload))

    def on_inner_run_finish(self, payload: InnerRuntimeFinishPayload) -> None:
        self.events.append(("finish", payload))

    def on_inner_run_error(self, payload: InnerRuntimeErrorPayload) -> None:
        self.events.append(("error", payload))


class TestBranchEvaluatorInnerRuntime(unittest.TestCase):
    def test_regime_fold_emits_branch_loop_events(self) -> None:
        X_fit, y_fit, tr_idx, va_idx = _make_regime_dataset()
        hook = _CollectingHook()
        dispatcher = InnerRuntimeDispatcher.from_hooks((hook,))
        config = BranchEvaluationConfig(
            regime_branch_mode=True,
            regime_gate_idx=(0, 1, 2, 3),
            base_regime_min_branch_train=1,
            regime_branch_parallel_workers=1,
            interval_alpha=0.1,
        )

        result = evaluate_regime_fold(
            genome=({"name": "x"},),
            X_fit=X_fit,
            y_fit=y_fit,
            tr_idx=tr_idx,
            va_idx=va_idx,
            l2=0.0,
            regime_min_branch_train=1,
            config=config,
            fit_predict_fn=_fit_predict_stub,
            build_interval_bounds_fn=_build_interval_bounds_stub,
            summarize_fold_fn=_summarize_fold_stub,
            inner_runtime_dispatcher=dispatcher,
            inner_runtime_context={"task_id": "branch_eval::regime"},
        )

        self.assertIn("rmse", result)
        event_names = [name for name, _ in hook.events]
        self.assertIn("start", event_names)
        self.assertIn("finish", event_names)
        self.assertEqual(event_names.count("round"), 4)
        self.assertNotIn("error", event_names)

        start_payload = next(payload for name, payload in hook.events if name == "start")
        finish_payload = next(payload for name, payload in hook.events if name == "finish")
        self.assertEqual(str(start_payload.runtime_key), "branch_evaluation.regime_fold")
        self.assertEqual(str(start_payload.context.get("task_id")), "branch_eval::regime")
        self.assertEqual(int(finish_payload.completed_rounds), 4)

    def test_fold_batch_emits_fold_batch_events(self) -> None:
        X_fit, y_fit, tr_idx, va_idx = _make_regime_dataset()
        hook = _CollectingHook()
        dispatcher = InnerRuntimeDispatcher.from_hooks((hook,))
        config = BranchEvaluationConfig(
            regime_branch_mode=True,
            regime_gate_idx=(0, 1, 2, 3),
            base_regime_min_branch_train=1,
            regime_branch_parallel_workers=1,
            interval_alpha=0.1,
        )

        results = evaluate_symmetric_residual_fold_batch(
            genomes=[({"name": "x0"},), ({"name": "x1"},)],
            metas=[{"tuned_l2": 0.0, "strict4_min_train_ratio": 0.01}, {"tuned_l2": 0.1, "strict4_min_train_ratio": 0.01}],
            X_fit=X_fit,
            y_fit=y_fit,
            tr_idx=tr_idx,
            va_idx=va_idx,
            base_ridge_l2=0.0,
            config=config,
            batched_predict_fn=_batched_predict_stub,
            symmetric_interval_batch_fn=_symmetric_interval_batch_stub,
            interval_metrics_batch_fn=_interval_metrics_batch_stub,
            summarize_fold_fn=_summarize_fold_stub,
            rmse_fn=_rmse_stub,
            graph_cache=None,
            batch_key_prefix="branch_batch_test",
            inner_runtime_dispatcher=dispatcher,
            inner_runtime_context={"task_id": "branch_eval::batch"},
        )

        self.assertEqual(len(results), 2)
        event_names = [name for name, _ in hook.events]
        self.assertIn("start", event_names)
        self.assertIn("finish", event_names)
        self.assertEqual(event_names.count("round"), 4)
        self.assertNotIn("error", event_names)

        start_payload = next(payload for name, payload in hook.events if name == "start")
        finish_payload = next(payload for name, payload in hook.events if name == "finish")
        self.assertEqual(str(start_payload.runtime_key), "branch_evaluation.fold_batch")
        self.assertEqual(str(start_payload.context.get("task_id")), "branch_eval::batch")
        self.assertEqual(int(finish_payload.completed_rounds), 4)

    def test_regime_fold_clamps_parallel_workers_to_local_offer(self) -> None:
        X_fit, y_fit, tr_idx, va_idx = _make_regime_dataset()
        hook = _CollectingHook()
        dispatcher = InnerRuntimeDispatcher.from_hooks((hook,))
        config = BranchEvaluationConfig(
            regime_branch_mode=True,
            regime_gate_idx=(0, 1, 2, 3),
            base_regime_min_branch_train=1,
            regime_branch_parallel_workers=4,
            interval_alpha=0.1,
        )

        with patch(
            "core.symbolic.feature_space.branch_evaluator.detect_local_execution_offer",
            return_value=ExecutionResourceOffer(threads=4),
        ):
            result = evaluate_regime_fold(
                genome=({"name": "x"},),
                X_fit=X_fit,
                y_fit=y_fit,
                tr_idx=tr_idx,
                va_idx=va_idx,
                l2=0.0,
                regime_min_branch_train=1,
                config=config,
                fit_predict_fn=_fit_predict_stub,
                build_interval_bounds_fn=_build_interval_bounds_stub,
                summarize_fold_fn=_summarize_fold_stub,
                inner_runtime_dispatcher=dispatcher,
                inner_runtime_context={"task_id": "branch_eval::local_offer"},
            )

        self.assertIn("rmse", result)
        start_payload = next(payload for name, payload in hook.events if name == "start")
        branch_budget = dict(dict(start_payload.metadata.get("resource_budget", {})).get("branch", {}))
        total_request = dict(branch_budget.get("total_request", {}))
        self.assertEqual(int(start_payload.metadata.get("parallel_workers", 0)), 3)
        self.assertEqual(int(total_request.get("threads", 0)), 4)

    def test_regime_fold_respects_parent_execution_grant(self) -> None:
        X_fit, y_fit, tr_idx, va_idx = _make_regime_dataset()
        hook = _CollectingHook()
        dispatcher = InnerRuntimeDispatcher.from_hooks((hook,))
        config = BranchEvaluationConfig(
            regime_branch_mode=True,
            regime_gate_idx=(0, 1, 2, 3),
            base_regime_min_branch_train=1,
            regime_branch_parallel_workers=4,
            interval_alpha=0.1,
        )

        result = evaluate_regime_fold(
            genome=({"name": "x"},),
            X_fit=X_fit,
            y_fit=y_fit,
            tr_idx=tr_idx,
            va_idx=va_idx,
            l2=0.0,
            regime_min_branch_train=1,
            config=config,
            fit_predict_fn=_fit_predict_stub,
            build_interval_bounds_fn=_build_interval_bounds_stub,
            summarize_fold_fn=_summarize_fold_stub,
            inner_runtime_dispatcher=dispatcher,
            inner_runtime_context={
                "task_id": "branch_eval::grant",
                "execution_resource_grant": issue_execution_resource_grant(
                    {"threads": 2, "backend": "thread", "label": "problem_eval"},
                    phase="mlblack_inner_problem",
                    label="problem_eval",
                ).as_dict(),
            },
        )

        self.assertIn("rmse", result)
        start_payload = next(payload for name, payload in hook.events if name == "start")
        finish_payload = next(payload for name, payload in hook.events if name == "finish")
        branch_budget = dict(dict(start_payload.metadata.get("resource_budget", {})).get("branch", {}))
        total_request = dict(branch_budget.get("total_request", {}))
        usage_report = dict(finish_payload.metadata.get("usage_report", {}))

        self.assertEqual(int(total_request.get("threads", 0)), 1)
        self.assertEqual(int(start_payload.metadata.get("parallel_workers", 0)), 1)
        self.assertEqual(int(usage_report.get("granted_threads", 0)), 2)
        self.assertEqual(int(usage_report.get("peak_threads", 0)), 1)


if __name__ == "__main__":
    unittest.main()
