from __future__ import annotations

import unittest

from core.execution import issue_execution_resource_grant
from core.symbolic.feature_space.regime_router import Strict4RouterSpec
from evaluation import FitPredictCallbackConfig, IntervalCallbackConfig, ProblemEvaluationCallbacks


def _jsonable_stub(value):
    return value


def _rmse_stub(y_true, y_pred) -> float:
    _ = (y_true, y_pred)
    return 0.0


class TestProblemEvaluationResources(unittest.TestCase):
    def test_problem_callbacks_expose_fold_and_branch_requests(self) -> None:
        callbacks = ProblemEvaluationCallbacks(
            interval_config=IntervalCallbackConfig(
                interval_alpha=0.1,
                interval_method="native_quantile_cqr",
                interval_calib_ratio=0.2,
                interval_quantile_l2=1e-4,
                regime_branch_mode=True,
                regime_gate_idx=(0, 1, 2, 3),
                base_regime_min_branch_train=16,
                regime_branch_parallel_workers=3,
                regime_policy=Strict4RouterSpec(),
            ),
            fit_predict_config=FitPredictCallbackConfig(
                random_seed=42,
                inner_opt_enabled=True,
                inner_opt_adam_steps=20,
                inner_opt_adam_lr=1e-2,
                inner_opt_lbfgs_steps=10,
                inner_opt_lbfgs_lr=0.5,
                inner_opt_accept_rmse_tol=0.0,
                inner_opt_accept_rel_tol=0.0,
                inner_opt_guard_patience=2,
                inner_opt_guard_check_interval=1,
                inner_opt_alt_freeze_readout=False,
                inner_opt_grad_clip_norm=0.0,
                inner_opt_residual_clip_q=0.95,
            ),
            jsonable_fn=_jsonable_stub,
            rmse_fn=_rmse_stub,
        )

        components = callbacks.execution_resource_requests(rolling_folds=6, label="problem_eval")
        total = callbacks.execution_resource_request(rolling_folds=6, label="problem_eval")

        self.assertEqual(len(components), 2)
        self.assertEqual(str(components[0].label), "problem_eval:fold")
        self.assertEqual(str(components[1].label), "problem_eval:branch_workers")
        self.assertEqual(int(components[1].threads), 3)
        self.assertEqual(str(components[1].backend), "thread")
        self.assertEqual(int(total.threads), 4)
        self.assertEqual(str(total.backend), "thread")
        self.assertEqual(int(total.metadata.get("rolling_folds", 0)), 6)
        self.assertTrue(bool(total.metadata.get("regime_branch_mode")))
        self.assertTrue(bool(total.metadata.get("inner_opt_enabled")))

    def test_problem_callbacks_apply_parent_grant_before_declaring_branch_workers(self) -> None:
        callbacks = ProblemEvaluationCallbacks(
            interval_config=IntervalCallbackConfig(
                interval_alpha=0.1,
                interval_method="native_quantile_cqr",
                interval_calib_ratio=0.2,
                interval_quantile_l2=1e-4,
                regime_branch_mode=True,
                regime_gate_idx=(0, 1, 2, 3),
                base_regime_min_branch_train=16,
                regime_branch_parallel_workers=4,
                regime_policy=Strict4RouterSpec(),
            ),
            fit_predict_config=FitPredictCallbackConfig(
                random_seed=42,
                inner_opt_enabled=True,
                inner_opt_adam_steps=20,
                inner_opt_adam_lr=1e-2,
                inner_opt_lbfgs_steps=10,
                inner_opt_lbfgs_lr=0.5,
                inner_opt_accept_rmse_tol=0.0,
                inner_opt_accept_rel_tol=0.0,
                inner_opt_guard_patience=2,
                inner_opt_guard_check_interval=1,
                inner_opt_alt_freeze_readout=False,
                inner_opt_grad_clip_norm=0.0,
                inner_opt_residual_clip_q=0.95,
            ),
            jsonable_fn=_jsonable_stub,
            rmse_fn=_rmse_stub,
        )
        callbacks.set_execution_resource_grant(
            issue_execution_resource_grant(
                {"threads": 2, "backend": "thread", "label": "problem_eval"},
                phase="mlblack_inner_problem",
                label="problem_eval",
            )
        )

        components = callbacks.execution_resource_requests(rolling_folds=6, label="problem_eval")
        total = callbacks.execution_resource_request(rolling_folds=6, label="problem_eval")

        self.assertEqual(len(components), 1)
        self.assertEqual(str(components[0].label), "problem_eval:fold")
        self.assertEqual(int(total.threads), 1)
        self.assertEqual(str(total.backend), "serial")
        self.assertIsNotNone(total.metadata.get("execution_resource_grant"))


if __name__ == "__main__":
    unittest.main()
