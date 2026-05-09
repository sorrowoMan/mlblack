from __future__ import annotations

import unittest

import numpy as np

from config import CapabilitySpec, FlowAssemblySpec, NumericizerSpec, TrainerAssemblySpec
from core.common.contracts import ProcessedDataset
from workflow import SemanticTrainFlowSpec, TrainDataBundle, run_semantic_train_flow


class TestMetricGuardCapability(unittest.TestCase):
    def _bundle(self) -> TrainDataBundle:
        rng = np.random.default_rng(7)
        X = rng.normal(size=(96, 5))
        y = (1.8 * X[:, 0] - 0.7 * X[:, 1] + 0.25 * X[:, 2] + 0.1).reshape(-1, 1)
        return TrainDataBundle(
            train=ProcessedDataset(
                X_train=X,
                y_train=y,
                feature_names=("x0", "x1", "x2", "x3", "x4"),
                target_names=("y",),
            )
        )

    def test_metric_guard_pass_and_report(self) -> None:
        spec = SemanticTrainFlowSpec(
            assembly=FlowAssemblySpec(
                trainer=TrainerAssemblySpec(
                    trainer_key="ridge",
                    trainer_params={"l2": 0.0},
                ),
                numericizer=NumericizerSpec(key="default", params={}),
                capabilities=(
                    CapabilitySpec(
                        key="metric_guard",
                        params={
                            "name": "rmse_guard",
                            "rules": [
                                {"split": "train", "metric": "rmse", "op": "le", "threshold": 0.2},
                            ],
                            "hard_fail": True,
                        },
                    ),
                ),
            ),
            eval_splits=("train",),
            save_artifact=False,
            save_report=False,
            capability_strict=True,
        )

        result = run_semantic_train_flow(self._bundle(), spec=spec)
        guard = dict(result.report.get("metric_guard", {}))
        self.assertTrue(bool(guard.get("ok")))
        self.assertEqual(len(list(guard.get("violations", []))), 0)

    def test_metric_guard_fails_in_strict_mode(self) -> None:
        spec = SemanticTrainFlowSpec(
            assembly=FlowAssemblySpec(
                trainer=TrainerAssemblySpec(
                    trainer_key="ridge",
                    trainer_params={"l2": 0.0},
                ),
                numericizer=NumericizerSpec(key="default", params={}),
                capabilities=(
                    CapabilitySpec(
                        key="metric_guard",
                        params={
                            "name": "rmse_guard_fail",
                            "rules": [
                                {"split": "train", "metric": "rmse", "op": "le", "threshold": -1.0},
                            ],
                            "hard_fail": True,
                        },
                    ),
                ),
            ),
            eval_splits=("train",),
            save_artifact=False,
            save_report=False,
            capability_strict=True,
        )

        with self.assertRaises(RuntimeError):
            run_semantic_train_flow(self._bundle(), spec=spec)


if __name__ == "__main__":
    unittest.main()
