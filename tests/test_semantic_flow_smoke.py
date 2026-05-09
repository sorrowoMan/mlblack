from __future__ import annotations

import unittest

import numpy as np

from config import CapabilitySpec, FlowAssemblySpec, NumericizerSpec, TrainerAssemblySpec
from core.common.contracts import ProcessedDataset
from workflow import SemanticTrainFlowSpec, TrainDataBundle, run_semantic_train_flow


class TestSemanticFlowSmoke(unittest.TestCase):
    def test_ridge_semantic_flow_runs(self) -> None:
        rng = np.random.default_rng(0)
        X = rng.normal(size=(80, 4))
        y = (1.5 * X[:, 0] - 0.8 * X[:, 1] + 0.2).reshape(-1, 1)

        bundle = TrainDataBundle(
            train=ProcessedDataset(
                X_train=X,
                y_train=y,
                feature_names=("x0", "x1", "x2", "x3"),
                target_names=("y",),
            )
        )

        spec = SemanticTrainFlowSpec(
            assembly=FlowAssemblySpec(
                trainer=TrainerAssemblySpec(
                    trainer_key="ridge",
                    trainer_params={"l2": 0.0},
                ),
                numericizer=NumericizerSpec(key="default", params={}),
                capabilities=(
                    CapabilitySpec(
                        key="noop",
                        params={
                            "name": "semantic_declared_capability",
                            "priority": 1,
                        },
                    ),
                ),
            ),
            eval_splits=("train",),
            save_artifact=False,
            save_report=False,
        )

        result = run_semantic_train_flow(bundle, spec=spec)
        self.assertIn("train", result.metrics)
        self.assertLess(result.metrics["train"]["rmse"], 0.2)
        cap_report = dict(result.report.get("capabilities", {}))
        self.assertEqual(int(cap_report.get("count", 0)), 1)
        cap_items = list(cap_report.get("items", []))
        self.assertEqual(cap_items[0].get("name"), "semantic_declared_capability")


if __name__ == "__main__":
    unittest.main()
