from __future__ import annotations

import unittest

import numpy as np

from config import FlowAssemblySpec, NumericizerSpec, TrainerAssemblySpec
from core.common.contracts import ProcessedDataset
from core.state import (
    ARTIFACT_REF,
    BUNDLE_REF,
    FLOW_SPEC_REF,
    METRICS_REF,
    PROCESSED_REF,
    REPORT_REF,
    RESULT_REF,
    RUN_STAGE,
)
from workflow import ContextStore, InMemorySnapshotStore, SemanticTrainFlowSpec, TrainDataBundle, run_semantic_train_flow


class TestFlowStateRuntime(unittest.TestCase):
    def test_context_snapshot_refs_are_written(self) -> None:
        rng = np.random.default_rng(101)
        X = rng.normal(size=(64, 4))
        y = (1.2 * X[:, 0] - 0.4 * X[:, 1] + 0.6).reshape(-1, 1)

        bundle = TrainDataBundle(
            train=ProcessedDataset(
                X_train=X,
                y_train=y,
                feature_names=("x0", "x1", "x2", "x3"),
                target_names=("y",),
            )
        )

        context_store = ContextStore()
        snapshot_store = InMemorySnapshotStore()
        spec = SemanticTrainFlowSpec(
            assembly=FlowAssemblySpec(
                trainer=TrainerAssemblySpec(
                    trainer_key="ridge",
                    trainer_params={"l2": 0.0},
                ),
                numericizer=NumericizerSpec(key="default", params={}),
            ),
            eval_splits=("train",),
            save_artifact=False,
            save_report=False,
            context_store=context_store,
            snapshot_store=snapshot_store,
            run_name="state_runtime_smoke",
        )

        result = run_semantic_train_flow(bundle, spec=spec)
        self.assertIn("train", result.metrics)

        for key in (
            FLOW_SPEC_REF,
            BUNDLE_REF,
            PROCESSED_REF,
            ARTIFACT_REF,
            METRICS_REF,
            REPORT_REF,
            RESULT_REF,
        ):
            ref = context_store.get(key)
            self.assertIsInstance(ref, str)
            self.assertTrue(snapshot_store.has(ref))

        self.assertEqual(str(context_store.get(RUN_STAGE)), "finished")
        state = dict(result.report.get("state", {}))
        self.assertGreaterEqual(int(state.get("snapshot_count", 0)), 7)
        context_refs = dict(state.get("context_refs", {}))
        self.assertEqual(str(context_refs.get(RUN_STAGE)), "finished")


if __name__ == "__main__":
    unittest.main()

