from __future__ import annotations

import unittest

import numpy as np

from config import FlowAssemblySpec, NumericizerSpec, TrainerAssemblySpec
from core.common.contracts import ProcessedDataset
from core.state import MODEL_PROCESSED_REF, MODEL_SPEC_REF
from workflow import (
    ContextStore,
    InMemorySnapshotStore,
    ModelSpec,
    SemanticTrainFlowSpec,
    TrainDataBundle,
    run_semantic_train_flow,
)


class TestModelSpecRuntime(unittest.TestCase):
    def test_model_spec_selects_feature_and_target_subspace(self) -> None:
        rng = np.random.default_rng(123)
        X = rng.normal(size=(80, 4))
        y0 = 0.5 * X[:, 0] - 0.2 * X[:, 2] + 0.1
        y1 = -1.2 * X[:, 1] + 0.8 * X[:, 3] + 0.3
        Y = np.column_stack([y0, y1])

        bundle = TrainDataBundle(
            train=ProcessedDataset(
                X_train=X,
                y_train=Y,
                feature_names=("x0", "x1", "x2", "x3"),
                target_names=("y0", "y1"),
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
            model_spec=ModelSpec(
                model_id="m_x1x3_to_y1",
                feature_names=("x1", "x3"),
                target_names=("y1",),
                strict=True,
            ),
            eval_splits=("train",),
            save_artifact=False,
            save_report=False,
            context_store=context_store,
            snapshot_store=snapshot_store,
        )

        result = run_semantic_train_flow(bundle, spec=spec)
        self.assertEqual(result.processed.X_train.shape[1], 2)
        self.assertEqual(result.processed.y_train.shape[1], 1)
        self.assertEqual(tuple(result.processed.feature_names or ()), ("x1", "x3"))
        self.assertEqual(tuple(result.processed.target_names or ()), ("y1",))

        model_block = dict(result.report.get("model_spec", {}))
        self.assertEqual(str(model_block.get("model_id")), "m_x1x3_to_y1")

        spec_ref = context_store.get(MODEL_SPEC_REF)
        proc_ref = context_store.get(MODEL_PROCESSED_REF)
        self.assertIsInstance(spec_ref, str)
        self.assertIsInstance(proc_ref, str)
        self.assertTrue(snapshot_store.has(spec_ref))
        self.assertTrue(snapshot_store.has(proc_ref))


if __name__ == "__main__":
    unittest.main()

