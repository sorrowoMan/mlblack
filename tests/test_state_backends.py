from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from config import FlowAssemblySpec, NumericizerSpec, TrainerAssemblySpec
from core.common.contracts import ProcessedDataset
from core.state import PROCESSED_REF, create_context_store, create_snapshot_store
from workflow import ModelSpec, SemanticTrainFlowSpec, TrainDataBundle, run_semantic_train_flow


class TestStateBackends(unittest.TestCase):
    def test_factory_creates_sqlite_backends(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "state.sqlite3")
            ctx = create_context_store(backend="sqlite", db_path=db_path, namespace="ctx_ns")
            snap = create_snapshot_store(backend="sqlite", db_path=db_path, namespace="snap_ns")

            ctx.set("a", {"v": 1})
            self.assertTrue(ctx.has("a"))
            self.assertEqual(dict(ctx.get("a")), {"v": 1})

            sid = snap.write({"x": 1}, kind="demo")
            self.assertTrue(snap.has(sid))
            self.assertEqual(dict(snap.read(sid)), {"x": 1})

    def test_semantic_flow_accepts_backend_mapping_config(self) -> None:
        rng = np.random.default_rng(91)
        X = rng.normal(size=(72, 3))
        y = (1.7 * X[:, 0] - 0.3 * X[:, 1] + 0.2).reshape(-1, 1)
        bundle = TrainDataBundle(
            train=ProcessedDataset(
                X_train=X,
                y_train=y,
                feature_names=("x0", "x1", "x2"),
                target_names=("y",),
            )
        )

        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "runtime_state.sqlite3")
            spec = SemanticTrainFlowSpec(
                assembly=FlowAssemblySpec(
                    trainer=TrainerAssemblySpec(
                        trainer_key="ridge",
                        trainer_params={"l2": 0.0},
                    ),
                    numericizer=NumericizerSpec(key="default", params={}),
                ),
                model_spec=ModelSpec(model_id="sqlite_backend_model", feature_names=("x0", "x1"), target_names=("y",)),
                eval_splits=("train",),
                save_artifact=False,
                save_report=False,
                context_store={
                    "backend": "sqlite",
                    "db_path": db_path,
                    "namespace": "ctx",
                },
                snapshot_store={
                    "backend": "sqlite",
                    "db_path": db_path,
                    "namespace": "snap",
                },
            )

            result = run_semantic_train_flow(bundle, spec=spec)
            self.assertIn("train", result.metrics)
            context_refs = dict(result.report.get("state", {}).get("context_refs", {}))
            self.assertIn(PROCESSED_REF, context_refs)


if __name__ == "__main__":
    unittest.main()

