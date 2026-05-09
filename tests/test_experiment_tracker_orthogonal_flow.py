from __future__ import annotations

import gc
import shutil
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

from config import CapabilitySpec, FlowAssemblySpec, NumericizerSpec, TrainerAssemblySpec
from core.common.contracts import ProcessedDataset
from core.flow_experiment_tracker import list_experiment_artifact_catalog, list_experiment_run_catalog
from workflow import SemanticTrainFlowSpec, TrainDataBundle, run_semantic_train_flow


def _bundle(seed: int = 20260504) -> TrainDataBundle:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(140, 4))
    y = (
        2.1 * (x[:, 0] * x[:, 1])
        + 0.75 * np.sin(x[:, 2])
        - 0.4 * x[:, 3]
        + rng.normal(scale=0.03, size=x.shape[0])
    ).reshape(-1, 1)
    return TrainDataBundle(
        train=ProcessedDataset(
            X_train=np.asarray(x[:100], dtype=float),
            y_train=np.asarray(y[:100], dtype=float),
            feature_names=("x0", "x1", "x2", "x3"),
            target_names=("y",),
        ),
        test=ProcessedDataset(
            X_train=np.asarray(x[100:], dtype=float),
            y_train=np.asarray(y[100:], dtype=float),
            feature_names=("x0", "x1", "x2", "x3"),
            target_names=("y",),
        ),
    )


class TestExperimentTrackerOrthogonalFlow(unittest.TestCase):
    def test_tracker_materializes_real_orthogonal_symbolic_run(self) -> None:
        tmp_dir = Path(tempfile.mkdtemp(prefix="mlblack_exp_tracker_orth_"))
        try:
            db_path = tmp_dir / "experiments.sqlite3"
            spec = SemanticTrainFlowSpec(
                assembly=FlowAssemblySpec(
                    trainer=TrainerAssemblySpec(
                        trainer_key="symbolic",
                        trainer_params={
                            "parameter_backend": "ridge",
                            "task": "point",
                            "structure_engine": {
                                "structure_mode": "orthogonal_basis_search",
                                "search_driver": "orthogonal_basis",
                                "dynamic_pool_enabled": True,
                                "metadata": {"supports_piecewise_basis": True},
                            },
                            "candidate_limit": 36,
                            "group_count": 5,
                            "seed_candidate_count": 8,
                            "min_basis_count": 2,
                            "max_basis_count": 4,
                            "rolling_folds": 2,
                            "selection_mode": "rmse_first",
                            "gate_feature_names": ("x2",),
                            "search_graph_cache_enabled": False,
                        },
                    ),
                    numericizer=NumericizerSpec(key="default", params={}),
                    capabilities=(
                        CapabilitySpec(
                            key="experiment_tracker",
                            params={
                                "db_path": str(db_path),
                                "namespace": "ut_orthogonal",
                                "io_mode": "batched",
                                "commit_interval": 0,
                            },
                        ),
                    ),
                ),
                eval_splits=("train", "test"),
                save_artifact=False,
                save_report=False,
                capability_strict=True,
                run_name="orthogonal_tracker_smoke",
            )

            result = run_semantic_train_flow(_bundle(), spec=spec)
            tracker = dict(result.report.get("experiment_tracker", {}))
            run_id = str(tracker.get("run_id", ""))
            self.assertTrue(run_id)

            run_rows = list_experiment_run_catalog(str(db_path), trainer_name="symbolic_orthogonal")
            self.assertEqual(len(run_rows), 1)
            run_row = dict(run_rows[0])
            self.assertEqual(str(run_row.get("orthogonality_status")), "reported")
            self.assertIn(str(run_row.get("piecewise_gate_status")), {"configured", "enabled"})
            self.assertEqual(str(run_row.get("basis_scope")), "global")
            self.assertEqual(str(run_row.get("family_ref")), "family:symbolic")
            self.assertIsNotNone(run_row.get("outer_objective_score"))
            self.assertGreater(float(run_row.get("outer_objective_score") or 0.0), 0.0)
            self.assertEqual(str(dict(run_row.get("orthogonal_search_objective_json", {})).get("status")), "reported")

            artifact_rows = list_experiment_artifact_catalog(str(db_path), trainer_name="symbolic_orthogonal")
            self.assertEqual(len(artifact_rows), 1)
            artifact_row = dict(artifact_rows[0])
            self.assertEqual(str(artifact_row.get("orthogonality_status")), "reported")
            self.assertGreater(float(artifact_row.get("outer_objective_score") or 0.0), 0.0)
            self.assertEqual(str(dict(artifact_row.get("orthogonal_search_objective_json", {})).get("status")), "reported")
        finally:
            gc.collect()
            time.sleep(0.1)
            shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
