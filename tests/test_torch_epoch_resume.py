from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from config import TrainerAssemblySpec
from core.common.contracts import ProcessedDataset
from core.state import TRAINER_STATE_REF, create_context_store, create_snapshot_store
from core.trainers.torch_trainer import TorchMLPTrainerConfig, TorchMLPSurrogateTrainer
from training import TrainTask, TrainingInit
from workflow import TrainFlowSpec, run_train_flow


class TestTorchEpochResume(unittest.TestCase):
    def test_runtime_mechanisms_apply_without_forcing_resume(self) -> None:
        rng = np.random.default_rng(101)
        X = rng.normal(size=(96, 6))
        y = (1.5 * X[:, 0] - 0.4 * X[:, 1] + 0.35 * np.sin(X[:, 2])).reshape(-1, 1)
        ds = ProcessedDataset(
            X_train=X,
            y_train=y,
            feature_names=tuple(f"x{i}" for i in range(6)),
            target_names=("y",),
        )

        parent = TorchMLPSurrogateTrainer(
            config=TorchMLPTrainerConfig(
                artifact_id="torch_mechanism_parent",
                hidden_dims=(16, 8),
                epochs=4,
                batch_size=16,
                val_ratio=0.2,
                early_stop_patience=20,
                random_seed=42,
                device="cpu",
            )
        )
        parent_result = parent.fit_task(
            TrainTask.from_data(ds, task_id="mlp_torch::signal_parent"),
            TrainingInit(mode="fresh"),
        )

        trainer = TorchMLPSurrogateTrainer(
            config=TorchMLPTrainerConfig(
                artifact_id="torch_mechanism_child",
                hidden_dims=(16, 8),
                epochs=4,
                batch_size=16,
                val_ratio=0.2,
                early_stop_patience=20,
                random_seed=42,
                device="cpu",
                mechanisms=(
                    {"key": "state_signal_view.prediction_residual", "params": {}},
                    {"key": "sample_weighting.loss_adaptive", "params": {"alpha": 0.5, "power": 1.0}},
                    {
                        "key": "sampling.row_feature_subsample",
                        "params": {"row_fraction": 0.5, "feature_fraction": 0.5, "random_seed": 7},
                    },
                    {"key": "aggregation.ensemble_summary", "params": {}},
                ),
            )
        )
        result = trainer.fit_task(
            TrainTask.from_data(ds, task_id="mlp_torch::mechanisms"),
            TrainingInit(mode="fresh", parent_artifact=parent_result.artifact),
        )

        runtime = dict(result.artifact.metadata.get("runtime_mechanisms", {}))
        trace = list(runtime.get("trace", []))
        trace_status = {(str(row.get("mechanism_key")), str(row.get("status"))) for row in trace}
        self.assertIn(("state_signal_view.prediction_residual", "applied"), trace_status)
        self.assertIn(("sample_weighting.loss_adaptive", "applied"), trace_status)
        self.assertIn(("sampling.row_feature_subsample", "applied"), trace_status)
        self.assertIn(("aggregation.ensemble_summary", "applied"), trace_status)

        summary = dict(runtime.get("aggregation_summary", {}))
        self.assertEqual(int(summary.get("selected_rows", 0)), 48)
        self.assertEqual(int(summary.get("selected_features", 0)), 3)

        self.assertFalse(bool(result.artifact.metadata.get("resume", {}).get("enabled")))
        self.assertEqual(len(tuple(result.artifact.input_feature_indices or ())), 3)
        self.assertEqual(len(tuple(result.trainer_state.payload.get("input_feature_indices", ()) or ())), 3)
        self.assertIn("runtime_mechanisms", result.trainer_state.payload)

        pred = np.asarray(result.artifact.predict(X[:5]), dtype=float)
        self.assertEqual(tuple(pred.shape), (5, 1))

    def test_epoch_checkpoint_and_resume(self) -> None:
        rng = np.random.default_rng(123)
        X = rng.normal(size=(96, 6))
        y = (1.8 * X[:, 0] - 0.7 * X[:, 1] + 0.5 * np.sin(X[:, 2])).reshape(-1, 1)
        ds = ProcessedDataset(
            X_train=X,
            y_train=y,
            feature_names=tuple(f"x{i}" for i in range(6)),
            target_names=("y",),
        )

        with tempfile.TemporaryDirectory() as tmp:
            ck_dir = Path(tmp) / "torch_epoch_ckpt"

            cfg_1 = TorchMLPTrainerConfig(
                artifact_id="torch_resume_case",
                hidden_dims=(16, 8),
                epochs=4,
                batch_size=16,
                val_ratio=0.2,
                early_stop_patience=20,
                random_seed=42,
                checkpoint_dir=str(ck_dir),
                checkpoint_every_epochs=2,
            )
            trainer_1 = TorchMLPSurrogateTrainer(config=cfg_1)
            art_1 = trainer_1.fit(ds)
            self.assertTrue((ck_dir / "latest.pt").exists())

            cfg_2 = TorchMLPTrainerConfig(
                artifact_id="torch_resume_case",
                hidden_dims=(16, 8),
                epochs=6,
                batch_size=16,
                val_ratio=0.2,
                early_stop_patience=20,
                random_seed=42,
                checkpoint_dir=str(ck_dir),
                checkpoint_every_epochs=2,
                resume_training_from=str(ck_dir / "latest.pt"),
            )
            trainer_2 = TorchMLPSurrogateTrainer(config=cfg_2)
            art_2 = trainer_2.fit(ds)

            md = dict(art_2.metadata)
            resume = dict(md.get("resume", {}))
            self.assertTrue(bool(resume.get("enabled", False)))
            self.assertTrue(str(resume.get("from", "")).endswith("latest.pt"))
            self.assertGreaterEqual(int(resume.get("start_epoch", 0)), 2)

            model_meta = dict(md.get("model", {}))
            self.assertGreaterEqual(int(model_meta.get("last_completed_epoch", 0)), int(resume.get("start_epoch", 1)))

            pred_1 = np.asarray(art_1.predict(X), dtype=float)
            pred_2 = np.asarray(art_2.predict(X), dtype=float)
            self.assertEqual(pred_1.shape, pred_2.shape)

    def test_workflow_resume_closed_loop_with_trainer_state(self) -> None:
        rng = np.random.default_rng(222)
        X = rng.normal(size=(96, 6))
        y = (1.2 * X[:, 0] - 0.6 * X[:, 1] + 0.3 * np.sin(X[:, 2])).reshape(-1, 1)
        ds = ProcessedDataset(
            X_train=X,
            y_train=y,
            feature_names=tuple(f"x{i}" for i in range(6)),
            target_names=("y",),
        )

        spec_1 = TrainFlowSpec(
            assembly=TrainerAssemblySpec(
                trainer_key="mlp_torch",
                trainer_params={
                    "artifact_id": "torch_workflow_resume_case",
                    "hidden_dims": (16, 8),
                    "epochs": 4,
                    "batch_size": 16,
                    "val_ratio": 0.2,
                    "early_stop_patience": 20,
                    "random_seed": 42,
                    "device": "cpu",
                },
            ),
            eval_splits=("train",),
            save_artifact=False,
            save_report=False,
            run_name="torch_workflow_resume_parent",
        )
        res_1 = run_train_flow(ds, spec=spec_1)
        self.assertIsNotNone(res_1.trainer_state)

        spec_2 = TrainFlowSpec(
            assembly=TrainerAssemblySpec(
                trainer_key="mlp_torch",
                trainer_params={
                    "artifact_id": "torch_workflow_resume_case",
                    "hidden_dims": (16, 8),
                    "epochs": 6,
                    "batch_size": 16,
                    "val_ratio": 0.2,
                    "early_stop_patience": 20,
                    "random_seed": 42,
                    "device": "cpu",
                },
            ),
            training_init=TrainingInit(mode="resume", parent_state=res_1.trainer_state),
            eval_splits=("train",),
            save_artifact=False,
            save_report=False,
            run_name="torch_workflow_resume_child",
        )
        res_2 = run_train_flow(ds, spec=spec_2)

        self.assertIsNotNone(res_2.trainer_state)
        training_block = dict(res_2.report.get("training", {}))
        self.assertEqual(str(training_block.get("requested_init", {}).get("mode")), "resume")
        self.assertEqual(str(training_block.get("fit_report", {}).get("training_mode")), "resume")
        self.assertTrue(bool(training_block.get("trainer_state_available")))

        resume_meta = dict(res_2.artifact.metadata.get("resume", {}))
        self.assertTrue(bool(resume_meta.get("enabled", False)))
        self.assertGreaterEqual(int(resume_meta.get("start_epoch", 0)), 2)

    def test_workflow_resume_from_parent_state_path(self) -> None:
        rng = np.random.default_rng(333)
        X = rng.normal(size=(80, 5))
        y = (1.4 * X[:, 0] - 0.4 * X[:, 1] + 0.2 * np.cos(X[:, 2])).reshape(-1, 1)
        ds = ProcessedDataset(
            X_train=X,
            y_train=y,
            feature_names=tuple(f"x{i}" for i in range(5)),
            target_names=("y",),
        )

        with tempfile.TemporaryDirectory() as tmp:
            ck_dir = Path(tmp) / "workflow_ck"
            spec_1 = TrainFlowSpec(
                assembly=TrainerAssemblySpec(
                    trainer_key="mlp_torch",
                    trainer_params={
                        "artifact_id": "torch_workflow_path_case",
                        "hidden_dims": (12, 6),
                        "epochs": 4,
                        "batch_size": 16,
                        "val_ratio": 0.2,
                        "early_stop_patience": 20,
                        "random_seed": 7,
                        "device": "cpu",
                        "checkpoint_dir": str(ck_dir),
                        "checkpoint_every_epochs": 2,
                    },
                ),
                eval_splits=("train",),
                save_artifact=False,
                save_report=False,
                run_name="torch_workflow_path_parent",
            )
            res_1 = run_train_flow(ds, spec=spec_1)
            self.assertTrue((ck_dir / "latest.pt").exists())
            self.assertIsNotNone(res_1.trainer_state)

            spec_2 = TrainFlowSpec(
                assembly=TrainerAssemblySpec(
                    trainer_key="mlp_torch",
                    trainer_params={
                        "artifact_id": "torch_workflow_path_case",
                        "hidden_dims": (12, 6),
                        "epochs": 6,
                        "batch_size": 16,
                        "val_ratio": 0.2,
                        "early_stop_patience": 20,
                        "random_seed": 7,
                        "device": "cpu",
                    },
                ),
                training_init={
                    "mode": "resume",
                    "parent_state": str(ck_dir / "latest.pt"),
                    "metadata": {"case": "path_resume"},
                },
                eval_splits=("train",),
                save_artifact=False,
                save_report=False,
                run_name="torch_workflow_path_child",
            )
            res_2 = run_train_flow(ds, spec=spec_2)

            training_block = dict(res_2.report.get("training", {}))
            self.assertEqual(str(training_block.get("requested_init", {}).get("mode")), "resume")
            self.assertEqual(
                str(training_block.get("requested_init", {}).get("metadata", {}).get("parent_state_path")),
                str(ck_dir / "latest.pt"),
            )
            self.assertEqual(str(training_block.get("fit_report", {}).get("training_mode")), "resume")
            self.assertTrue(bool(res_2.trainer_state is not None))

    def test_workflow_resume_from_snapshot_ref(self) -> None:
        rng = np.random.default_rng(444)
        X = rng.normal(size=(72, 5))
        y = (0.9 * X[:, 0] - 0.2 * X[:, 1] + 0.4 * np.sin(X[:, 2])).reshape(-1, 1)
        ds = ProcessedDataset(
            X_train=X,
            y_train=y,
            feature_names=tuple(f"x{i}" for i in range(5)),
            target_names=("y",),
        )

        context_store = create_context_store(backend="memory")
        snapshot_store = create_snapshot_store(backend="memory")

        spec_1 = TrainFlowSpec(
            assembly=TrainerAssemblySpec(
                trainer_key="mlp_torch",
                trainer_params={
                    "artifact_id": "torch_workflow_snapshot_case",
                    "hidden_dims": (12, 6),
                    "epochs": 4,
                    "batch_size": 16,
                    "val_ratio": 0.2,
                    "early_stop_patience": 20,
                    "random_seed": 21,
                    "device": "cpu",
                },
            ),
            eval_splits=("train",),
            save_artifact=False,
            save_report=False,
            run_name="torch_workflow_snapshot_parent",
            context_store=context_store,
            snapshot_store=snapshot_store,
        )
        _ = run_train_flow(ds, spec=spec_1)
        trainer_state_ref = context_store.get(TRAINER_STATE_REF)
        self.assertTrue(bool(trainer_state_ref))

        spec_2 = TrainFlowSpec(
            assembly=TrainerAssemblySpec(
                trainer_key="mlp_torch",
                trainer_params={
                    "artifact_id": "torch_workflow_snapshot_case",
                    "hidden_dims": (12, 6),
                    "epochs": 6,
                    "batch_size": 16,
                    "val_ratio": 0.2,
                    "early_stop_patience": 20,
                    "random_seed": 21,
                    "device": "cpu",
                },
            ),
            training_init={
                "mode": "resume",
                "parent_state": {"snapshot_ref": str(trainer_state_ref)},
            },
            eval_splits=("train",),
            save_artifact=False,
            save_report=False,
            run_name="torch_workflow_snapshot_child",
            context_store=context_store,
            snapshot_store=snapshot_store,
        )
        res_2 = run_train_flow(ds, spec=spec_2)

        training_block = dict(res_2.report.get("training", {}))
        requested_init = dict(training_block.get("requested_init", {}))
        self.assertEqual(str(requested_init.get("parent_state_source")), "snapshot_ref")
        self.assertEqual(str(requested_init.get("parent_state_snapshot_ref")), str(trainer_state_ref))
        self.assertTrue(bool(res_2.trainer_state is not None))

    def test_workflow_resume_from_context_key_locator(self) -> None:
        rng = np.random.default_rng(555)
        X = rng.normal(size=(72, 5))
        y = (1.1 * X[:, 0] - 0.3 * X[:, 1] + 0.25 * np.cos(X[:, 2])).reshape(-1, 1)
        ds = ProcessedDataset(
            X_train=X,
            y_train=y,
            feature_names=tuple(f"x{i}" for i in range(5)),
            target_names=("y",),
        )

        context_store = create_context_store(backend="memory")
        snapshot_store = create_snapshot_store(backend="memory")

        spec_1 = TrainFlowSpec(
            assembly=TrainerAssemblySpec(
                trainer_key="mlp_torch",
                trainer_params={
                    "artifact_id": "torch_workflow_context_case",
                    "hidden_dims": (10, 6),
                    "epochs": 4,
                    "batch_size": 16,
                    "val_ratio": 0.2,
                    "early_stop_patience": 20,
                    "random_seed": 11,
                    "device": "cpu",
                },
            ),
            eval_splits=("train",),
            save_artifact=False,
            save_report=False,
            run_name="torch_workflow_context_parent",
            context_store=context_store,
            snapshot_store=snapshot_store,
        )
        _ = run_train_flow(ds, spec=spec_1)

        spec_2 = TrainFlowSpec(
            assembly=TrainerAssemblySpec(
                trainer_key="mlp_torch",
                trainer_params={
                    "artifact_id": "torch_workflow_context_case",
                    "hidden_dims": (10, 6),
                    "epochs": 6,
                    "batch_size": 16,
                    "val_ratio": 0.2,
                    "early_stop_patience": 20,
                    "random_seed": 11,
                    "device": "cpu",
                },
            ),
            training_init=TrainingInit(mode="resume", parent_state=f"context://{TRAINER_STATE_REF}"),
            eval_splits=("train",),
            save_artifact=False,
            save_report=False,
            run_name="torch_workflow_context_child",
            context_store=context_store,
            snapshot_store=snapshot_store,
        )
        res_2 = run_train_flow(ds, spec=spec_2)

        training_block = dict(res_2.report.get("training", {}))
        requested_init = dict(training_block.get("requested_init", {}))
        self.assertEqual(str(requested_init.get("parent_state_source")), "context_key")
        self.assertEqual(str(requested_init.get("parent_state_context_key")), TRAINER_STATE_REF)
        self.assertTrue(bool(res_2.trainer_state is not None))


if __name__ == "__main__":
    unittest.main()
