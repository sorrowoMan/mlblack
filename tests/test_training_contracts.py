from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from config import FlowAssemblySpec, NumericizerSpec, TrainerAssemblySpec
from core.common.contracts import ProcessedDataset
from core.trainers.trainer import RidgeSurrogateTrainer, RidgeTrainerConfig
from training import (
    FitResult,
    TrainTask,
    TrainerCapabilities,
    TrainerState,
    TrainingCompatibilityError,
    TrainingInit,
    validate_training_setup,
)
from workflow import SemanticTrainFlowSpec, TrainDataBundle, TrainFlowSpec, run_semantic_train_flow, run_train_flow


def _make_processed_dataset(seed: int = 11) -> ProcessedDataset:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(40, 4))
    y = (1.4 * X[:, 0] - 0.5 * X[:, 1] + 0.2).reshape(-1, 1)
    return ProcessedDataset(
        X_train=X,
        y_train=y,
        feature_names=("x0", "x1", "x2", "x3"),
        target_names=("y",),
        metadata={"source": "unit_test"},
    )


class IncrementalAwareRidgeTrainer(RidgeSurrogateTrainer):
    name = "ridge_incremental"

    def capabilities(self) -> TrainerCapabilities:
        return TrainerCapabilities(
            supports_fresh=True,
            supports_resume=True,
            supports_warm_start=True,
            supports_incremental=True,
            supports_recalibration=True,
            metadata={"trainer_name": self.name},
        )


class TestTrainingContracts(unittest.TestCase):
    def test_train_task_from_data_builds_structured_task(self) -> None:
        data = _make_processed_dataset()
        weights = np.linspace(1.0, 2.0, num=data.X_train.shape[0], dtype=float)

        task = TrainTask.from_data(
            data,
            schema={"input": "processed"},
            objective={"kind": "regression"},
            sample_weight=weights,
            metadata={"case": "train_task"},
            task_id="unit::task",
        )

        self.assertIs(task.data, data)
        self.assertEqual(task.task_id, "unit::task")
        self.assertEqual(task.schema, {"input": "processed"})
        self.assertEqual(task.objective, {"kind": "regression"})
        self.assertEqual(task.metadata["case"], "train_task")
        self.assertEqual(task.sample_weight.shape[0], data.X_train.shape[0])

    def test_validate_training_setup_enforces_parent_requirements(self) -> None:
        trainer = RidgeSurrogateTrainer(RidgeTrainerConfig(l2=0.0))

        resume_verdict = validate_training_setup(
            trainer.capabilities(),
            TrainingInit(mode="resume"),
        )
        self.assertFalse(resume_verdict.supported)
        self.assertIn("resume mode requires parent_state", resume_verdict.reasons)

        warm_start_verdict = validate_training_setup(
            trainer.capabilities(),
            TrainingInit(mode="warm_start"),
        )
        self.assertFalse(warm_start_verdict.supported)
        self.assertIn("warm_start mode requires parent_artifact or parent_state", warm_start_verdict.reasons)

    def test_ridge_fit_task_emits_trainer_state(self) -> None:
        trainer = RidgeSurrogateTrainer(RidgeTrainerConfig(l2=0.0))
        data = _make_processed_dataset()
        task = TrainTask.from_data(data, metadata={"case": "bridge"}, task_id="bridge::fit")

        fit_result = trainer.fit_task(task, TrainingInit(mode="fresh"))

        self.assertIsInstance(fit_result, FitResult)
        self.assertEqual(fit_result.lineage.mode, "fresh")
        self.assertEqual(fit_result.lineage.trainer_name, "ridge")
        self.assertEqual(fit_result.lineage.metadata["task_id"], "bridge::fit")
        self.assertIn("task_signature", fit_result.report)
        self.assertIn("training_signature", fit_result.artifact.metadata)
        self.assertIsNotNone(fit_result.trainer_state)
        self.assertEqual(str(fit_result.trainer_state.trainer_name), "ridge")
        pred = np.asarray(fit_result.artifact.predict(data.X_train), dtype=float)
        self.assertEqual(pred.shape[0], data.X_train.shape[0])

    def test_ridge_trainer_state_roundtrip_and_warm_start(self) -> None:
        trainer = RidgeSurrogateTrainer(RidgeTrainerConfig(l2=0.0))
        data = _make_processed_dataset(seed=23)
        parent_result = trainer.fit_task(
            TrainTask.from_data(data, task_id="ridge::parent"),
            TrainingInit(mode="fresh"),
        )

        state = parent_result.trainer_state
        self.assertIsNotNone(state)

        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / "ridge_state.pkl"
            trainer.save_trainer_state(state_path, state)
            loaded = trainer.load_trainer_state(state_path)
            self.assertEqual(str(loaded.trainer_name), "ridge")

            resumed = trainer.fit_task(
                TrainTask.from_data(data, task_id="ridge::resume"),
                TrainingInit(mode="resume", parent_state=loaded),
            )
            self.assertEqual(str(resumed.report["training_mode"]), "resume")
            self.assertIsNotNone(resumed.trainer_state)

        warm = trainer.fit_task(
            TrainTask.from_data(data, task_id="ridge::warm"),
            TrainingInit(mode="warm_start", parent_artifact=parent_result.artifact),
        )
        self.assertEqual(str(warm.report["training_mode"]), "warm_start")
        self.assertTrue(bool(warm.artifact.metadata.get("resume", {}).get("enabled")))

    def test_incremental_mode_rejects_feature_space_pollution(self) -> None:
        trainer = IncrementalAwareRidgeTrainer(RidgeTrainerConfig(l2=0.0))
        base_data = _make_processed_dataset(seed=17)
        parent_result = trainer.fit_task(
            TrainTask.from_data(base_data, task_id="parent::fresh"),
            TrainingInit(mode="fresh"),
        )

        shifted_data = ProcessedDataset(
            X_train=base_data.X_train[:, :3],
            y_train=base_data.y_train,
            feature_names=("x0", "x1", "x2"),
            target_names=("y",),
            metadata={"source": "unit_test"},
        )

        with self.assertRaises(TrainingCompatibilityError) as ctx:
            trainer.fit_task(
                TrainTask.from_data(shifted_data, task_id="child::incremental"),
                TrainingInit(mode="incremental", parent_artifact=parent_result.artifact),
            )

        self.assertIn("feature_signature", str(ctx.exception))

    def test_resume_mode_rejects_foreign_trainer_state(self) -> None:
        trainer = IncrementalAwareRidgeTrainer(RidgeTrainerConfig(l2=0.0))
        data = _make_processed_dataset(seed=19)
        parent_result = trainer.fit_task(
            TrainTask.from_data(data, task_id="parent::fresh"),
            TrainingInit(mode="fresh"),
        )
        parent_signature = dict(parent_result.report["task_signature"])
        foreign_state = TrainerState(
            trainer_name="foreign_trainer",
            payload={},
            schema_signature=parent_signature.get("schema_signature"),
            feature_signature=parent_signature.get("feature_signature"),
            target_signature=parent_signature.get("target_signature"),
            objective_signature=parent_signature.get("objective_signature"),
            pipeline_signature=parent_signature.get("pipeline_signature"),
            numericizer_signature=parent_signature.get("numericizer_signature"),
            regime_signature=parent_signature.get("regime_signature"),
        )

        with self.assertRaises(TrainingCompatibilityError) as ctx:
            trainer.fit_task(
                TrainTask.from_data(data, task_id="child::resume"),
                TrainingInit(mode="resume", parent_state=foreign_state),
            )

        self.assertIn("trainer_family", str(ctx.exception))

    def test_run_train_flow_report_contains_training_contract_block(self) -> None:
        data = _make_processed_dataset()
        spec = TrainFlowSpec(
            assembly=TrainerAssemblySpec(
                trainer_key="ridge",
                trainer_params={"l2": 0.0},
            ),
            eval_splits=("train",),
            save_artifact=False,
            save_report=False,
            run_name="training_contract_test",
        )

        result = run_train_flow(data, spec=spec)

        training_block = dict(result.report.get("training", {}))
        self.assertEqual(training_block.get("trainer_state_available"), True)
        self.assertEqual(training_block.get("lineage", {}).get("mode"), "fresh")
        self.assertEqual(training_block.get("fit_report", {}).get("training_mode"), "fresh")
        self.assertIn("task_signature", dict(training_block.get("fit_report", {})))
        self.assertEqual(training_block.get("requested_init", {}).get("mode"), "fresh")
        self.assertTrue(training_block.get("trainer_capabilities", {}).get("supports_fresh"))

    def test_run_train_flow_accepts_explicit_training_init(self) -> None:
        data = _make_processed_dataset(seed=29)
        trainer = IncrementalAwareRidgeTrainer(RidgeTrainerConfig(l2=0.0))
        parent_result = trainer.fit_task(
            TrainTask.from_data(data, task_id="parent::fresh"),
            TrainingInit(mode="fresh"),
        )
        spec = TrainFlowSpec(
            assembly=TrainerAssemblySpec(
                trainer_key="ridge",
                trainer_params={"l2": 0.0},
            ),
            training_init=TrainingInit(
                mode="warm_start",
                parent_artifact=parent_result.artifact,
                metadata={"case": "workflow_explicit"},
            ),
            eval_splits=("train",),
            save_artifact=False,
            save_report=False,
            run_name="workflow_training_init_explicit",
        )

        with patch("core.orchestration.workflow.build_trainer", return_value=IncrementalAwareRidgeTrainer(RidgeTrainerConfig(l2=0.0))):
            result = run_train_flow(data, spec=spec)

        training_block = dict(result.report.get("training", {}))
        self.assertEqual(training_block.get("requested_init", {}).get("mode"), "warm_start")
        self.assertEqual(training_block.get("fit_report", {}).get("training_mode"), "warm_start")
        self.assertEqual(training_block.get("lineage", {}).get("mode"), "warm_start")
        self.assertEqual(training_block.get("requested_init", {}).get("parent_artifact_id"), "ridge_surrogate_v1")

    def test_run_semantic_train_flow_propagates_training_init_mapping(self) -> None:
        data = _make_processed_dataset(seed=31)
        trainer = IncrementalAwareRidgeTrainer(RidgeTrainerConfig(l2=0.0))
        parent_result = trainer.fit_task(
            TrainTask.from_data(data, task_id="parent::fresh"),
            TrainingInit(mode="fresh"),
        )
        bundle = TrainDataBundle(train=data)
        spec = SemanticTrainFlowSpec(
            assembly=FlowAssemblySpec(
                trainer=TrainerAssemblySpec(
                    trainer_key="ridge",
                    trainer_params={"l2": 0.0},
                ),
                numericizer=NumericizerSpec(key="default", params={}),
            ),
            training_init={
                "mode": "warm_start",
                "parent_artifact": parent_result.artifact,
                "metadata": {"case": "semantic_mapping"},
            },
            eval_splits=("train",),
            save_artifact=False,
            save_report=False,
            run_name="semantic_training_init_mapping",
        )

        with patch("core.orchestration.workflow.build_trainer", return_value=IncrementalAwareRidgeTrainer(RidgeTrainerConfig(l2=0.0))):
            result = run_semantic_train_flow(bundle, spec=spec)

        training_block = dict(result.report.get("training", {}))
        self.assertEqual(training_block.get("requested_init", {}).get("mode"), "warm_start")
        self.assertEqual(training_block.get("fit_report", {}).get("training_mode"), "warm_start")
        self.assertEqual(training_block.get("requested_init", {}).get("metadata", {}).get("case"), "semantic_mapping")


if __name__ == "__main__":
    unittest.main()
