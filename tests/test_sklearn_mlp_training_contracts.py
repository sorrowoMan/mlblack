from __future__ import annotations

import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np
from sklearn.exceptions import ConvergenceWarning

from config import TrainerAssemblySpec, build_trainer
from core.common.contracts import ProcessedDataset
from training import TrainTask, TrainingInit


def _make_processed_dataset(seed: int = 512) -> ProcessedDataset:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(64, 4))
    y = (0.7 * x[:, 0] - 0.4 * x[:, 1] + 0.2 * np.sin(x[:, 2]) + 0.1).reshape(-1, 1)
    return ProcessedDataset(
        X_train=x,
        y_train=y,
        feature_names=("x0", "x1", "x2", "x3"),
        target_names=("y",),
        metadata={"source": "sklearn_mlp_training_contract_test"},
    )


class TestSklearnMLPTrainingContracts(unittest.TestCase):
    def _build_trainer(self):
        return build_trainer(
            TrainerAssemblySpec(
                trainer_key="sklearn_mlp",
                trainer_params={
                    "hidden_layer_sizes": (24,),
                    "activation": "relu",
                    "solver": "adam",
                    "max_iter": 18,
                    "early_stopping": False,
                    "random_seed": 42,
                    "verbose": False,
                },
            )
        )

    def test_fit_task_emits_trainer_state_and_signature(self) -> None:
        trainer = self._build_trainer()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            result = trainer.fit_task(
                TrainTask.from_data(_make_processed_dataset(), task_id="sklearn_mlp::fresh"),
                TrainingInit(mode="fresh"),
            )

        self.assertIsNotNone(result.trainer_state)
        self.assertEqual(str(result.report["training_mode"]), "fresh")
        self.assertIn("training_signature", result.artifact.metadata)
        self.assertIsNotNone(result.trainer_state.feature_signature)

    def test_save_load_and_warm_start_from_state(self) -> None:
        data = _make_processed_dataset(seed=513)
        parent_trainer = self._build_trainer()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            parent_result = parent_trainer.fit_task(
                TrainTask.from_data(data, task_id="sklearn_mlp::parent"),
                TrainingInit(mode="fresh"),
            )

        self.assertIsNotNone(parent_result.trainer_state)

        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / "sklearn_mlp_state.pkl"
            parent_trainer.save_trainer_state(state_path, parent_result.trainer_state)
            loaded_state = parent_trainer.load_trainer_state(state_path)

            warm_trainer = self._build_trainer()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                warmed = warm_trainer.fit_task(
                    TrainTask.from_data(data, task_id="sklearn_mlp::warm"),
                    TrainingInit(mode="warm_start", parent_state=loaded_state),
                )
            self.assertEqual(str(warmed.report["training_mode"]), "warm_start")
            self.assertIsNotNone(warmed.trainer_state)
            self.assertTrue(bool(warmed.artifact.metadata.get("resume", {}).get("enabled")))
            pred = np.asarray(warmed.artifact.predict(data.X_train), dtype=float)
            self.assertEqual(pred.shape, (data.X_train.shape[0], 1))

    def test_runtime_mechanisms_apply_without_forcing_warm_start(self) -> None:
        data = _make_processed_dataset(seed=514)
        parent_trainer = self._build_trainer()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            parent_result = parent_trainer.fit_task(
                TrainTask.from_data(data, task_id="sklearn_mlp::signal_parent"),
                TrainingInit(mode="fresh"),
            )

        trainer = build_trainer(
            TrainerAssemblySpec(
                trainer_key="sklearn_mlp",
                trainer_params={
                    "hidden_layer_sizes": (24,),
                    "activation": "relu",
                    "solver": "adam",
                    "max_iter": 18,
                    "early_stopping": False,
                    "random_seed": 42,
                    "verbose": False,
                    "mechanisms": [
                        {"key": "state_signal_view.prediction_residual", "params": {}},
                        {"key": "sample_weighting.loss_adaptive", "params": {"alpha": 0.5, "power": 1.0}},
                        {
                            "key": "sampling.row_feature_subsample",
                            "params": {"row_fraction": 0.5, "feature_fraction": 0.5, "random_seed": 7},
                        },
                        {"key": "aggregation.ensemble_summary", "params": {}},
                    ],
                },
            )
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            result = trainer.fit_task(
                TrainTask.from_data(data, task_id="sklearn_mlp::mechanisms"),
                TrainingInit(mode="fresh", parent_artifact=parent_result.artifact),
            )

        runtime_warning_messages = [str(item.message) for item in caught if isinstance(item.message, RuntimeWarning)]
        self.assertTrue(any("sample_weight" in msg for msg in runtime_warning_messages))

        runtime = dict(result.artifact.metadata.get("runtime_mechanisms", {}))
        trace = list(runtime.get("trace", []))
        trace_status = {(str(row.get("mechanism_key")), str(row.get("status"))) for row in trace}
        self.assertIn(("state_signal_view.prediction_residual", "applied"), trace_status)
        self.assertIn(("sample_weighting.loss_adaptive", "applied"), trace_status)
        self.assertIn(("sampling.row_feature_subsample", "applied"), trace_status)
        self.assertIn(("aggregation.ensemble_summary", "applied"), trace_status)

        summary = dict(runtime.get("aggregation_summary", {}))
        self.assertEqual(int(summary.get("selected_rows", 0)), 32)
        self.assertEqual(int(summary.get("selected_features", 0)), 2)

        self.assertFalse(bool(result.artifact.metadata.get("resume", {}).get("enabled")))
        self.assertTrue(bool(result.artifact.metadata.get("fit_context", {}).get("sample_weight_ignored")))
        self.assertEqual(len(tuple(result.artifact.input_feature_indices or ())), 2)
        self.assertEqual(len(tuple(result.trainer_state.payload.get("input_feature_indices", ()) or ())), 2)
        self.assertIn("runtime_mechanisms", result.trainer_state.payload)

        pred = np.asarray(result.artifact.predict(data.X_train), dtype=float)
        self.assertEqual(pred.shape, (data.X_train.shape[0], 1))


if __name__ == "__main__":
    unittest.main()
