from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from config import TrainerAssemblySpec, build_trainer
from core.common.contracts import ProcessedDataset
from training import TrainTask, TrainingInit


def _make_processed_dataset(seed: int = 404) -> ProcessedDataset:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(72, 5))
    y = (1.6 * x[:, 0] - 0.8 * x[:, 1] + 0.3 * np.sin(x[:, 2]) + 0.1 * x[:, 3]).reshape(-1, 1)
    return ProcessedDataset(
        X_train=x,
        y_train=y,
        feature_names=("x0", "x1", "x2", "x3", "x4"),
        target_names=("y",),
        metadata={"source": "xgboost_training_contract_test"},
    )


class TestXGBoostTrainingContracts(unittest.TestCase):
    def _build_trainer(self, trainer_params: dict[str, object] | None = None):
        params = {
            "n_estimators": 4,
            "max_depth": 3,
            "learning_rate": 0.1,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "tree_method": "hist",
            "n_jobs": 1,
            "random_seed": 42,
            "verbosity": 0,
        }
        if trainer_params:
            params.update(dict(trainer_params))
        return build_trainer(
            TrainerAssemblySpec(
                trainer_key="xgboost",
                trainer_params=params,
            )
        )

    def test_fit_task_emits_trainer_state_and_signature(self) -> None:
        trainer = self._build_trainer({"n_estimators": 4})
        result = trainer.fit_task(
            TrainTask.from_data(_make_processed_dataset(), task_id="xgboost::fresh"),
            TrainingInit(mode="fresh"),
        )

        self.assertIsNotNone(result.trainer_state)
        self.assertEqual(str(result.report["training_mode"]), "fresh")
        self.assertIn("training_signature", result.artifact.metadata)
        self.assertIn("tree_boosting_family", result.artifact.metadata)
        self.assertEqual(result.artifact.metadata["tree_boosting_family"]["backend"]["backend"], "xgboost")
        self.assertIsNotNone(result.trainer_state.payload.get("tree_boosting_family_signature"))
        self.assertIsNotNone(result.trainer_state.feature_signature)

    def test_grouped_family_spec_is_consumed(self) -> None:
        trainer = self._build_trainer(
            {
                "family_spec": {
                    "boosting": {
                        "n_estimators": 6,
                        "learning_rate": 0.07,
                        "objective": "reg:squarederror",
                        "tree_method": "hist",
                    },
                    "sampling": {
                        "subsample": 0.85,
                        "colsample_bytree": 0.75,
                    },
                    "regularization": {
                        "max_depth": 4,
                        "min_child_weight": 1.5,
                    },
                    "execution": {
                        "n_jobs": 1,
                        "random_seed": 7,
                    },
                    "task_head": {
                        "task": "point",
                        "objective_family": "regression",
                        "uncertainty_mode": "residual_std",
                    },
                }
            }
        )
        result = trainer.fit_task(
            TrainTask.from_data(_make_processed_dataset(seed=408), task_id="xgboost::grouped"),
            TrainingInit(mode="fresh"),
        )

        self.assertEqual(int(result.artifact.metadata["tree_boosting_family"]["boosting"]["n_estimators"]), 6)
        self.assertEqual(float(result.artifact.metadata["tree_boosting_family"]["sampling"]["subsample"]), 0.85)
        self.assertEqual(int(result.artifact.model.get_booster().num_boosted_rounds()), 6)

    def test_save_load_and_resume_continue_boosting(self) -> None:
        data = _make_processed_dataset(seed=405)
        parent_trainer = self._build_trainer({"n_estimators": 4})
        parent_result = parent_trainer.fit_task(
            TrainTask.from_data(data, task_id="xgboost::parent"),
            TrainingInit(mode="fresh"),
        )

        parent_rounds = int(parent_result.artifact.model.get_booster().num_boosted_rounds())
        self.assertEqual(parent_rounds, 4)

        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / "xgboost_state.pkl"
            parent_trainer.save_trainer_state(state_path, parent_result.trainer_state)
            loaded_state = parent_trainer.load_trainer_state(state_path)

            resume_trainer = self._build_trainer({"n_estimators": 3})
            resumed = resume_trainer.fit_task(
                TrainTask.from_data(data, task_id="xgboost::resume"),
                TrainingInit(mode="resume", parent_state=loaded_state),
            )
            resumed_rounds = int(resumed.artifact.model.get_booster().num_boosted_rounds())
            self.assertEqual(str(resumed.report["training_mode"]), "resume")
            self.assertEqual(resumed_rounds, parent_rounds + 3)

    def test_resume_from_config_path_uses_saved_state(self) -> None:
        data = _make_processed_dataset(seed=406)
        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / "xgboost_state.pkl"
            parent_trainer = self._build_trainer({"n_estimators": 2})
            parent_result = parent_trainer.fit_task(
                TrainTask.from_data(data, task_id="xgboost::config_parent"),
                TrainingInit(mode="fresh"),
            )
            parent_trainer.save_trainer_state(state_path, parent_result.trainer_state)

            resume_trainer = self._build_trainer({"n_estimators": 2, "resume_training_from": str(state_path)})
            resumed = resume_trainer.fit_task(
                TrainTask.from_data(data, task_id="xgboost::config_resume"),
                TrainingInit(mode="fresh"),
            )
            self.assertEqual(str(resumed.report["training_mode"]), "fresh")
            self.assertTrue(bool(resumed.artifact.metadata.get("resume", {}).get("enabled")))
            self.assertEqual(
                int(resumed.artifact.model.get_booster().num_boosted_rounds()),
                4,
            )

    def test_resume_rejects_tree_boosting_family_component_drift(self) -> None:
        data = _make_processed_dataset(seed=409)
        parent_trainer = self._build_trainer({"n_estimators": 4, "max_depth": 3})
        parent_result = parent_trainer.fit_task(
            TrainTask.from_data(data, task_id="xgboost::drift_parent"),
            TrainingInit(mode="fresh"),
        )

        drift_trainer = self._build_trainer({"n_estimators": 2, "max_depth": 5})
        with self.assertRaises(ValueError) as ctx:
            drift_trainer.fit_task(
                TrainTask.from_data(data, task_id="xgboost::drift_resume"),
                TrainingInit(mode="resume", parent_state=parent_result.trainer_state),
            )
        self.assertIn("tree boosting family components changed", str(ctx.exception))

    def test_runtime_mechanisms_apply_without_forcing_continuation(self) -> None:
        data = _make_processed_dataset(seed=407)
        parent_trainer = self._build_trainer({"n_estimators": 3})
        parent_result = parent_trainer.fit_task(
            TrainTask.from_data(data, task_id="xgboost::signal_parent"),
            TrainingInit(mode="fresh"),
        )

        trainer = build_trainer(
            TrainerAssemblySpec(
                trainer_key="xgboost",
                trainer_params={
                    "n_estimators": 4,
                    "max_depth": 3,
                    "learning_rate": 0.1,
                    "subsample": 1.0,
                    "colsample_bytree": 1.0,
                    "tree_method": "hist",
                    "n_jobs": 1,
                    "random_seed": 42,
                    "verbosity": 0,
                    "mechanisms": [
                        {"key": "state_signal_view.prediction_residual", "params": {}},
                        {"key": "sample_weighting.loss_adaptive", "params": {"alpha": 0.5, "power": 1.0}},
                        {
                            "key": "sampling.row_feature_subsample",
                            "params": {"row_fraction": 0.5, "feature_fraction": 0.6, "random_seed": 7},
                        },
                        {"key": "aggregation.ensemble_summary", "params": {}},
                    ],
                },
            )
        )
        result = trainer.fit_task(
            TrainTask.from_data(data, task_id="xgboost::mechanisms"),
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
        self.assertEqual(int(summary.get("selected_rows", 0)), 36)
        self.assertEqual(int(summary.get("selected_features", 0)), 3)
        self.assertEqual(int(summary.get("num_boosted_rounds", 0)), 4)

        self.assertFalse(bool(result.artifact.metadata.get("resume", {}).get("enabled")))
        self.assertEqual(int(result.artifact.model.get_booster().num_boosted_rounds()), 4)
        self.assertEqual(len(tuple(result.artifact.input_feature_indices or ())), 3)
        self.assertEqual(len(tuple(result.trainer_state.payload.get("input_feature_indices", ()) or ())), 3)
        self.assertIn("runtime_mechanisms", result.trainer_state.payload)
        self.assertIn("tree_boosting_family", result.artifact.metadata)

        pred = np.asarray(result.artifact.predict(data.X_train[:5]), dtype=float)
        self.assertEqual(tuple(pred.shape), (5, 1))


if __name__ == "__main__":
    unittest.main()
