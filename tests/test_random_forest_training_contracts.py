from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from config import TrainerAssemblySpec, build_trainer
from core.common.contracts import ProcessedDataset
from training import TrainTask, TrainingInit


def _make_processed_dataset(seed: int = 515) -> ProcessedDataset:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(96, 6))
    y = (1.2 * x[:, 0] - 0.7 * x[:, 1] + 0.5 * np.sin(x[:, 2]) + 0.15 * x[:, 3]).reshape(-1, 1)
    return ProcessedDataset(
        X_train=x,
        y_train=y,
        feature_names=("x0", "x1", "x2", "x3", "x4", "x5"),
        target_names=("y",),
        metadata={"source": "random_forest_training_contract_test"},
    )


class TestRandomForestTrainingContracts(unittest.TestCase):
    def _build_trainer(self, trainer_params: dict[str, object] | None = None):
        params = {
            "n_estimators": 4,
            "max_depth": 4,
            "max_features": 1.0,
            "bootstrap": True,
            "criterion": "squared_error",
            "n_jobs": 1,
            "random_seed": 42,
        }
        if trainer_params:
            params.update(dict(trainer_params))
        return build_trainer(
            TrainerAssemblySpec(
                trainer_key="random_forest",
                trainer_params=params,
            )
        )

    def test_fit_task_emits_trainer_state_and_tree_family_metadata(self) -> None:
        trainer = self._build_trainer({"n_estimators": 5})
        result = trainer.fit_task(
            TrainTask.from_data(_make_processed_dataset(), task_id="random_forest::fresh"),
            TrainingInit(mode="fresh"),
        )

        self.assertIsNotNone(result.trainer_state)
        self.assertEqual(str(result.report["training_mode"]), "fresh")
        self.assertIn("training_signature", result.artifact.metadata)
        self.assertIn("tree_family", result.artifact.metadata)
        self.assertEqual(result.artifact.metadata["tree_family"]["ensemble"]["ensemble_kind"], "random_forest")
        self.assertEqual(len(result.artifact.model.estimators_), 5)
        self.assertIsNotNone(result.trainer_state.payload.get("tree_family_signature"))

    def test_grouped_family_spec_is_consumed(self) -> None:
        trainer = self._build_trainer(
            {
                "family_spec": {
                    "ensemble": {
                        "n_estimators": 6,
                        "oob_score": False,
                        "n_jobs": 1,
                        "random_seed": 7,
                    },
                    "sampling": {
                        "bootstrap": True,
                        "max_features": 0.5,
                    },
                    "splitter": {
                        "criterion": "squared_error",
                    },
                    "regularization": {
                        "max_depth": 3,
                        "min_samples_leaf": 2,
                    },
                    "task_head": {
                        "task": "point",
                        "objective_family": "regression",
                        "uncertainty_mode": "ensemble_std",
                    },
                }
            }
        )
        result = trainer.fit_task(
            TrainTask.from_data(_make_processed_dataset(seed=516), task_id="random_forest::grouped"),
            TrainingInit(mode="fresh"),
        )

        self.assertEqual(len(result.artifact.model.estimators_), 6)
        self.assertEqual(result.artifact.metadata["tree_family"]["sampling"]["max_features"], 0.5)
        unc = np.asarray(result.artifact.uncertainty(_make_processed_dataset(seed=517).X_train[:3]), dtype=float)
        self.assertEqual(tuple(unc.shape), (3, 1))

    def test_save_load_and_resume_append_trees(self) -> None:
        data = _make_processed_dataset(seed=518)
        parent_trainer = self._build_trainer({"n_estimators": 4})
        parent_result = parent_trainer.fit_task(
            TrainTask.from_data(data, task_id="random_forest::parent"),
            TrainingInit(mode="fresh"),
        )

        self.assertEqual(len(parent_result.artifact.model.estimators_), 4)

        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / "random_forest_state.pkl"
            parent_trainer.save_trainer_state(state_path, parent_result.trainer_state)
            loaded_state = parent_trainer.load_trainer_state(state_path)

            resume_trainer = self._build_trainer({"n_estimators": 3})
            resumed = resume_trainer.fit_task(
                TrainTask.from_data(data, task_id="random_forest::resume"),
                TrainingInit(mode="resume", parent_state=loaded_state),
            )
            self.assertEqual(str(resumed.report["training_mode"]), "resume")
            self.assertEqual(len(resumed.artifact.model.estimators_), 7)

    def test_resume_rejects_tree_family_component_drift(self) -> None:
        data = _make_processed_dataset(seed=519)
        parent_trainer = self._build_trainer({"n_estimators": 4, "max_features": 1.0})
        parent_result = parent_trainer.fit_task(
            TrainTask.from_data(data, task_id="random_forest::drift_parent"),
            TrainingInit(mode="fresh"),
        )

        drift_trainer = self._build_trainer({"n_estimators": 2, "max_features": 0.5})
        with self.assertRaises(ValueError) as ctx:
            drift_trainer.fit_task(
                TrainTask.from_data(data, task_id="random_forest::drift_resume"),
                TrainingInit(mode="resume", parent_state=parent_result.trainer_state),
            )
        self.assertIn("tree family components changed", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
