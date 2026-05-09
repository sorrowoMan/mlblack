from __future__ import annotations

import unittest

import numpy as np

from config import TrainerAssemblySpec, build_trainer
from core.common.contracts import ProcessedDataset
from training import TrainTask, TrainingCompatibilityError, TrainingInit


def _make_processed_dataset(seed: int = 622) -> ProcessedDataset:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(96, 6))
    y = (1.1 * x[:, 0] - 0.9 * x[:, 1] + 0.4 * np.sin(x[:, 2]) + 0.2 * x[:, 3]).reshape(-1, 1)
    return ProcessedDataset(
        X_train=x,
        y_train=y,
        feature_names=("x0", "x1", "x2", "x3", "x4", "x5"),
        target_names=("y",),
        metadata={"source": "tree_ensemble_preset_test"},
    )


class TestTreeEnsemblePresetsAndMechanisms(unittest.TestCase):
    def _build_trainer(self, trainer_key: str, trainer_params: dict[str, object] | None = None):
        params: dict[str, object] = {
            "n_estimators": 4,
            "max_depth": 4,
            "max_features": 1.0,
            "criterion": "squared_error",
            "random_seed": 42,
            "n_jobs": 1,
        }
        if trainer_key == "adaboost":
            params.update(
                {
                    "n_estimators": 8,
                    "max_depth": 3,
                    "learning_rate": 0.5,
                    "loss": "linear",
                }
            )
            params.pop("n_jobs", None)
        if trainer_key == "bagging":
            params.update(
                {
                    "bootstrap": True,
                    "bootstrap_features": False,
                    "max_samples": 1.0,
                }
            )
        if trainer_key == "extra_trees":
            params.update(
                {
                    "bootstrap": False,
                }
            )
        if trainer_params:
            params.update(dict(trainer_params))
        return build_trainer(
            TrainerAssemblySpec(
                trainer_key=trainer_key,
                trainer_params=params,
            )
        )

    def test_presets_fit_and_emit_tree_family_metadata(self) -> None:
        data = _make_processed_dataset()
        for trainer_key in ("extra_trees", "bagging", "adaboost"):
            trainer = self._build_trainer(trainer_key)
            result = trainer.fit_task(
                TrainTask.from_data(data, task_id=f"{trainer_key}::fresh"),
                TrainingInit(mode="fresh"),
            )

            self.assertIsNotNone(result.trainer_state)
            self.assertEqual(
                str(result.artifact.metadata["tree_family"]["ensemble"]["ensemble_kind"]),
                trainer_key,
            )
            self.assertIn("runtime_mechanisms", result.artifact.metadata)
            active = list(result.artifact.metadata["runtime_mechanisms"].get("active_components", []))
            self.assertTrue(any(str(row.get("key")) == "aggregation.ensemble_summary" for row in active))
            pred = np.asarray(result.artifact.predict(data.X_train[:4]), dtype=float)
            self.assertEqual(tuple(pred.shape), (4, 1))
            unc = np.asarray(result.artifact.uncertainty(data.X_train[:4]), dtype=float)
            self.assertEqual(tuple(unc.shape), (4, 1))

    def test_adaboost_resume_is_rejected_by_contract(self) -> None:
        data = _make_processed_dataset(seed=623)
        parent = self._build_trainer("adaboost")
        parent_result = parent.fit_task(
            TrainTask.from_data(data, task_id="adaboost::parent"),
            TrainingInit(mode="fresh"),
        )

        child = self._build_trainer("adaboost")
        with self.assertRaises(TrainingCompatibilityError):
            child.fit_task(
                TrainTask.from_data(data, task_id="adaboost::resume"),
                TrainingInit(mode="resume", parent_state=parent_result.trainer_state),
            )

    def test_runtime_mechanisms_apply_state_weighting_and_sampling(self) -> None:
        data = _make_processed_dataset(seed=624)
        parent = self._build_trainer("random_forest", {"n_estimators": 3, "max_depth": 3})
        parent_result = parent.fit_task(
            TrainTask.from_data(data, task_id="random_forest::parent"),
            TrainingInit(mode="fresh"),
        )

        trainer = self._build_trainer(
            "random_forest",
            {
                "n_estimators": 4,
                "max_depth": 3,
                "mechanisms": [
                    {"key": "state_signal_view.prediction_residual", "params": {}},
                    {"key": "sample_weighting.loss_adaptive", "params": {"alpha": 0.5, "power": 1.0}},
                    {
                        "key": "sampling.row_feature_subsample",
                        "params": {"row_fraction": 0.6, "feature_fraction": 0.75, "random_seed": 7},
                    },
                    {"key": "aggregation.ensemble_summary", "params": {}},
                ],
            },
        )
        result = trainer.fit_task(
            TrainTask.from_data(data, task_id="random_forest::mechanisms"),
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
        self.assertEqual(int(summary.get("selected_rows", 0)), 57)
        self.assertEqual(int(summary.get("selected_features", 0)), 4)
        self.assertEqual(int(summary.get("estimator_count", 0)), 4)
        self.assertIsNotNone(result.artifact.input_feature_indices)
        self.assertEqual(len(tuple(result.artifact.input_feature_indices or ())), 4)
        self.assertEqual(len(tuple(result.trainer_state.payload.get("input_feature_indices", ()) or ())), 4)

        pred = np.asarray(result.artifact.predict(data.X_train[:5]), dtype=float)
        self.assertEqual(tuple(pred.shape), (5, 1))


if __name__ == "__main__":
    unittest.main()
