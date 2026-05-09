from __future__ import annotations

import unittest

import numpy as np

from config import TrainerAssemblySpec, build_trainer
from core.common.contracts import ProcessedDataset
from training import TrainTask, TrainingInit


def _make_processed_dataset(seed: int = 541) -> ProcessedDataset:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(96, 6))
    y = (1.25 * x[:, 0] - 0.65 * x[:, 1] + 0.18 * x[:, 2] + 0.11).reshape(-1, 1)
    return ProcessedDataset(
        X_train=x,
        y_train=y,
        feature_names=("x0", "x1", "x2", "x3", "x4", "x5"),
        target_names=("y",),
        metadata={"source": "linear_family_preset_test"},
    )


class TestLinearFamilyPresets(unittest.TestCase):
    def _build_trainer(self, trainer_params: dict[str, object] | None = None):
        params = {
            "l2": 1e-6,
        }
        if trainer_params:
            params.update(dict(trainer_params))
        return build_trainer(
            TrainerAssemblySpec(
                trainer_key="ridge",
                trainer_params=params,
            )
        )

    def test_fit_task_emits_linear_family_metadata(self) -> None:
        trainer = self._build_trainer()
        result = trainer.fit_task(
            TrainTask.from_data(_make_processed_dataset(), task_id="ridge::fresh"),
            TrainingInit(mode="fresh"),
        )

        self.assertIsNotNone(result.trainer_state)
        self.assertIn("linear_family", result.artifact.metadata)
        self.assertEqual(str(result.artifact.metadata["linear_family"]["backend"]["solver_kind"]), "ridge")
        self.assertIsNotNone(result.trainer_state.payload.get("linear_family_signature"))

    def test_grouped_family_spec_is_consumed(self) -> None:
        trainer = self._build_trainer(
            {
                "family_spec": {
                    "backend": {
                        "solver_kind": "ridge",
                    },
                    "regularization": {
                        "penalty": "l2",
                        "l2": 0.25,
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
            TrainTask.from_data(_make_processed_dataset(seed=542), task_id="ridge::grouped"),
            TrainingInit(mode="fresh"),
        )

        self.assertAlmostEqual(float(result.artifact.metadata["linear_family"]["regularization"]["l2"]), 0.25, places=12)
        pred = np.asarray(result.artifact.predict(_make_processed_dataset(seed=543).X_train[:4]), dtype=float)
        self.assertEqual(tuple(pred.shape), (4, 1))

    def test_resume_rejects_linear_family_component_drift(self) -> None:
        data = _make_processed_dataset(seed=544)
        parent = self._build_trainer(
            {
                "family_spec": {
                    "task_head": {
                        "uncertainty_mode": "residual_std",
                    }
                }
            }
        )
        parent_result = parent.fit_task(
            TrainTask.from_data(data, task_id="ridge::parent"),
            TrainingInit(mode="fresh"),
        )

        drift = self._build_trainer(
            {
                "family_spec": {
                    "task_head": {
                        "uncertainty_mode": "calibrated_residual",
                    }
                }
            }
        )
        with self.assertRaises(ValueError) as ctx:
            drift.fit_task(
                TrainTask.from_data(data, task_id="ridge::resume_drift"),
                TrainingInit(mode="resume", parent_state=parent_result.trainer_state),
            )
        self.assertIn("linear family components changed", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
