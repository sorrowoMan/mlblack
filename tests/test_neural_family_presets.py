from __future__ import annotations

import unittest
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning

from config import TrainerAssemblySpec, build_trainer
from core.common.contracts import ProcessedDataset
from training import TrainTask, TrainingInit


def _make_processed_dataset(seed: int = 611) -> ProcessedDataset:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(96, 6))
    y = (1.1 * x[:, 0] - 0.6 * x[:, 1] + 0.35 * np.sin(x[:, 2]) + 0.2 * x[:, 3]).reshape(-1, 1)
    return ProcessedDataset(
        X_train=x,
        y_train=y,
        feature_names=("x0", "x1", "x2", "x3", "x4", "x5"),
        target_names=("y",),
        metadata={"source": "neural_family_preset_test"},
    )


class TestNeuralFamilyPresets(unittest.TestCase):
    def test_torch_grouped_family_spec_is_consumed(self) -> None:
        trainer = build_trainer(
            TrainerAssemblySpec(
                trainer_key="mlp_torch",
                trainer_params={
                    "device": "cpu",
                    "family_spec": {
                        "backbone": {
                            "hidden_layers": (20, 10),
                            "activation": "relu",
                            "dropout": 0.1,
                        },
                        "optimization": {
                            "optimizer": "adamw",
                            "max_steps": 4,
                            "lr": 5e-4,
                            "weight_decay": 1e-4,
                            "early_stop_patience": 20,
                            "early_stop_min_delta": 1e-6,
                            "random_seed": 7,
                        },
                        "batching": {
                            "batch_size": 16,
                            "shuffle": True,
                            "val_ratio": 0.2,
                        },
                    },
                },
            )
        )
        result = trainer.fit_task(
            TrainTask.from_data(_make_processed_dataset(), task_id="mlp_torch::grouped_family"),
            TrainingInit(mode="fresh"),
        )

        neural_family = dict(result.artifact.metadata.get("neural_family", {}))
        self.assertEqual(tuple(neural_family["backbone"]["hidden_layers"]), (20, 10))
        self.assertEqual(str(neural_family["backend"]["runtime_backend"]), "torch")
        self.assertIsNotNone(result.trainer_state.payload.get("neural_family_signature"))

    def test_sklearn_grouped_family_spec_is_consumed(self) -> None:
        trainer = build_trainer(
            TrainerAssemblySpec(
                trainer_key="sklearn_mlp",
                trainer_params={
                    "family_spec": {
                        "backbone": {
                            "hidden_layers": (18,),
                            "activation": "relu",
                        },
                        "optimization": {
                            "solver": "adam",
                            "max_steps": 18,
                            "tol": 1e-4,
                            "n_iter_no_change": 20,
                            "early_stopping": False,
                            "random_seed": 11,
                        },
                        "batching": {
                            "batch_size": 16,
                            "validation_fraction": 0.15,
                        },
                    },
                },
            )
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            result = trainer.fit_task(
                TrainTask.from_data(_make_processed_dataset(seed=612), task_id="sklearn_mlp::grouped_family"),
                TrainingInit(mode="fresh"),
            )

        neural_family = dict(result.artifact.metadata.get("neural_family", {}))
        self.assertEqual(tuple(neural_family["backbone"]["hidden_layers"]), (18,))
        self.assertEqual(str(neural_family["backend"]["runtime_backend"]), "scikit-learn")
        self.assertIsNotNone(result.trainer_state.payload.get("neural_family_signature"))

    def test_sklearn_warm_start_rejects_neural_family_drift(self) -> None:
        data = _make_processed_dataset(seed=613)
        parent = build_trainer(
            TrainerAssemblySpec(
                trainer_key="sklearn_mlp",
                trainer_params={
                    "family_spec": {
                        "backbone": {"hidden_layers": (24,), "activation": "relu"},
                        "optimization": {
                            "solver": "adam",
                            "max_steps": 18,
                            "early_stopping": False,
                            "random_seed": 5,
                        },
                    }
                },
            )
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            parent_result = parent.fit_task(
                TrainTask.from_data(data, task_id="sklearn_mlp::drift_parent"),
                TrainingInit(mode="fresh"),
            )

        drift = build_trainer(
            TrainerAssemblySpec(
                trainer_key="sklearn_mlp",
                trainer_params={
                    "family_spec": {
                        "backbone": {"hidden_layers": (48,), "activation": "relu"},
                        "optimization": {
                            "solver": "adam",
                            "max_steps": 18,
                            "early_stopping": False,
                            "random_seed": 5,
                        },
                    }
                },
            )
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            with self.assertRaises(ValueError) as ctx:
                drift.fit_task(
                    TrainTask.from_data(data, task_id="sklearn_mlp::drift_child"),
                    TrainingInit(mode="warm_start", parent_state=parent_result.trainer_state),
                )
        self.assertIn("neural family components changed", str(ctx.exception))

    def test_torch_gradient_norm_and_batch_priority_mechanisms(self) -> None:
        data = _make_processed_dataset(seed=614)
        parent = build_trainer(
            TrainerAssemblySpec(
                trainer_key="mlp_torch",
                trainer_params={
                    "device": "cpu",
                    "hidden_dims": (16, 8),
                    "epochs": 4,
                    "batch_size": 16,
                    "val_ratio": 0.2,
                    "early_stop_patience": 20,
                    "random_seed": 9,
                },
            )
        )
        parent_result = parent.fit_task(
            TrainTask.from_data(data, task_id="mlp_torch::gradient_parent"),
            TrainingInit(mode="fresh"),
        )

        trainer = build_trainer(
            TrainerAssemblySpec(
                trainer_key="mlp_torch",
                trainer_params={
                    "device": "cpu",
                    "hidden_dims": (16, 8),
                    "epochs": 4,
                    "batch_size": 16,
                    "val_ratio": 0.2,
                    "early_stop_patience": 20,
                    "random_seed": 9,
                    "mechanisms": [
                        {"key": "state_signal_view.gradient_norm", "params": {}},
                        {
                            "key": "sampling.batch_priority_subsample",
                            "params": {
                                "source_key": "gradient_norm_ref",
                                "batch_size": 8,
                                "num_batches": 3,
                                "mode": "topk",
                            },
                        },
                        {"key": "aggregation.ensemble_summary", "params": {}},
                    ],
                },
            )
        )
        result = trainer.fit_task(
            TrainTask.from_data(data, task_id="mlp_torch::gradient_child"),
            TrainingInit(mode="fresh", parent_artifact=parent_result.artifact),
        )

        runtime = dict(result.artifact.metadata.get("runtime_mechanisms", {}))
        trace = list(runtime.get("trace", []))
        trace_status = {(str(row.get("mechanism_key")), str(row.get("status"))) for row in trace}
        self.assertIn(("state_signal_view.gradient_norm", "applied"), trace_status)
        self.assertIn(("sampling.batch_priority_subsample", "applied"), trace_status)
        self.assertIn(("aggregation.ensemble_summary", "applied"), trace_status)

        summary = dict(runtime.get("aggregation_summary", {}))
        self.assertEqual(int(summary.get("selected_rows", 0)), 24)
        self.assertIn("gradient_norm_ref", tuple(summary.get("active_signal_keys", ())))


if __name__ == "__main__":
    unittest.main()
