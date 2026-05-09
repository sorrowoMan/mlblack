from __future__ import annotations

import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from project.scaffold import build_scaffold_spec, run_project_scaffold


def _make_csv(root: Path, *, seed: int = 731, n: int = 144) -> Path:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-01-01", periods=n, freq="D").astype(str)
    x0 = rng.normal(size=n)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    x3 = rng.normal(size=n)
    x4 = rng.normal(size=n)
    x5 = rng.normal(size=n)
    target = 1.25 * x0 - 0.75 * x1 + 0.35 * np.sin(x2) + 0.18 * x3 * x4 - 0.12 * x5
    frame = pd.DataFrame(
        {
            "date": dates,
            "x0": x0,
            "x1": x1,
            "x2": x2,
            "x3": x3,
            "x4": x4,
            "x5": x5,
            "target": target,
        }
    )
    path = root / "processed.csv"
    frame.to_csv(path, index=False)
    return path


def _payload(*, csv_path: Path, output_dir: Path, run_name: str, trainer_key: str, trainer_params: dict[str, object]) -> dict[str, object]:
    return {
        "data": {
            "csv_path": str(csv_path),
            "target_col": "target",
            "date_col": "date",
            "feature_recipe": "raw_all_numeric",
            "split_mode": "ratio",
            "test_ratio": 0.25,
            "random_seed": 42,
        },
        "train": {
            "trainer_key": trainer_key,
            "trainer_params": dict(trainer_params),
            "run_name": run_name,
            "output_dir": str(output_dir),
            "state_backend": {
                "context": {"backend": "memory"},
                "snapshot": {"backend": "memory"},
            },
            "execution": {
                "backend": "serial",
                "max_workers": 1,
                "gpu_strategy": "none",
                "default_device": "cpu",
            },
            "eval_splits": ["train", "test"],
        },
    }


class TestScaffoldFamilyReuse(unittest.TestCase):
    def test_standard_scaffold_reuses_same_assembly_across_families(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv_path = _make_csv(root)

            cases = [
                (
                    "ridge",
                    {
                        "family_spec": {
                            "regularization": {
                                "l2": 1e-6,
                            }
                        },
                    },
                    "linear",
                ),
                (
                    "random_forest",
                    {
                        "family_spec": {
                            "ensemble": {
                                "ensemble_kind": "random_forest",
                                "n_estimators": 8,
                                "n_jobs": 1,
                                "random_seed": 42,
                            },
                            "sampling": {
                                "bootstrap": True,
                                "max_features": 0.75,
                            },
                            "splitter": {
                                "criterion": "squared_error",
                            },
                            "regularization": {
                                "max_depth": 4,
                                "min_samples_leaf": 1,
                            },
                        }
                    },
                    "tree",
                ),
                (
                    "xgboost",
                    {
                        "family_spec": {
                            "boosting": {
                                "n_estimators": 24,
                                "learning_rate": 0.08,
                                "objective": "reg:squarederror",
                                "tree_method": "hist",
                            },
                            "sampling": {
                                "subsample": 0.9,
                                "colsample_bytree": 0.9,
                            },
                            "regularization": {
                                "max_depth": 3,
                            },
                            "execution": {
                                "n_jobs": 1,
                                "random_seed": 42,
                            },
                        },
                    },
                    "boosting",
                ),
                (
                    "mlp_torch",
                    {
                        "device": "cpu",
                        "family_spec": {
                            "backbone": {
                                "hidden_layers": (32, 16),
                                "activation": "relu",
                                "dropout": 0.0,
                            },
                            "optimization": {
                                "optimizer": "adamw",
                                "max_steps": 10,
                                "lr": 1e-3,
                                "weight_decay": 1e-4,
                                "early_stop_patience": 20,
                                "early_stop_min_delta": 1e-6,
                                "random_seed": 42,
                            },
                            "batching": {
                                "batch_size": 16,
                                "shuffle": True,
                                "val_ratio": 0.2,
                            },
                        },
                    },
                    "neural_torch",
                ),
                (
                    "sklearn_mlp",
                    {
                        "family_spec": {
                            "backbone": {
                                "hidden_layers": (32,),
                                "activation": "relu",
                            },
                            "optimization": {
                                "solver": "adam",
                                "max_steps": 24,
                                "tol": 1e-4,
                                "n_iter_no_change": 20,
                                "early_stopping": False,
                                "random_seed": 42,
                            },
                            "batching": {
                                "batch_size": 16,
                                "validation_fraction": 0.15,
                            },
                        },
                    },
                    "neural_sklearn",
                ),
            ]

            for trainer_key, trainer_params, family_kind in cases:
                payload = _payload(
                    csv_path=csv_path,
                    output_dir=root / f"runs_{trainer_key}",
                    run_name=f"scaffold_{trainer_key}",
                    trainer_key=trainer_key,
                    trainer_params=trainer_params,
                )
                spec = build_scaffold_spec(payload)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", ConvergenceWarning)
                    result = run_project_scaffold(spec)

                self.assertIn("train", result.metrics, msg=trainer_key)
                self.assertIn("test", result.metrics, msg=trainer_key)
                self.assertTrue(np.isfinite(float(result.metrics["test"]["rmse"])), msg=trainer_key)

                if family_kind == "tree":
                    self.assertIn("tree_family", result.artifact.metadata, msg=trainer_key)
                elif family_kind.startswith("neural"):
                    self.assertIn("neural_family", result.artifact.metadata, msg=trainer_key)
                elif family_kind == "linear":
                    self.assertIn("linear_family", result.artifact.metadata, msg=trainer_key)
                elif family_kind == "boosting":
                    self.assertIn("tree_boosting_family", result.artifact.metadata, msg=trainer_key)
                    self.assertIn("runtime_mechanisms", result.artifact.metadata, msg=trainer_key)


if __name__ == "__main__":
    unittest.main()
