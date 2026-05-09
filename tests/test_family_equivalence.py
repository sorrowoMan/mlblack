from __future__ import annotations

import math
import tempfile
import unittest
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from config import TrainerAssemblySpec, build_trainer
from core.orchestration.workflow import _to_processed_bundle
from project.scaffold import _table_to_bundle, build_scaffold_spec, run_project_scaffold
from training import TrainTask, TrainingInit


def _make_csv(root: Path, *, seed: int = 907, n: int = 144) -> Path:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-03-01", periods=n, freq="D").astype(str)
    x0 = rng.normal(size=n)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    x3 = rng.normal(size=n)
    x4 = rng.normal(size=n)
    x5 = rng.normal(size=n)
    target = 1.15 * x0 - 0.68 * x1 + 0.33 * np.sin(x2) + 0.2 * x3 * x4 - 0.1 * x5
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


def _payload(
    *,
    csv_path: Path,
    output_dir: Path,
    run_name: str,
    trainer_key: str,
    trainer_params: dict[str, object],
    training_init: dict[str, object] | None = None,
) -> dict[str, object]:
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
            "training_init": dict(training_init or {"mode": "fresh"}),
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


def _evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    if yt.ndim == 1:
        yt = yt.reshape(-1, 1)
    if yp.ndim == 1:
        yp = yp.reshape(-1, 1)
    if yt.shape != yp.shape:
        raise ValueError(f"prediction shape mismatch: y_true={yt.shape}, y_pred={yp.shape}")

    err = yp - yt
    mse = float(np.mean(err**2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(err)))
    yt_flat = yt.reshape(-1)
    yp_flat = yp.reshape(-1)
    ss_tot = float(np.sum((yt_flat - np.mean(yt_flat)) ** 2))
    ss_res = float(np.sum((yp_flat - yt_flat) ** 2))
    r2 = float("nan") if ss_tot <= 1e-12 else float(1.0 - ss_res / ss_tot)
    return {"rmse": rmse, "mae": mae, "r2": r2}


def _normalize(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        if isinstance(value, float) and math.isnan(value):
            return "nan"
        return value
    if isinstance(value, np.generic):
        return _normalize(value.item())
    if isinstance(value, np.ndarray):
        return [_normalize(v) for v in value.tolist()]
    if isinstance(value, dict):
        return {str(k): _normalize(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_normalize(v) for v in value]
    return str(value)


def _state_signature_view(state: Any) -> dict[str, Any]:
    payload = dict(getattr(state, "payload", {}) or {})
    return {
        "trainer_name": str(getattr(state, "trainer_name", "")),
        "schema_signature": getattr(state, "schema_signature", None),
        "feature_signature": getattr(state, "feature_signature", None),
        "target_signature": getattr(state, "target_signature", None),
        "objective_signature": getattr(state, "objective_signature", None),
        "pipeline_signature": getattr(state, "pipeline_signature", None),
        "numericizer_signature": getattr(state, "numericizer_signature", None),
        "regime_signature": getattr(state, "regime_signature", None),
        "symbolic_family_signature": getattr(state, "symbolic_family_signature", None),
        "payload_training_signature": _normalize(payload.get("training_signature")),
        "payload_input_feature_indices": _normalize(payload.get("input_feature_indices")),
        "payload_runtime_mechanisms": _normalize(payload.get("runtime_mechanisms")),
        "payload_linear_family_signature": payload.get("linear_family_signature"),
        "payload_tree_family_signature": payload.get("tree_family_signature"),
        "payload_tree_boosting_family_signature": payload.get("tree_boosting_family_signature"),
        "payload_neural_family_signature": payload.get("neural_family_signature"),
        "metadata": _normalize(getattr(state, "metadata", {})),
    }


class TestFamilyEquivalence(unittest.TestCase):
    def _assert_metric_blocks_equal(self, left: dict[str, float], right: dict[str, float]) -> None:
        for key in ("rmse", "mae", "r2"):
            self.assertAlmostEqual(float(left[key]), float(right[key]), places=10, msg=key)

    def _fit_direct(
        self,
        *,
        spec: Any,
        processed: Any,
        init: TrainingInit | None = None,
    ):
        trainer = build_trainer(
            TrainerAssemblySpec(
                trainer_key=spec.train.trainer_key,
                trainer_params=dict(spec.train.trainer_params),
                pipeline_key=spec.train.pipeline_key,
                pipeline_params=dict(spec.train.pipeline_params),
                biases=tuple(spec.train.biases),
            )
        )
        task = TrainTask.from_data(
            processed,
            task_id=f"{spec.train.run_name}::fit",
            metadata={
                "run_name": str(spec.train.run_name),
                "trainer_name": str(getattr(trainer, "name", type(trainer).__name__)),
                "model_id": None,
            },
        )
        return trainer.fit_task(task, init or TrainingInit(mode="fresh"))

    def _assert_fit_result_equivalent(
        self,
        *,
        processed: Any,
        direct_result: Any,
        scaffold_result: Any,
    ) -> None:
        direct_train_pred = np.asarray(direct_result.artifact.predict(processed.X_train), dtype=float)
        scaffold_train_pred = np.asarray(scaffold_result.artifact.predict(scaffold_result.processed.X_train), dtype=float)
        np.testing.assert_allclose(direct_train_pred, scaffold_train_pred, rtol=1e-8, atol=1e-8)

        self.assertIsNotNone(processed.X_test)
        self.assertIsNotNone(processed.y_test)
        self.assertIsNotNone(scaffold_result.processed.X_test)
        self.assertIsNotNone(scaffold_result.processed.y_test)
        direct_test_pred = np.asarray(direct_result.artifact.predict(np.asarray(processed.X_test, dtype=float)), dtype=float)
        scaffold_test_pred = np.asarray(
            scaffold_result.artifact.predict(np.asarray(scaffold_result.processed.X_test, dtype=float)),
            dtype=float,
        )
        np.testing.assert_allclose(direct_test_pred, scaffold_test_pred, rtol=1e-8, atol=1e-8)

        direct_metrics = {
            "train": _evaluate_regression(processed.y_train, direct_train_pred),
            "test": _evaluate_regression(np.asarray(processed.y_test, dtype=float), direct_test_pred),
        }
        self._assert_metric_blocks_equal(direct_metrics["train"], scaffold_result.metrics["train"])
        self._assert_metric_blocks_equal(direct_metrics["test"], scaffold_result.metrics["test"])

        self.assertEqual(
            _normalize(direct_result.artifact.metadata),
            _normalize(scaffold_result.artifact.metadata),
        )
        self.assertIsNotNone(direct_result.trainer_state)
        self.assertIsNotNone(scaffold_result.trainer_state)
        self.assertEqual(
            _state_signature_view(direct_result.trainer_state),
            _state_signature_view(scaffold_result.trainer_state),
        )

    def _assert_family_equivalent(self, trainer_key: str, trainer_params: dict[str, object]) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv_path = _make_csv(root)
            payload = _payload(
                csv_path=csv_path,
                output_dir=root / f"runs_{trainer_key}",
                run_name=f"equiv_{trainer_key}",
                trainer_key=trainer_key,
                trainer_params=trainer_params,
            )
            spec = build_scaffold_spec(payload)
            bundle = _table_to_bundle(spec.data)
            processed, _ = _to_processed_bundle(bundle)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                direct_result = self._fit_direct(spec=spec, processed=processed, init=TrainingInit(mode="fresh"))
                scaffold_result = run_project_scaffold(spec)

            self._assert_fit_result_equivalent(
                processed=processed,
                direct_result=direct_result,
                scaffold_result=scaffold_result,
            )

    def _assert_continuation_equivalent(
        self,
        *,
        trainer_key: str,
        parent_params: dict[str, object],
        child_params: dict[str, object],
        mode: str,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv_path = _make_csv(root, seed=931)

            parent_spec = build_scaffold_spec(
                _payload(
                    csv_path=csv_path,
                    output_dir=root / f"runs_{trainer_key}_parent",
                    run_name=f"equiv_{trainer_key}_parent",
                    trainer_key=trainer_key,
                    trainer_params=parent_params,
                    training_init={"mode": "fresh"},
                )
            )
            child_spec = build_scaffold_spec(
                _payload(
                    csv_path=csv_path,
                    output_dir=root / f"runs_{trainer_key}_child",
                    run_name=f"equiv_{trainer_key}_child",
                    trainer_key=trainer_key,
                    trainer_params=child_params,
                    training_init={"mode": str(mode)},
                )
            )

            bundle = _table_to_bundle(parent_spec.data)
            processed, _ = _to_processed_bundle(bundle)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                direct_parent = self._fit_direct(spec=parent_spec, processed=processed, init=TrainingInit(mode="fresh"))
                scaffold_parent = run_project_scaffold(parent_spec)
                direct_child = self._fit_direct(
                    spec=child_spec,
                    processed=processed,
                    init=TrainingInit(mode=str(mode), parent_state=direct_parent.trainer_state),
                )
                child_spec.train.training_init["parent_state"] = scaffold_parent.trainer_state
                scaffold_child = run_project_scaffold(child_spec)

            self._assert_fit_result_equivalent(
                processed=processed,
                direct_result=direct_child,
                scaffold_result=scaffold_child,
            )

    def test_ridge_direct_vs_scaffold_equivalent(self) -> None:
        self._assert_family_equivalent(
            "ridge",
            {
                "family_spec": {
                    "regularization": {
                        "l2": 1e-6,
                    }
                }
            },
        )

    def test_random_forest_direct_vs_scaffold_equivalent(self) -> None:
        self._assert_family_equivalent(
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
        )

    def test_xgboost_direct_vs_scaffold_equivalent(self) -> None:
        self._assert_family_equivalent(
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
                }
            },
        )

    def test_extra_trees_direct_vs_scaffold_equivalent(self) -> None:
        self._assert_family_equivalent(
            "extra_trees",
            {
                "family_spec": {
                    "ensemble": {
                        "ensemble_kind": "extra_trees",
                        "n_estimators": 8,
                        "n_jobs": 1,
                        "random_seed": 42,
                    },
                    "sampling": {
                        "bootstrap": False,
                        "max_features": 0.75,
                    },
                    "splitter": {
                        "criterion": "squared_error",
                        "splitter": "random",
                    },
                    "regularization": {
                        "max_depth": 4,
                        "min_samples_leaf": 1,
                    },
                }
            },
        )

    def test_bagging_direct_vs_scaffold_equivalent(self) -> None:
        self._assert_family_equivalent(
            "bagging",
            {
                "family_spec": {
                    "ensemble": {
                        "ensemble_kind": "bagging",
                        "n_estimators": 8,
                        "n_jobs": 1,
                        "random_seed": 42,
                    },
                    "sampling": {
                        "bootstrap": True,
                        "bootstrap_features": False,
                        "max_samples": 0.9,
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
        )

    def test_adaboost_direct_vs_scaffold_equivalent(self) -> None:
        self._assert_family_equivalent(
            "adaboost",
            {
                "family_spec": {
                    "ensemble": {
                        "ensemble_kind": "adaboost",
                        "n_estimators": 8,
                        "learning_rate": 0.5,
                        "loss": "linear",
                        "random_seed": 42,
                    },
                    "sampling": {
                        "max_features": 1.0,
                    },
                    "splitter": {
                        "criterion": "squared_error",
                    },
                    "regularization": {
                        "max_depth": 3,
                        "min_samples_leaf": 1,
                    },
                }
            },
        )

    def test_torch_mlp_direct_vs_scaffold_equivalent(self) -> None:
        self._assert_family_equivalent(
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
        )

    def test_sklearn_mlp_direct_vs_scaffold_equivalent(self) -> None:
        self._assert_family_equivalent(
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
        )

    def test_random_forest_resume_direct_vs_scaffold_equivalent(self) -> None:
        parent_params = {
            "family_spec": {
                "ensemble": {
                    "ensemble_kind": "random_forest",
                    "n_estimators": 4,
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
        }
        child_params = {
            "family_spec": {
                "ensemble": {
                    "ensemble_kind": "random_forest",
                    "n_estimators": 3,
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
        }
        self._assert_continuation_equivalent(
            trainer_key="random_forest",
            parent_params=parent_params,
            child_params=child_params,
            mode="resume",
        )

    def test_torch_mlp_resume_direct_vs_scaffold_equivalent(self) -> None:
        parent_params = {
            "device": "cpu",
            "hidden_dims": (16, 8),
            "epochs": 4,
            "batch_size": 16,
            "val_ratio": 0.2,
            "early_stop_patience": 20,
            "random_seed": 42,
        }
        child_params = {
            "device": "cpu",
            "hidden_dims": (16, 8),
            "epochs": 6,
            "batch_size": 16,
            "val_ratio": 0.2,
            "early_stop_patience": 20,
            "random_seed": 42,
        }
        self._assert_continuation_equivalent(
            trainer_key="mlp_torch",
            parent_params=parent_params,
            child_params=child_params,
            mode="resume",
        )

    def test_sklearn_mlp_warm_start_direct_vs_scaffold_equivalent(self) -> None:
        parent_params = {
            "family_spec": {
                "backbone": {
                    "hidden_layers": (24,),
                    "activation": "relu",
                },
                "optimization": {
                    "solver": "adam",
                    "max_steps": 18,
                    "tol": 1e-4,
                    "n_iter_no_change": 20,
                    "early_stopping": False,
                    "random_seed": 42,
                },
                "batching": {
                    "batch_size": 16,
                    "validation_fraction": 0.15,
                },
            }
        }
        child_params = {
            "family_spec": {
                "backbone": {
                    "hidden_layers": (24,),
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
            }
        }
        self._assert_continuation_equivalent(
            trainer_key="sklearn_mlp",
            parent_params=parent_params,
            child_params=child_params,
            mode="warm_start",
        )


if __name__ == "__main__":
    unittest.main()
