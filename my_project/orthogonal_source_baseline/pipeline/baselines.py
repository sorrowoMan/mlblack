from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from config import TrainerAssemblySpec, build_trainer
from core.common.contracts import ProcessedDataset
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from training import TrainTask, TrainingInit


@dataclass(frozen=True)
class BaselineFitResult:
    rows: tuple[dict[str, Any], ...]
    neural_training_rows: tuple[dict[str, Any], ...]
    neural_curve_rows: tuple[dict[str, Any], ...]


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y = np.asarray(y_true, dtype=float).reshape(-1)
    pred = np.asarray(y_pred, dtype=float).reshape(-1)
    return {
        "rmse": float(mean_squared_error(y, pred) ** 0.5),
        "mae": float(mean_absolute_error(y, pred)),
        "r2": float(r2_score(y, pred)),
    }


def _model_factories(seed: int):
    return {
        "ridge": lambda: make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "random_forest": lambda: RandomForestRegressor(
            n_estimators=160,
            max_depth=5,
            min_samples_leaf=3,
            random_state=int(seed),
            n_jobs=-1,
        ),
        "gradient_boosting": lambda: GradientBoostingRegressor(
            n_estimators=180,
            learning_rate=0.04,
            max_depth=3,
            random_state=int(seed),
        ),
    }


def _target_scale(y_train: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = np.asarray(y_train, dtype=float)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    mean = np.mean(y, axis=0).reshape(1, -1)
    std = (np.std(y, axis=0) + 1e-8).reshape(1, -1)
    return (y - mean) / std, mean, std


def _inverse_target_scale(y_scaled: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    pred = np.asarray(y_scaled, dtype=float)
    if pred.ndim == 1:
        pred = pred.reshape(-1, 1)
    return pred * np.asarray(std, dtype=float).reshape(1, -1) + np.asarray(mean, dtype=float).reshape(1, -1)


def _build_formal_neural_trainer(seed: int):
    return build_trainer(
        TrainerAssemblySpec(
            trainer_key="sklearn_mlp",
            pipeline_key="zscore",
            trainer_params={
                "family_spec": {
                    "backbone": {
                        "hidden_layers": (96, 48),
                        "activation": "relu",
                    },
                    "optimization": {
                        "solver": "adam",
                        "alpha": 1.0e-4,
                        "learning_rate_init": 3.0e-4,
                        "max_steps": 240,
                        "tol": 1.0e-4,
                        "n_iter_no_change": 18,
                        "early_stopping": True,
                        "random_seed": int(seed),
                    },
                    "batching": {
                        "batch_size": 512,
                        "validation_fraction": 0.12,
                    },
                },
            },
        )
    )


def _scalar_or_empty(value: Any) -> float | str:
    if value is None:
        return ""
    try:
        return float(value)
    except Exception:
        return str(value)


def _fit_formal_neural_baseline(
    *,
    feature_space: str,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any], tuple[dict[str, Any], ...]]:
    y_scaled, y_mean, y_std = _target_scale(y_train)
    dataset = ProcessedDataset(
        X_train=np.asarray(X_train, dtype=float),
        y_train=np.asarray(y_scaled, dtype=float),
        X_test=np.asarray(X_test, dtype=float),
        y_test=(np.asarray(y_test, dtype=float).reshape(-1, 1) - y_mean) / y_std,
        feature_names=tuple(f"{feature_space}__x{i}" for i in range(np.asarray(X_train).shape[1])),
        target_names=("target_scaled",),
        metadata={
            "input_protocol": "orthogonal_source_baseline.neural",
            "feature_space": str(feature_space),
            "target_scaling": "zscore",
        },
    )
    trainer = _build_formal_neural_trainer(seed)
    caught_warnings: list[warnings.WarningMessage]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        result = trainer.fit_task(
            TrainTask.from_data(
                dataset,
                task_id=f"orthogonal_source_baseline::{feature_space}::neural_mlp",
                metadata={"feature_space": str(feature_space), "target_scaling": "zscore"},
            ),
            TrainingInit(mode="fresh"),
        )
        caught_warnings = list(caught)

    pred_train = _inverse_target_scale(result.artifact.predict(np.asarray(X_train, dtype=float)), y_mean, y_std)
    pred_test = _inverse_target_scale(result.artifact.predict(np.asarray(X_test, dtype=float)), y_mean, y_std)
    train_metrics = _metrics(y_train, pred_train)
    test_metrics = _metrics(y_test, pred_test)
    metric_row = {
        "feature_space": str(feature_space),
        "model": "neural_mlp",
        "feature_count": int(np.asarray(X_train).shape[1]),
        "train_rmse": float(train_metrics["rmse"]),
        "train_mae": float(train_metrics["mae"]),
        "train_r2": float(train_metrics["r2"]),
        "test_rmse": float(test_metrics["rmse"]),
        "test_mae": float(test_metrics["mae"]),
        "test_r2": float(test_metrics["r2"]),
    }

    metadata = dict(getattr(result.artifact, "metadata", {}) or {})
    diagnostics = dict(metadata.get("training_diagnostics", {}) or {})
    convergence_warnings = [
        str(w.message)
        for w in caught_warnings
        if issubclass(w.category, ConvergenceWarning)
    ]
    warning_messages = [str(w.message) for w in caught_warnings]
    report_row = {
        "feature_space": str(feature_space),
        "model": "neural_mlp",
        "trainer_key": "sklearn_mlp",
        "pipeline_key": "zscore",
        "target_scaled": True,
        "target_mean": float(np.asarray(y_mean).reshape(-1)[0]),
        "target_std": float(np.asarray(y_std).reshape(-1)[0]),
        "n_iter": int(diagnostics.get("n_iter", 0) or 0),
        "max_iter": int(diagnostics.get("max_iter", 0) or 0),
        "early_stopping": bool(diagnostics.get("early_stopping", False)),
        "stopped_by": str(diagnostics.get("stopped_by", "")),
        "reached_max_iter": bool(diagnostics.get("reached_max_iter", False)),
        "final_loss": _scalar_or_empty(diagnostics.get("loss")),
        "best_loss": _scalar_or_empty(diagnostics.get("best_loss")),
        "best_validation_score": _scalar_or_empty(diagnostics.get("best_validation_score")),
        "loss_curve_length": int(diagnostics.get("loss_curve_length", 0) or 0),
        "validation_curve_length": int(diagnostics.get("validation_curve_length", 0) or 0),
        "convergence_warning_count": int(len(convergence_warnings)),
        "warning_count": int(len(warning_messages)),
        "warnings": " | ".join(warning_messages),
    }

    curve_rows: list[dict[str, Any]] = []
    for i, value in enumerate(tuple(diagnostics.get("loss_curve", ()) or ())):
        curve_rows.append(
            {
                "feature_space": str(feature_space),
                "model": "neural_mlp",
                "curve": "loss",
                "step": int(i + 1),
                "value": float(value),
            }
        )
    for i, value in enumerate(tuple(diagnostics.get("validation_scores", ()) or ())):
        curve_rows.append(
            {
                "feature_space": str(feature_space),
                "model": "neural_mlp",
                "curve": "validation_score",
                "step": int(i + 1),
                "value": float(value),
            }
        )

    return metric_row, report_row, tuple(curve_rows)


def fit_baseline_models(
    *,
    raw_train: np.ndarray,
    raw_test: np.ndarray,
    basis_train: np.ndarray,
    basis_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    seed: int,
) -> BaselineFitResult:
    rows: list[dict[str, Any]] = []
    neural_training_rows: list[dict[str, Any]] = []
    neural_curve_rows: list[dict[str, Any]] = []
    raw_train_arr = np.asarray(raw_train, dtype=float)
    raw_test_arr = np.asarray(raw_test, dtype=float)
    basis_train_arr = np.asarray(basis_train, dtype=float)
    basis_test_arr = np.asarray(basis_test, dtype=float)
    feature_sets = {
        "raw_features": (raw_train_arr, raw_test_arr),
        "orthogonal_sources": (basis_train_arr, basis_test_arr),
        "raw_plus_orthogonal_sources": (
            np.hstack([raw_train_arr, basis_train_arr]),
            np.hstack([raw_test_arr, basis_test_arr]),
        ),
    }
    for feature_space, (X_tr, X_te) in feature_sets.items():
        if X_tr.shape[1] <= 0:
            continue
        for model_name, factory in _model_factories(int(seed)).items():
            model = factory()
            model.fit(X_tr, np.asarray(y_train, dtype=float).reshape(-1))
            pred_train = model.predict(X_tr)
            pred_test = model.predict(X_te)
            train_metrics = _metrics(y_train, pred_train)
            test_metrics = _metrics(y_test, pred_test)
            rows.append(
                {
                    "feature_space": str(feature_space),
                    "model": str(model_name),
                    "feature_count": int(X_tr.shape[1]),
                    "train_rmse": float(train_metrics["rmse"]),
                    "train_mae": float(train_metrics["mae"]),
                    "train_r2": float(train_metrics["r2"]),
                    "test_rmse": float(test_metrics["rmse"]),
                    "test_mae": float(test_metrics["mae"]),
                    "test_r2": float(test_metrics["r2"]),
                }
            )
        neural_row, neural_report, neural_curves = _fit_formal_neural_baseline(
            feature_space=str(feature_space),
            X_train=X_tr,
            X_test=X_te,
            y_train=y_train,
            y_test=y_test,
            seed=int(seed),
        )
        rows.append(neural_row)
        neural_training_rows.append(neural_report)
        neural_curve_rows.extend(neural_curves)
    return BaselineFitResult(
        rows=tuple(rows),
        neural_training_rows=tuple(neural_training_rows),
        neural_curve_rows=tuple(neural_curve_rows),
    )


def summarize_feature_space_winners(rows: tuple[Mapping[str, Any], ...]) -> dict[str, Any]:
    by_model: dict[str, dict[str, Any]] = {}
    for model_name in sorted({str(row.get("model")) for row in rows}):
        model_rows = [dict(row) for row in rows if str(row.get("model")) == model_name]
        raw = next((row for row in model_rows if row.get("feature_space") == "raw_features"), None)
        if not raw:
            continue
        raw_rmse = float(raw.get("test_rmse", float("nan")))
        candidates = {
            str(row.get("feature_space")): float(row.get("test_rmse", float("nan")))
            for row in model_rows
            if row.get("feature_space")
        }
        winner, winner_rmse = min(candidates.items(), key=lambda item: item[1])
        orth_rmse = candidates.get("orthogonal_sources", float("nan"))
        augmented_rmse = candidates.get("raw_plus_orthogonal_sources", float("nan"))
        by_model[model_name] = {
            "raw_test_rmse": raw_rmse,
            "orthogonal_test_rmse": orth_rmse,
            "raw_plus_orthogonal_test_rmse": augmented_rmse,
            "rmse_delta_orth_minus_raw": float(orth_rmse - raw_rmse),
            "rmse_delta_augmented_minus_raw": float(augmented_rmse - raw_rmse),
            "winner": str(winner),
            "winner_test_rmse": float(winner_rmse),
        }
    return by_model


__all__ = ["BaselineFitResult", "fit_baseline_models", "summarize_feature_space_winners"]
