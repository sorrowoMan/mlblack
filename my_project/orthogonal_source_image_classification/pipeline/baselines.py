from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y = np.asarray(y_true, dtype=int).reshape(-1)
    pred = np.asarray(y_pred, dtype=int).reshape(-1)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
    }


def _model_factories(seed: int):
    return {
        "logistic_regression": lambda: make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=1200,
                C=1.0,
                solver="lbfgs",
                random_state=int(seed),
            ),
        ),
        "random_forest": lambda: RandomForestClassifier(
            n_estimators=240,
            max_depth=None,
            min_samples_leaf=1,
            random_state=int(seed),
            n_jobs=-1,
        ),
        "gradient_boosting": lambda: GradientBoostingClassifier(
            n_estimators=140,
            learning_rate=0.05,
            max_depth=2,
            random_state=int(seed),
        ),
        "neural_mlp": lambda: make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(96, 48),
                activation="relu",
                solver="adam",
                alpha=1.0e-4,
                batch_size=128,
                learning_rate_init=5.0e-4,
                max_iter=260,
                early_stopping=True,
                validation_fraction=0.12,
                n_iter_no_change=18,
                random_state=int(seed),
            ),
        ),
    }


def fit_classification_baselines(
    *,
    raw_train: np.ndarray,
    raw_test: np.ndarray,
    formula_pool_train: np.ndarray,
    formula_pool_test: np.ndarray,
    representation_train: np.ndarray,
    representation_test: np.ndarray,
    basis_train: np.ndarray,
    basis_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    seed: int,
) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    raw_train_arr = np.asarray(raw_train, dtype=float)
    raw_test_arr = np.asarray(raw_test, dtype=float)
    formula_pool_train_arr = np.asarray(formula_pool_train, dtype=float)
    formula_pool_test_arr = np.asarray(formula_pool_test, dtype=float)
    representation_train_arr = np.asarray(representation_train, dtype=float)
    representation_test_arr = np.asarray(representation_test, dtype=float)
    basis_train_arr = np.asarray(basis_train, dtype=float)
    basis_test_arr = np.asarray(basis_test, dtype=float)
    feature_sets = {
        "raw_pixels": (raw_train_arr, raw_test_arr),
        "formula_pool": (formula_pool_train_arr, formula_pool_test_arr),
        "image_representation": (representation_train_arr, representation_test_arr),
        "orthogonal_sources": (basis_train_arr, basis_test_arr),
        "image_representation_plus_orthogonal_sources": (
            np.hstack([representation_train_arr, basis_train_arr]),
            np.hstack([representation_test_arr, basis_test_arr]),
        ),
    }
    for feature_space, (X_tr, X_te) in feature_sets.items():
        if X_tr.shape[1] <= 0:
            continue
        for model_name, factory in _model_factories(int(seed)).items():
            model = factory()
            model.fit(X_tr, np.asarray(y_train, dtype=int).reshape(-1))
            pred_train = model.predict(X_tr)
            pred_test = model.predict(X_te)
            train_metrics = _metrics(y_train, pred_train)
            test_metrics = _metrics(y_test, pred_test)
            rows.append(
                {
                    "feature_space": str(feature_space),
                    "model": str(model_name),
                    "feature_count": int(X_tr.shape[1]),
                    "train_accuracy": float(train_metrics["accuracy"]),
                    "train_balanced_accuracy": float(train_metrics["balanced_accuracy"]),
                    "train_macro_f1": float(train_metrics["macro_f1"]),
                    "test_accuracy": float(test_metrics["accuracy"]),
                    "test_balanced_accuracy": float(test_metrics["balanced_accuracy"]),
                    "test_macro_f1": float(test_metrics["macro_f1"]),
                }
            )
    return tuple(rows)


def summarize_classification_winners(rows: tuple[Mapping[str, Any], ...]) -> dict[str, Any]:
    by_model: dict[str, dict[str, Any]] = {}
    for model_name in sorted({str(row.get("model")) for row in rows}):
        model_rows = [dict(row) for row in rows if str(row.get("model")) == model_name]
        raw = next((row for row in model_rows if row.get("feature_space") == "raw_pixels"), None)
        representation = next((row for row in model_rows if row.get("feature_space") == "image_representation"), None)
        if not raw or not representation:
            continue
        raw_acc = float(raw.get("test_accuracy", float("nan")))
        representation_acc = float(representation.get("test_accuracy", float("nan")))
        candidates = {
            str(row.get("feature_space")): float(row.get("test_accuracy", float("nan")))
            for row in model_rows
            if row.get("feature_space")
        }
        winner, winner_acc = max(candidates.items(), key=lambda item: item[1])
        by_model[model_name] = {
            "raw_test_accuracy": raw_acc,
            "image_representation_test_accuracy": representation_acc,
            "orthogonal_test_accuracy": candidates.get("orthogonal_sources", float("nan")),
            "image_representation_plus_orthogonal_test_accuracy": candidates.get(
                "image_representation_plus_orthogonal_sources",
                float("nan"),
            ),
            "accuracy_delta_orth_minus_raw": float(candidates.get("orthogonal_sources", float("nan")) - raw_acc),
            "accuracy_delta_representation_minus_raw": float(representation_acc - raw_acc),
            "accuracy_delta_orth_minus_representation": float(
                candidates.get("orthogonal_sources", float("nan")) - representation_acc
            ),
            "accuracy_delta_augmented_minus_representation": float(
                candidates.get("image_representation_plus_orthogonal_sources", float("nan")) - representation_acc
            ),
            "winner": str(winner),
            "winner_test_accuracy": float(winner_acc),
        }
    return by_model


__all__ = ["fit_classification_baselines", "summarize_classification_winners"]
