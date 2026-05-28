from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from mlblack.core.contracts import ComponentContract
from mlblack.core.problem import LearningProblem
from mlblack.core.types import Feedback, UnknownState
from mlblack.pipeline.data_views import NumericDataView


class SupervisedClassificationProblem(LearningProblem):
    name = "supervised_classification"
    context_requires = ('candidate.model', 'data.X_train', 'data.y_train')
    context_optional = ('model.predict_proba', 'data.X_valid', 'data.y_valid')
    context_provides = ('feedback.objectives', 'feedback.metrics', 'feedback.residuals', 'feedback.signals')
    context_mutates = ()
    context_cache = ()
    requires_metrics = ()
    metrics_fallback = "strict"
    context_notes = 'Reads context fields: candidate.model, data.X_train, data.y_train; provides feedback.objectives, feedback.metrics, feedback.residuals, feedback.signals.'
    contract = ComponentContract(
        name=name,
        requires=("candidate.model", "data.X_train", "data.y_train"),
        optional=("model.predict_proba", "data.X_valid", "data.y_valid"),
        provides=("feedback.objectives", "feedback.metrics", "feedback.residuals", "feedback.signals"),
        supports_gradient=False,
        supports_batch=False,
        supports_resume=False,
        metadata={"task": "classification"},
    )

    def __init__(
        self,
        data: NumericDataView,
        *,
        use_valid_objective: bool = True,
        complexity_weight: float = 0.0,
        objective_metrics: Sequence[str] = ("log_loss", "error_rate"),
        positive_label: Any | None = None,
    ) -> None:
        self.data = data
        self.use_valid_objective = bool(use_valid_objective)
        self.complexity_weight = float(complexity_weight)
        self.objective_metrics = tuple(str(item) for item in objective_metrics)
        self.positive_label = positive_label

    def evaluate(self, model: Any, state: UnknownState, context: Mapping[str, Any]) -> Feedback:
        _ = state
        _ = context
        train_metrics, train_residual = _classification_metrics(
            model,
            self.data.X_train,
            self.data.y_train,
            prefix="train",
            positive_label=self.positive_label,
        )
        metrics = dict(train_metrics)
        objective_source = train_metrics
        objective_prefix = "train"
        residual = train_residual
        if self.data.X_valid is not None and self.data.y_valid is not None:
            valid_metrics, valid_residual = _classification_metrics(
                model,
                self.data.X_valid,
                self.data.y_valid,
                prefix="valid",
                positive_label=self.positive_label,
            )
            metrics.update(valid_metrics)
            residual = valid_residual
            if self.use_valid_objective:
                objective_source = valid_metrics
                objective_prefix = "valid"
        complexity = _safe_complexity(model)
        metrics["complexity.model"] = complexity
        objectives = _build_objectives(objective_source, objective_prefix, self.objective_metrics)
        if self.complexity_weight > 0.0:
            objectives = np.concatenate([objectives, np.asarray([self.complexity_weight * complexity], dtype=float)])
        return Feedback(
            objectives=objectives,
            residuals=residual,
            metrics=metrics,
            signals={
                "task": "classification",
                "has_probability": hasattr(model, "predict_proba"),
                "primary_objectives": tuple(self.objective_metrics),
                "primary_prefix": objective_prefix,
            },
        )

    def describe(self) -> Mapping[str, Any]:
        return {
            "name": self.name,
            "task": "classification",
            "n_train": int(self.data.X_train.shape[0]),
            "n_features": int(self.data.X_train.shape[1]),
            "has_valid": self.data.X_valid is not None,
            "objective_metrics": tuple(self.objective_metrics),
            "positive_label": self.positive_label,
        }


def _classification_metrics(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    *,
    prefix: str,
    positive_label: Any | None = None,
) -> tuple[dict[str, float], np.ndarray]:
    target = np.asarray(y).reshape(-1)
    classes = _class_order(target, getattr(model, "classes_", None))
    if hasattr(model, "predict_proba"):
        proba = np.asarray(model.predict_proba(X), dtype=float)
        if proba.ndim == 1:
            proba = np.column_stack([1.0 - proba, proba])
        pred_idx = np.argmax(proba, axis=1)
        model_classes = tuple(getattr(model, "classes_", classes))
        pred = np.asarray([model_classes[int(idx)] if int(idx) < len(model_classes) else int(idx) for idx in pred_idx])
    else:
        pred = np.asarray(model.predict(X)).reshape(-1)
        proba = _onehot_proba(pred, classes)
    if proba.shape[1] != len(classes):
        classes = tuple(range(proba.shape[1]))
    proba = np.clip(proba, 1e-12, 1.0)
    proba = proba / np.sum(proba, axis=1, keepdims=True)
    class_to_idx = {value: idx for idx, value in enumerate(classes)}
    y_idx = np.asarray([class_to_idx.get(value, 0) for value in target], dtype=int)
    log_loss = -float(np.mean(np.log(proba[np.arange(target.shape[0]), y_idx])))
    accuracy = float(np.mean(pred == target))
    precision, recall, f1 = _macro_precision_recall_f1(target, pred, classes)
    metrics = {
        f"{prefix}.accuracy": accuracy,
        f"{prefix}.error_rate": 1.0 - accuracy,
        f"{prefix}.log_loss": log_loss,
        f"{prefix}.precision_macro": precision,
        f"{prefix}.recall_macro": recall,
        f"{prefix}.f1_macro": f1,
    }
    if len(classes) == 2:
        pos = positive_label if positive_label is not None else classes[-1]
        pos_idx = class_to_idx.get(pos, len(classes) - 1)
        binary_y = (target == pos).astype(int)
        score = proba[:, int(pos_idx)]
        metrics[f"{prefix}.auc_roc"] = _binary_auc(binary_y, score)
        metrics[f"{prefix}.average_precision"] = _average_precision(binary_y, score)
        p_bin, r_bin, f1_bin = _binary_precision_recall_f1(binary_y, (score >= 0.5).astype(int))
        metrics[f"{prefix}.precision"] = p_bin
        metrics[f"{prefix}.recall"] = r_bin
        metrics[f"{prefix}.f1"] = f1_bin
    residual = (pred != target).astype(float)
    return metrics, residual


def _build_objectives(metrics: Mapping[str, float], prefix: str, objective_metrics: Sequence[str]) -> np.ndarray:
    values: list[float] = []
    for name in objective_metrics:
        key = f"{prefix}.{name}"
        if key not in metrics:
            raise ValueError(f"classification objective metric is not available: {key}")
        value = float(metrics[key])
        if name in {"accuracy", "auc_roc", "average_precision", "precision", "recall", "f1", "f1_macro"}:
            value = 1.0 - value
        values.append(value)
    return np.asarray(values or [float(metrics[f"{prefix}.log_loss"])], dtype=float)


def _class_order(target: np.ndarray, model_classes: Any = None) -> tuple[Any, ...]:
    if model_classes is not None:
        values = tuple(np.asarray(model_classes).reshape(-1).tolist())
        if values:
            return values
    return tuple(np.unique(target).tolist())


def _onehot_proba(pred: np.ndarray, classes: Sequence[Any]) -> np.ndarray:
    out = np.full((pred.shape[0], len(classes)), 1e-12, dtype=float)
    class_to_idx = {value: idx for idx, value in enumerate(classes)}
    for idx, value in enumerate(pred):
        out[idx, class_to_idx.get(value, 0)] = 1.0
    return out


def _macro_precision_recall_f1(y_true: np.ndarray, y_pred: np.ndarray, classes: Sequence[Any]) -> tuple[float, float, float]:
    rows = [_binary_precision_recall_f1((y_true == cls).astype(int), (y_pred == cls).astype(int)) for cls in classes]
    if not rows:
        return 0.0, 0.0, 0.0
    return tuple(float(np.mean([row[idx] for row in rows])) for idx in range(3))  # type: ignore[return-value]


def _binary_precision_recall_f1(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    y = np.asarray(y_true, dtype=int).reshape(-1)
    p = np.asarray(y_pred, dtype=int).reshape(-1)
    tp = float(np.sum((y == 1) & (p == 1)))
    fp = float(np.sum((y == 0) & (p == 1)))
    fn = float(np.sum((y == 1) & (p == 0)))
    precision = 0.0 if tp + fp <= 0 else tp / (tp + fp)
    recall = 0.0 if tp + fn <= 0 else tp / (tp + fn)
    f1 = 0.0 if precision + recall <= 0 else 2.0 * precision * recall / (precision + recall)
    return float(precision), float(recall), float(f1)


def _binary_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=int).reshape(-1)
    s = np.asarray(score, dtype=float).reshape(-1)
    pos = s[y == 1]
    neg = s[y == 0]
    if pos.size == 0 or neg.size == 0:
        return 0.5
    wins = 0.0
    for value in pos:
        wins += float(np.sum(value > neg)) + 0.5 * float(np.sum(value == neg))
    return float(wins / float(pos.size * neg.size))


def _average_precision(y_true: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=int).reshape(-1)
    s = np.asarray(score, dtype=float).reshape(-1)
    if np.sum(y == 1) == 0:
        return 0.0
    order = np.argsort(-s)
    y_sorted = y[order]
    tp = np.cumsum(y_sorted == 1)
    precision = tp / np.arange(1, y_sorted.shape[0] + 1)
    return float(np.sum(precision[y_sorted == 1]) / float(np.sum(y == 1)))


def _safe_complexity(model: Any) -> float:
    if hasattr(model, "estimators_"):
        try:
            return float(len(model.estimators_))
        except Exception:
            return 1.0
    if hasattr(model, "logit_models"):
        return float(len(tuple(model.logit_models)))
    return 1.0

