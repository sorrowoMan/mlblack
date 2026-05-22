from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class BinaryLogisticProbabilityModel:
    """Probability wrapper around a scalar logit model."""

    logit_model: Any
    temperature: float = 1.0
    threshold: float = 0.5
    classes_: Sequence[Any] = (0, 1)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        logits = np.asarray(self.logit_model.predict(X), dtype=float).reshape(-1)
        return logits / max(float(self.temperature), 1e-12)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        logits = self.decision_function(X)
        p1 = 1.0 / (1.0 + np.exp(-np.clip(logits, -60.0, 60.0)))
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X: np.ndarray) -> np.ndarray:
        idx = (self.predict_proba(X)[:, 1] >= float(self.threshold)).astype(int)
        classes = tuple(self.classes_)
        if len(classes) >= 2:
            return np.asarray([classes[i] for i in idx])
        return idx

    def describe(self) -> dict[str, Any]:
        return {
            "name": "binary_logistic_probability_model",
            "temperature": float(self.temperature),
            "threshold": float(self.threshold),
            "classes": list(self.classes_),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class SoftmaxProbabilityModel:
    """Probability wrapper around one scalar logit model per class."""

    logit_models: Sequence[Any]
    temperature: float = 1.0
    classes_: Sequence[Any] = tuple()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        logits = [np.asarray(model.predict(X), dtype=float).reshape(-1) for model in tuple(self.logit_models)]
        if not logits:
            raise ValueError("SoftmaxProbabilityModel requires at least one logit model")
        return np.column_stack(logits) / max(float(self.temperature), 1e-12)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        logits = self.decision_function(X)
        logits = logits - np.max(logits, axis=1, keepdims=True)
        exp = np.exp(np.clip(logits, -60.0, 60.0))
        return exp / np.sum(exp, axis=1, keepdims=True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        idx = np.argmax(self.predict_proba(X), axis=1)
        classes = tuple(self.classes_) or tuple(range(len(tuple(self.logit_models))))
        return np.asarray([classes[int(i)] for i in idx])

    def describe(self) -> dict[str, Any]:
        return {
            "name": "softmax_probability_model",
            "num_classes": len(tuple(self.logit_models)),
            "temperature": float(self.temperature),
            "classes": list(self.classes_),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class TemperatureCalibratedProbabilityModel:
    """Temperature calibration wrapper for probability-capable models."""

    base_model: Any
    temperature: float = 1.0
    clip_eps: float = 1e-12
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self.base_model, "predict_proba"):
            base = BinaryLogisticProbabilityModel(self.base_model)
            proba = base.predict_proba(X)
        else:
            proba = np.asarray(self.base_model.predict_proba(X), dtype=float)
        proba = np.clip(proba, float(self.clip_eps), 1.0)
        logits = np.log(proba)
        logits = logits / max(float(self.temperature), float(self.clip_eps))
        logits = logits - np.max(logits, axis=1, keepdims=True)
        exp = np.exp(np.clip(logits, -60.0, 60.0))
        return exp / np.sum(exp, axis=1, keepdims=True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        idx = np.argmax(self.predict_proba(X), axis=1)
        classes = getattr(self.base_model, "classes_", tuple(range(self.predict_proba(X).shape[1])))
        return np.asarray([classes[int(i)] for i in idx])

    def describe(self) -> dict[str, Any]:
        return {
            "name": "temperature_calibrated_probability_model",
            "temperature": float(self.temperature),
            "metadata": dict(self.metadata),
        }
