from __future__ import annotations

from typing import Any

import numpy as np

from mlblack.backends.contracts import BackendCapabilityContract


class TensorFlowLossesCapability:
    contract = BackendCapabilityContract(
        backend="tensorflow",
        capability="losses",
        provides=("loss.mse", "metrics.regression"),
        methods={
            "loss.mse": "mse_loss(model_or_prediction, X, y) -> (loss, prediction)",
            "metrics.regression": "regression_metrics(prediction, target, prefix) -> dict",
        },
        tensor_kinds=("tf.Tensor", "np.ndarray"),
        heads=("point", "regression"),
        supports_autograd=True,
        notes="TensorFlow-backed scalar regression losses. Public return values are Python/numpy.",
    )

    def mse_loss(self, model_or_prediction: Any, X: Any, y: Any) -> tuple[float, np.ndarray]:
        if hasattr(model_or_prediction, "predict"):
            prediction = np.asarray(model_or_prediction.predict(np.asarray(X, dtype=float)), dtype=float).reshape(-1)
        else:
            if hasattr(model_or_prediction, "numpy"):
                model_or_prediction = model_or_prediction.numpy()
            prediction = np.asarray(model_or_prediction, dtype=float).reshape(-1)
        target = np.asarray(y, dtype=float).reshape(-1)
        if prediction.shape[0] != target.shape[0]:
            raise ValueError("prediction length differs from target length")
        return float(np.mean((prediction - target) ** 2)), prediction

    def regression_metrics(self, prediction: Any, target: Any, *, prefix: str = "train") -> dict[str, float]:
        if hasattr(prediction, "numpy"):
            prediction = prediction.numpy()
        pred = np.asarray(prediction, dtype=float).reshape(-1)
        y = np.asarray(target, dtype=float).reshape(-1)
        err = pred - y
        mse = float(np.mean(err**2))
        denom = float(np.sum((y - float(np.mean(y))) ** 2))
        r2 = 1.0 if denom <= 0.0 and float(np.sum(err**2)) <= 0.0 else 1.0 - float(np.sum(err**2)) / denom
        return {
            f"{prefix}.mse": mse,
            f"{prefix}.rmse": float(np.sqrt(mse)),
            f"{prefix}.mae": float(np.mean(np.abs(err))),
            f"{prefix}.r2": float(r2),
        }

    def scalar(self, loss: Any) -> float:
        if hasattr(loss, "numpy"):
            loss = loss.numpy()
        return float(np.asarray(loss, dtype=float))


__all__ = ["TensorFlowLossesCapability"]
