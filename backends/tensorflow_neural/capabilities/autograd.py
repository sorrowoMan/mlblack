from __future__ import annotations

from typing import Any

import numpy as np

from mlblack.backends.contracts import BackendCapabilityContract


class TensorFlowAutogradCapability:
    contract = BackendCapabilityContract(
        backend="tensorflow",
        capability="autograd",
        provides=("autograd.functional.grad", "autograd.value_and_grad", "autograd.gradients.flat_export"),
        methods={
            "autograd.functional.grad": "mse_parameter_gradient(model, X, y, l2) -> np.ndarray",
            "autograd.value_and_grad": "value_and_grad(fn, values) -> (value, gradient)",
            "autograd.gradients.flat_export": "flat_gradient(gradient) -> np.ndarray",
        },
        tensor_kinds=("tf.Tensor",),
        model_kinds=("TensorFlowMLPPointModel",),
        supports_autograd=True,
        supports_functional_params=True,
        notes="TensorFlow GradientTape functional gradient surface. No torch-style backward() contract is exposed.",
    )

    def mse_parameter_gradient(self, model: Any, X: Any, y: Any, *, l2: float = 0.0) -> np.ndarray:
        values = getattr(model, "values", None)
        predict_from_values = getattr(model, "predict_from_values", None)
        if values is None or not callable(predict_from_values):
            raise TypeError("tensorflow functional autograd requires a model exposing values and predict_from_values(values, X)")
        tf = _tf()
        x_arr = tf.convert_to_tensor(np.asarray(X, dtype=np.float32), dtype=tf.float32)
        y_arr = tf.reshape(tf.convert_to_tensor(np.asarray(y, dtype=np.float32), dtype=tf.float32), (-1,))
        values_tensor = tf.convert_to_tensor(np.asarray(values, dtype=np.float32), dtype=tf.float32)

        with tf.GradientTape() as tape:
            tape.watch(values_tensor)
            pred = tf.reshape(predict_from_values(values_tensor, x_arr), (-1,))
            loss = tf.reduce_mean(tf.square(pred - y_arr))
            if float(l2) > 0.0:
                loss = loss + (float(l2) * tf.reduce_sum(tf.square(values_tensor)))
        gradient = tape.gradient(loss, values_tensor)
        if gradient is None:
            raise RuntimeError("TensorFlow GradientTape returned no gradient for functional parameters")
        return np.asarray(gradient.numpy(), dtype=float).reshape(-1)

    def value_and_grad(self, fn: Any, values: Any) -> tuple[Any, np.ndarray]:
        tf = _tf()
        values_tensor = tf.convert_to_tensor(np.asarray(values, dtype=np.float32), dtype=tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(values_tensor)
            value = fn(values_tensor)
        gradient = tape.gradient(value, values_tensor)
        if gradient is None:
            raise RuntimeError("TensorFlow GradientTape returned no gradient")
        scalar = value.numpy() if hasattr(value, "numpy") else value
        return scalar, np.asarray(gradient.numpy(), dtype=float).reshape(-1)

    def flat_gradient(self, gradient: Any) -> np.ndarray:
        if hasattr(gradient, "numpy"):
            gradient = gradient.numpy()
        return np.asarray(gradient, dtype=float).reshape(-1)


def _tf() -> Any:
    try:
        import tensorflow as tf
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("tensorflow backend requires optional dependency 'tensorflow'") from exc
    return tf


__all__ = ["TensorFlowAutogradCapability"]
