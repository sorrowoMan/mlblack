from __future__ import annotations

from typing import Any

import numpy as np

from mlblack.backends.contracts import BackendCapabilityContract


class TensorFlowOptimizersCapability:
    contract = BackendCapabilityContract(
        backend="tensorflow",
        capability="optimizers",
        provides=("optimizer.sgd_step",),
        methods={
            "optimizer.sgd_step": "sgd_step(values, gradients, learning_rate) -> np.ndarray",
        },
        tensor_kinds=("tf.Tensor", "np.ndarray"),
        supports_functional_params=True,
        notes="Minimal TensorFlow-compatible functional SGD helper.",
    )

    def sgd_step(self, values: Any, gradients: Any, *, learning_rate: float) -> np.ndarray:
        if hasattr(values, "numpy"):
            values = values.numpy()
        if hasattr(gradients, "numpy"):
            gradients = gradients.numpy()
        return np.asarray(values, dtype=float).reshape(-1) - (float(learning_rate) * np.asarray(gradients, dtype=float).reshape(-1))


__all__ = ["TensorFlowOptimizersCapability"]
