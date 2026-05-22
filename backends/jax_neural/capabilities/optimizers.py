from __future__ import annotations

from typing import Any

import numpy as np

from mlblack.backends.contracts import BackendCapabilityContract


class JaxOptimizersCapability:
    contract = BackendCapabilityContract(
        backend="jax",
        capability="optimizers",
        provides=("optimizer.sgd_step",),
        methods={
            "optimizer.sgd_step": "sgd_step(values, gradients, learning_rate) -> np.ndarray",
        },
        tensor_kinds=("jax.Array", "np.ndarray"),
        supports_functional_params=True,
        notes="Minimal functional optimizer helpers. Stateful optax-style optimizers can be added later.",
    )

    def sgd_step(self, values: Any, gradients: Any, *, learning_rate: float) -> np.ndarray:
        return np.asarray(values, dtype=float).reshape(-1) - (float(learning_rate) * np.asarray(gradients, dtype=float).reshape(-1))


__all__ = ["JaxOptimizersCapability"]
