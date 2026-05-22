from __future__ import annotations

from typing import Any

import numpy as np

from mlblack.backends.contracts import BackendCapabilityContract


class JaxAutogradCapability:
    contract = BackendCapabilityContract(
        backend="jax",
        capability="autograd",
        provides=("autograd.functional.grad", "autograd.value_and_grad", "autograd.gradients.flat_export"),
        methods={
            "autograd.functional.grad": "mse_parameter_gradient(model, X, y, l2) -> np.ndarray",
            "autograd.value_and_grad": "value_and_grad(fn, values) -> (value, gradient)",
            "autograd.gradients.flat_export": "flat_gradient(gradient) -> np.ndarray",
        },
        tensor_kinds=("jax.Array",),
        model_kinds=("JaxMLPPointModel",),
        supports_autograd=True,
        supports_functional_params=True,
        notes="Functional JAX gradient surface. It intentionally does not expose torch-style backward().",
    )

    def mse_parameter_gradient(self, model: Any, X: Any, y: Any, *, l2: float = 0.0) -> np.ndarray:
        values = getattr(model, "values", None)
        predict_from_values = getattr(model, "predict_from_values", None)
        if values is None or not callable(predict_from_values):
            raise TypeError("jax functional autograd requires a model exposing values and predict_from_values(values, X)")
        jax = _jax()
        jnp = _jnp()
        x_arr = jnp.asarray(X, dtype=jnp.float32)
        y_arr = jnp.asarray(y, dtype=jnp.float32).reshape(-1)

        def loss_fn(candidate_values: Any) -> Any:
            pred = predict_from_values(candidate_values, x_arr).reshape(-1)
            mse = jnp.mean((pred - y_arr) ** 2)
            if float(l2) <= 0.0:
                return mse
            return mse + (float(l2) * jnp.sum(candidate_values**2))

        grad = jax.grad(loss_fn)(jnp.asarray(values, dtype=jnp.float32))
        return np.asarray(grad, dtype=float).reshape(-1)

    def value_and_grad(self, fn: Any, values: Any) -> tuple[Any, np.ndarray]:
        jax = _jax()
        value, grad = jax.value_and_grad(fn)(values)
        return value, np.asarray(grad, dtype=float).reshape(-1)

    def flat_gradient(self, gradient: Any) -> np.ndarray:
        return np.asarray(gradient, dtype=float).reshape(-1)


def _jax() -> Any:
    try:
        import jax
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("jax backend requires optional dependency 'jax'") from exc
    return jax


def _jnp() -> Any:
    try:
        import jax.numpy as jnp
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("jax backend requires optional dependency 'jax'") from exc
    return jnp


__all__ = ["JaxAutogradCapability"]
