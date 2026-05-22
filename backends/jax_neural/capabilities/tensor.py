from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from mlblack.backends.contracts import BackendCapabilityContract


class JaxTensorCapability:
    contract = BackendCapabilityContract(
        backend="jax",
        capability="tensor",
        provides=(
            "tensor",
            "tensor.from_numpy",
            "tensor.float_tensor",
            "tensor.device",
            "tensor.to_device",
        ),
        methods={
            "tensor.from_numpy": "as_tensor(value, dtype, device) -> jax.Array",
            "tensor.float_tensor": "float_tensor(value, device) -> jax.Array",
            "tensor.device": "device(context) -> str",
            "tensor.to_device": "to_device(value, device) -> value",
        },
        tensor_kinds=("jax.Array",),
        supports_gpu=True,
        notes="JAX array normalization. Device placement is intentionally minimal until a full L0 device bridge is added.",
    )

    def jax(self) -> Any:
        try:
            import jax
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("jax backend requires optional dependency 'jax'") from exc
        return jax

    def jnp(self) -> Any:
        try:
            import jax.numpy as jnp
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("jax backend requires optional dependency 'jax'") from exc
        return jnp

    def device(self, context: Mapping[str, Any] | None = None, *, fallback: str = "cpu", strict: bool = False) -> str:
        ctx = dict(context or {})
        resource = ctx.get("resource_context", ctx.get("resource", {}))
        raw = resource.get("device", fallback) if isinstance(resource, Mapping) else fallback
        requested = str(ctx.get("resource.device", raw or fallback) or "cpu").strip().lower()
        if requested.startswith("cuda"):
            requested = requested.replace("cuda", "gpu", 1)
        if strict and requested not in {"cpu", "gpu", "tpu"}:
            raise RuntimeError(f"unsupported jax device request: {requested!r}")
        return requested if requested in {"cpu", "gpu", "tpu"} else "cpu"

    def as_tensor(self, value: Any, *, dtype: Any | None = None, device: Any | None = None) -> Any:
        _ = device
        return self.jnp().asarray(value, dtype=dtype)

    def float_tensor(self, value: Any, *, device: Any | None = None) -> Any:
        _ = device
        return self.jnp().asarray(np.asarray(value, dtype=np.float32))

    def to_device(self, value: Any, device: Any) -> Any:
        _ = device
        return value


__all__ = ["JaxTensorCapability"]
