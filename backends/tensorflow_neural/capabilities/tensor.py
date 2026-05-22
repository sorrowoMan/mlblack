from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from mlblack.backends.contracts import BackendCapabilityContract


class TensorFlowTensorCapability:
    contract = BackendCapabilityContract(
        backend="tensorflow",
        capability="tensor",
        provides=(
            "tensor",
            "tensor.from_numpy",
            "tensor.float_tensor",
            "tensor.device",
            "tensor.to_device",
        ),
        methods={
            "tensor.from_numpy": "as_tensor(value, dtype, device) -> tf.Tensor",
            "tensor.float_tensor": "float_tensor(value, device) -> tf.Tensor",
            "tensor.device": "device(context) -> str",
            "tensor.to_device": "to_device(value, device) -> value",
        },
        tensor_kinds=("tf.Tensor",),
        supports_gpu=True,
        notes="TensorFlow tensor normalization. Device placement remains L0-owned.",
    )

    def tf(self) -> Any:
        return _tf()

    def device(self, context: Mapping[str, Any] | None = None, *, fallback: str = "cpu", strict: bool = False) -> str:
        ctx = dict(context or {})
        resource = ctx.get("resource_context", ctx.get("resource", {}))
        raw = resource.get("device", fallback) if isinstance(resource, Mapping) else fallback
        requested = str(ctx.get("resource.device", raw or fallback) or "cpu").strip().lower()
        if requested.startswith("cuda"):
            requested = requested.replace("cuda", "gpu", 1)
        if strict and requested not in {"cpu", "gpu"}:
            raise RuntimeError(f"unsupported tensorflow device request: {requested!r}")
        return requested if requested in {"cpu", "gpu"} else "cpu"

    def as_tensor(self, value: Any, *, dtype: Any | None = None, device: Any | None = None) -> Any:
        _ = device
        tf = _tf()
        return tf.convert_to_tensor(value, dtype=dtype)

    def float_tensor(self, value: Any, *, device: Any | None = None) -> Any:
        _ = device
        tf = _tf()
        return tf.convert_to_tensor(np.asarray(value, dtype=np.float32), dtype=tf.float32)

    def to_device(self, value: Any, device: Any) -> Any:
        _ = device
        return value


def _tf() -> Any:
    try:
        import tensorflow as tf
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("tensorflow backend requires optional dependency 'tensorflow'") from exc
    return tf


__all__ = ["TensorFlowTensorCapability"]
