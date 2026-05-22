from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from mlblack.backends.contracts import BackendCapabilityContract


class NumpyTensorCapability:
    contract = BackendCapabilityContract(
        backend="numpy",
        capability="tensor",
        provides=(
            "tensor",
            "tensor.from_numpy",
            "tensor.float_tensor",
            "tensor.device",
            "tensor.to_device",
        ),
        methods={
            "tensor.from_numpy": "as_tensor(value, dtype, device) -> np.ndarray",
            "tensor.float_tensor": "float_tensor(value, device) -> np.ndarray",
            "tensor.device": "device(context) -> 'cpu'",
            "tensor.to_device": "to_device(value, device) -> value",
        },
        tensor_kinds=("np.ndarray",),
        supports_gpu=False,
        notes="CPU ndarray tensor normalization. No autograd graph or device transfer semantics.",
    )

    def device(self, context: Mapping[str, Any] | None = None, *, fallback: str = "cpu", strict: bool = False) -> str:
        _ = context
        requested = str(fallback or "cpu").strip().lower()
        if strict and requested not in {"", "cpu"}:
            raise RuntimeError(f"numpy backend only supports cpu device, got: {requested!r}")
        return "cpu"

    def as_tensor(self, value: Any, *, dtype: Any | None = None, device: Any | None = None) -> np.ndarray:
        _ = device
        return np.asarray(value, dtype=dtype)

    def float_tensor(self, value: Any, *, device: Any | None = None) -> np.ndarray:
        _ = device
        return np.asarray(value, dtype=float)

    def to_device(self, value: Any, device: Any) -> Any:
        _ = device
        return value


__all__ = ["NumpyTensorCapability"]
