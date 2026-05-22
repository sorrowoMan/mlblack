from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from mlblack.backends.contracts import BackendCapabilityContract


class TorchTensorCapability:
    contract = BackendCapabilityContract(
        backend="torch",
        capability="tensor",
        provides=(
            "tensor",
            "tensor.from_numpy",
            "tensor.token_ids",
            "tensor.class_labels",
            "tensor.float_tensor",
            "tensor.device",
            "tensor.to_device",
        ),
        methods={
            "tensor.from_numpy": "as_tensor(value, dtype, device) -> torch.Tensor",
            "tensor.token_ids": "token_ids(value, device) -> torch.LongTensor",
            "tensor.class_labels": "class_labels(value, device) -> torch.LongTensor",
            "tensor.float_tensor": "float_tensor(value, device) -> torch.FloatTensor",
            "tensor.device": "device(context) -> torch.device",
            "tensor.to_device": "to_device(value, device) -> value",
        },
        tensor_kinds=("torch.Tensor",),
        supports_gpu=True,
        notes="Normalizes numpy arrays and ResourceContext device strings into torch tensors/devices.",
    )

    def torch(self) -> Any:
        try:
            import torch
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("torch backend requires optional dependency 'torch'") from exc
        return torch

    def device(self, context: Mapping[str, Any] | None = None, *, fallback: str = "cpu", strict: bool = False) -> Any:
        torch = self.torch()
        ctx = dict(context or {})
        resource = ctx.get("resource_context", ctx.get("resource", {}))
        if isinstance(resource, Mapping):
            raw = ctx.get("resource.device", resource.get("device", fallback))
        else:
            raw = ctx.get("resource.device", fallback)
        device_name = str(raw or fallback)
        if device_name.startswith("cuda") and not torch.cuda.is_available():
            if strict:
                raise RuntimeError(f"requested torch device is not available: {device_name}")
            device_name = "cpu"
        return torch.device(device_name)

    def as_tensor(self, value: Any, *, dtype: Any | None = None, device: Any | None = None) -> Any:
        torch = self.torch()
        target_device = device if device is not None else torch.device("cpu")
        return torch.as_tensor(value, dtype=dtype, device=target_device)

    def token_ids(self, value: np.ndarray, *, device: Any) -> Any:
        torch = self.torch()
        return self.as_tensor(np.asarray(value, dtype=np.int64), dtype=torch.long, device=device)

    def class_labels(self, value: np.ndarray, *, device: Any) -> Any:
        torch = self.torch()
        return self.as_tensor(np.asarray(value, dtype=np.int64).reshape(-1), dtype=torch.long, device=device)

    def float_tensor(self, value: np.ndarray, *, device: Any) -> Any:
        torch = self.torch()
        return self.as_tensor(np.asarray(value, dtype=np.float32), dtype=torch.float32, device=device)

    def to_device(self, value: Any, device: Any) -> Any:
        return value.to(device) if hasattr(value, "to") else value


__all__ = ["TorchTensorCapability"]
