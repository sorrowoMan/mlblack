from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from mlblack.backends.contracts import BackendCapabilityContract


class TorchAutogradCapability:
    contract = BackendCapabilityContract(
        backend="torch",
        capability="autograd",
        provides=(
            "autograd.mode.train",
            "autograd.mode.eval",
            "autograd.no_grad",
            "autograd.backward",
            "autograd.zero_grad",
            "autograd.gradients.flat_export",
            "parameters.flat_export",
            "parameters.summary",
            "parameters.state_to_cpu",
            "parameters.state_to_device",
            "parameters.state_json",
        ),
        methods={
            "autograd.mode.train": "train(model, device=None) -> model",
            "autograd.mode.eval": "eval(model, device=None) -> model",
            "autograd.no_grad": "no_grad() -> context manager",
            "autograd.zero_grad": "zero_grad(model) -> None",
            "autograd.backward": "backward(loss) -> None",
            "autograd.gradients.flat_export": "flat_grads(model) -> np.ndarray",
            "parameters.flat_export": "flat_parameters(model) -> np.ndarray",
            "parameters.summary": "parameter_layout_summary(model) -> dict",
        },
        tensor_kinds=("torch.Tensor",),
        model_kinds=("torch.nn.Module",),
        supports_autograd=True,
        supports_stateful_module=True,
        supports_resume=True,
    )

    def __init__(self, tensor: Any) -> None:
        self.tensor = tensor

    def torch(self) -> Any:
        return self.tensor.torch()

    def zero_grad(self, model: Any) -> None:
        if hasattr(model, "zero_grad"):
            model.zero_grad(set_to_none=True)

    def backward(self, loss: Any) -> None:
        loss.backward()

    def train(self, model: Any, *, device: Any | None = None) -> Any:
        if device is not None and hasattr(model, "to"):
            model.to(device)
        if hasattr(model, "train"):
            model.train()
        return model

    def eval(self, model: Any, *, device: Any | None = None) -> Any:
        if device is not None and hasattr(model, "to"):
            model.to(device)
        if hasattr(model, "eval"):
            model.eval()
        return model

    def no_grad(self) -> Any:
        return self.torch().no_grad()

    def grad_norm(self, params: Sequence[Any]) -> float:
        torch = self.torch()
        total = torch.zeros((), dtype=torch.float32, device=params[0].device if params else "cpu")
        for param in params:
            if param.grad is not None:
                total = total + torch.sum(param.grad.detach() ** 2)
        return float(torch.sqrt(total).detach().cpu().item())

    def clip_grad_norm(self, params: Sequence[Any], max_norm: float) -> float:
        torch = self.torch()
        value = torch.nn.utils.clip_grad_norm_(params, float(max_norm))
        return float(value.detach().cpu().item() if hasattr(value, "detach") else value)

    def trainable_parameters(self, model: Any) -> list[Any]:
        if not hasattr(model, "parameters"):
            raise TypeError("torch backend autograd requires a torch-like module with parameters()")
        return [param for param in model.parameters() if param.requires_grad]

    def flat_parameters(self, model: Any) -> np.ndarray:
        rows = [param.detach().cpu().numpy().reshape(-1).astype(float) for param in model.parameters()]
        return np.concatenate(rows) if rows else np.zeros(0, dtype=float)

    def flat_grads(self, model: Any) -> np.ndarray:
        rows: list[np.ndarray] = []
        for _name, param in model.named_parameters():
            grad = param.grad
            if grad is None:
                rows.append(np.zeros(int(param.numel()), dtype=float))
            else:
                rows.append(grad.detach().cpu().numpy().reshape(-1).astype(float))
        return np.concatenate(rows) if rows else np.zeros(0, dtype=float)

    def parameter_layout_summary(self, model: Any) -> dict[str, Any]:
        names: list[str] = []
        shapes: list[tuple[int, ...]] = []
        total = 0
        for name, param in model.named_parameters():
            shape = tuple(int(v) for v in param.detach().cpu().shape)
            names.append(str(name))
            shapes.append(shape)
            total += int(param.numel())
        return {"names": tuple(names), "shapes": tuple(shapes), "total_size": int(total)}

    def optimizer_state_to_cpu(self, value: Any) -> Any:
        torch = self.torch()
        if torch.is_tensor(value):
            return value.detach().cpu().clone()
        if isinstance(value, Mapping):
            return {key: self.optimizer_state_to_cpu(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self.optimizer_state_to_cpu(item) for item in value]
        if isinstance(value, tuple):
            return tuple(self.optimizer_state_to_cpu(item) for item in value)
        return value

    def optimizer_state_to_device(self, value: Any, device: Any) -> Any:
        torch = self.torch()
        if torch.is_tensor(value):
            return value.to(device)
        if isinstance(value, Mapping):
            return {_restore_key(key): self.optimizer_state_to_device(item, device) for key, item in value.items()}
        if isinstance(value, list):
            return [self.optimizer_state_to_device(item, device) for item in value]
        if isinstance(value, tuple):
            return tuple(self.optimizer_state_to_device(item, device) for item in value)
        return value

    def jsonable_torch_state(self, value: Any) -> Any:
        torch = self.torch()
        if torch.is_tensor(value):
            return {
                "__tensor__": True,
                "dtype": str(value.dtype),
                "shape": list(value.shape),
                "data": value.detach().cpu().tolist(),
            }
        if hasattr(value, "tolist"):
            return value.tolist()
        if isinstance(value, Mapping):
            return {str(key): self.jsonable_torch_state(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [self.jsonable_torch_state(item) for item in value]
        return value

    def restore_jsonable_torch_state(self, value: Any) -> Any:
        torch = self.torch()
        if isinstance(value, Mapping) and value.get("__tensor__") is True:
            dtype_name = str(value.get("dtype", "torch.float32")).replace("torch.", "")
            dtype = getattr(torch, dtype_name, torch.float32)
            return torch.tensor(value.get("data", []), dtype=dtype).reshape(tuple(int(v) for v in value.get("shape", ())))
        if isinstance(value, Mapping):
            return {_restore_key(key): self.restore_jsonable_torch_state(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self.restore_jsonable_torch_state(item) for item in value]
        return value


def _restore_key(key: Any) -> Any:
    if isinstance(key, str) and key.isdigit():
        return int(key)
    return key


__all__ = ["TorchAutogradCapability"]
