from __future__ import annotations

from typing import Any

from mlblack.backends.contracts import BackendCapabilityContract


class TorchOptimizersCapability:
    contract = BackendCapabilityContract(
        backend="torch",
        capability="optimizers",
        provides=("optimizer.build", "optimizer.step", "optimizer.zero_grad", "optimizer.state"),
        methods={
            "optimizer.build": "build_optimizer(model, config) -> torch.optim.Optimizer",
            "optimizer.step": "step(optimizer) -> None",
            "optimizer.zero_grad": "zero_grad(optimizer) -> None",
        },
        model_kinds=("torch.nn.Module",),
        supports_stateful_module=True,
        supports_resume=True,
    )

    def __init__(self, tensor: Any, autograd: Any) -> None:
        self.tensor = tensor
        self.autograd = autograd

    def torch(self) -> Any:
        return self.tensor.torch()

    def build_optimizer(self, model: Any, config: Any) -> Any:
        torch = self.torch()
        lr = max(float(getattr(config, "min_learning_rate", 0.0)), float(getattr(config, "learning_rate", 1e-3)))
        weight_decay = float(getattr(config, "weight_decay", 0.0))
        key = str(getattr(config, "optimizer", "adamw") or "adamw").strip().lower()
        params = self.autograd.trainable_parameters(model)
        if key in {"sgd", "gd", "gradient_descent"}:
            return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay)
        if key == "adam":
            return torch.optim.Adam(
                params,
                lr=lr,
                weight_decay=weight_decay,
                betas=(float(getattr(config, "beta1", 0.9)), float(getattr(config, "beta2", 0.999))),
                eps=float(getattr(config, "eps", 1e-8)),
            )
        if key == "adamw":
            return torch.optim.AdamW(
                params,
                lr=lr,
                weight_decay=weight_decay,
                betas=(float(getattr(config, "beta1", 0.9)), float(getattr(config, "beta2", 0.999))),
                eps=float(getattr(config, "eps", 1e-8)),
            )
        raise ValueError(f"unsupported torch optimizer: {getattr(config, 'optimizer', key)}")

    def zero_grad(self, optimizer: Any) -> None:
        optimizer.zero_grad(set_to_none=True)

    def step(self, optimizer: Any) -> None:
        optimizer.step()


__all__ = ["TorchOptimizersCapability"]
