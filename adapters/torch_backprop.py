from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from mlblack.core.adapter import OptimizerAdapter
from mlblack.core.contracts import ComponentContract
from mlblack.core.resources import ResourceContext
from mlblack.core.types import Feedback, UnknownState


@dataclass(frozen=True)
class TorchBackpropConfig:
    optimizer: str = "sgd"
    learning_rate: float = 1e-3
    min_learning_rate: float = 1e-9
    weight_decay: float = 0.0
    max_grad_norm: float | None = 10.0
    batch_size: int | None = None
    shuffle: bool = True
    drop_last: bool = False
    device: str = "cpu"
    device_policy: str = "fallback_cpu"  # fallback_cpu | strict
    loss: str = "mse"
    train_mode: bool = True
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    random_seed: int = 42


class TorchBackpropAdapter(OptimizerAdapter):
    """Torch engine adapter for parameter-vector neural representations.

    It updates UnknownState values by building a torch computation graph from
    the representation metadata. The representation still owns decoding; the
    adapter only owns parameter optimization.
    """

    name = "torch_backprop"
    context_requires = ('candidate.unknown_state', 'representation.numpy_mlp_point', 'problem.data.X_train', 'problem.data.y_train')
    context_optional = ('resource.device', 'feedback.objectives', 'feedback.metrics')
    context_provides = ('population.candidates', 'feedback.gradients')
    context_mutates = ('adapter.current_state',)
    context_cache = ()
    requires_metrics = ('objective',)
    metrics_fallback = "strict"
    context_notes = 'Reads context fields: candidate.unknown_state, representation.numpy_mlp_point, problem.data.X_train, problem.data.y_train; provides population.candidates, feedback.gradients; mutates adapter.current_state.'
    contract = ComponentContract(
        name=name,
        requires=(
            "candidate.unknown_state",
            "representation.numpy_mlp_point",
            "problem.data.X_train",
            "problem.data.y_train",
        ),
        optional=("resource.device", "feedback.objectives", "feedback.metrics"),
        provides=("population.candidates", "feedback.gradients"),
        mutates=("adapter.current_state",),
        supports_gradient=True,
        supports_batch=False,
        supports_resume=True,
        metadata={"family": "neural", "engine": "torch", "head": "point"},
    )

    def __init__(self, config: TorchBackpropConfig | None = None, **kwargs: Any) -> None:
        if config is not None and kwargs:
            raise ValueError("provide config or kwargs, not both")
        self.config = config or TorchBackpropConfig(**kwargs)
        self.current_state: UnknownState | None = None
        self.last_loss: float | None = None
        self.last_gradient_norm: float | None = None
        self.step_index = 0
        self.epoch_index = 0
        self.batch_cursor = 0
        self.last_batch_indices: tuple[int, ...] = tuple()
        self.actual_device = str(self.config.device)
        self.optimizer_state: dict[str, Any] = {}
        self._rng = np.random.default_rng(int(self.config.random_seed))

    def setup(self, trainer: Any) -> None:
        _ = trainer
        if self.current_state is None:
            self.last_loss = None
            self.last_gradient_norm = None
            self.step_index = 0
            self.epoch_index = 0
            self.batch_cursor = 0
            self.last_batch_indices = tuple()
            self.optimizer_state = {}

    def propose(self, trainer: Any, context: Mapping[str, Any]) -> Sequence[UnknownState]:
        if self.current_state is None:
            self.current_state = trainer.init_candidate(context)
        return (self.current_state,)

    def update(
        self,
        trainer: Any,
        states: Sequence[UnknownState],
        feedback: Sequence[Feedback],
        context: Mapping[str, Any],
    ) -> None:
        _ = feedback
        if not states:
            return
        state = states[0]
        grad, loss = self._compute_gradient(trainer, state, context)
        values = state.as_array()
        if grad.shape[0] != values.shape[0]:
            raise ValueError(f"gradient dimension {grad.shape[0]} does not match state dimension {values.shape[0]}")
        norm = float(np.linalg.norm(grad))
        self.last_gradient_norm = norm
        self.last_loss = float(loss)
        if self.config.max_grad_norm is not None and norm > float(self.config.max_grad_norm) and norm > 0.0:
            grad = grad * (float(self.config.max_grad_norm) / norm)
        lr = max(float(self.config.min_learning_rate), float(self.config.learning_rate))
        update = self._optimizer_update(grad, lr)
        self.current_state = state.with_values(
            values - update,
            adapter=self.name,
            learning_rate=lr,
            gradient_norm=norm,
            torch_loss=float(loss),
            optimizer=str(self.config.optimizer),
            epoch_index=int(self.epoch_index),
            batch_cursor=int(self.batch_cursor),
        )
        self.step_index += 1

    def get_state(self) -> Mapping[str, Any]:
        return {
            "current_state": None if self.current_state is None else self.current_state.as_array().tolist(),
            "last_loss": self.last_loss,
            "last_gradient_norm": self.last_gradient_norm,
            "step_index": int(self.step_index),
            "epoch_index": int(self.epoch_index),
            "batch_cursor": int(self.batch_cursor),
            "last_batch_indices": list(self.last_batch_indices),
            "optimizer": str(self.config.optimizer),
            "optimizer_state": _jsonable_optimizer_state(self.optimizer_state),
            "learning_rate": float(self.config.learning_rate),
            "device": str(self.config.device),
            "actual_device": str(self.actual_device),
            "device_policy": str(self.config.device_policy),
        }

    def set_state(self, state: Mapping[str, Any]) -> None:
        values = state.get("current_state")
        self.current_state = None if values is None else UnknownState(values=np.asarray(values, dtype=float))
        loss = state.get("last_loss")
        self.last_loss = None if loss is None else float(loss)
        grad_norm = state.get("last_gradient_norm")
        self.last_gradient_norm = None if grad_norm is None else float(grad_norm)
        self.step_index = int(state.get("step_index", self.step_index))
        self.epoch_index = int(state.get("epoch_index", self.epoch_index))
        self.batch_cursor = int(state.get("batch_cursor", self.batch_cursor))
        self.last_batch_indices = tuple(int(v) for v in state.get("last_batch_indices", ()))
        opt_state = state.get("optimizer_state")
        if isinstance(opt_state, Mapping):
            self.optimizer_state = _restore_optimizer_state(opt_state)
        self.actual_device = str(state.get("actual_device", self.actual_device))

    def _optimizer_update(self, grad: np.ndarray, learning_rate: float) -> np.ndarray:
        key = str(self.config.optimizer or "sgd").strip().lower()
        if key in {"sgd", "gd", "gradient_descent"}:
            return float(learning_rate) * grad
        if key in {"adam", "adamw"}:
            t = int(self.optimizer_state.get("t", 0)) + 1
            m = np.asarray(self.optimizer_state.get("m", np.zeros_like(grad)), dtype=float)
            v = np.asarray(self.optimizer_state.get("v", np.zeros_like(grad)), dtype=float)
            beta1 = float(self.config.beta1)
            beta2 = float(self.config.beta2)
            m = (beta1 * m) + ((1.0 - beta1) * grad)
            v = (beta2 * v) + ((1.0 - beta2) * (grad ** 2))
            m_hat = m / (1.0 - beta1 ** t)
            v_hat = v / (1.0 - beta2 ** t)
            update = float(learning_rate) * m_hat / (np.sqrt(v_hat) + float(self.config.eps))
            if key == "adamw" and float(self.config.weight_decay) > 0.0 and self.current_state is not None:
                update = update + (float(learning_rate) * float(self.config.weight_decay) * self.current_state.as_array())
            self.optimizer_state = {"t": t, "m": m, "v": v}
            return update
        raise ValueError(f"unsupported torch optimizer: {self.config.optimizer}")

    def _compute_gradient(
        self,
        trainer: Any,
        state: UnknownState,
        context: Mapping[str, Any],
    ) -> tuple[np.ndarray, float]:
        try:
            import torch
        except Exception as exc:
            raise RuntimeError("TorchBackpropAdapter requires the optional dependency 'torch'") from exc
        torch.manual_seed(int(self.config.random_seed) + int(self.step_index))

        problem = getattr(trainer, "problem", None)
        data = getattr(problem, "data", None)
        if data is None:
            raise ValueError("TorchBackpropAdapter requires trainer.problem.data")

        representation = getattr(trainer, "representation_pipeline", None)
        config = getattr(representation, "config", None)
        shapes = getattr(representation, "shapes", None)
        if config is None or shapes is None:
            raise TypeError("TorchBackpropAdapter currently requires NumpyMLPPointRepresentation")

        resource_context = ResourceContext.from_mapping(context.get("resource_context", context.get("resource", {})))
        device_name = str(context.get("resource.device", resource_context.device or self.config.device))
        if device_name.startswith("cuda") and not torch.cuda.is_available():
            if str(self.config.device_policy).lower() == "strict":
                raise RuntimeError(f"requested torch device is not available: {device_name}")
            device_name = "cpu"
        self.actual_device = device_name
        device = torch.device(device_name)
        X_np = np.asarray(data.X_train, dtype=np.float32)
        y_np = np.asarray(data.y_train, dtype=np.float32).reshape(-1, 1)
        idx = self._next_batch_indices(X_np.shape[0])
        if idx is not None:
            X_np = X_np[idx]
            y_np = y_np[idx]

        X = torch.as_tensor(X_np, dtype=torch.float32, device=device)
        y = torch.as_tensor(y_np, dtype=torch.float32, device=device)
        flat = np.asarray(state.values, dtype=np.float32).reshape(-1)
        params = _torch_split_parameters(torch, flat, tuple(shapes), device)
        out = X
        for idx in range(0, len(params), 2):
            weight = params[idx]
            bias = params[idx + 1]
            out = out @ weight + bias
            if idx < len(params) - 2:
                out = _torch_activate(torch, out, getattr(config, "activation", "relu"))
                dropout = float(getattr(config, "dropout", 0.0) or 0.0)
                if dropout > 0.0 and bool(self.config.train_mode):
                    out = torch.nn.functional.dropout(out, p=min(dropout, 0.95), training=True)

        loss_key = str(self.config.loss or "mse").strip().lower()
        if loss_key != "mse":
            raise ValueError(f"unsupported torch loss: {self.config.loss}")
        loss = torch.mean((out.reshape_as(y) - y) ** 2)
        if float(self.config.weight_decay) > 0.0:
            penalty = torch.zeros((), dtype=torch.float32, device=device)
            for idx in range(0, len(params), 2):
                penalty = penalty + torch.sum(params[idx] ** 2)
            loss = loss + (float(self.config.weight_decay) * penalty)
        loss.backward()
        grads = [param.grad.detach().cpu().numpy().reshape(-1) for param in params]
        grad = np.concatenate(grads).astype(float)
        return grad, float(loss.detach().cpu().item())

    def _next_batch_indices(self, n_rows: int) -> np.ndarray | None:
        if self.config.batch_size is None or int(self.config.batch_size) <= 0 or int(self.config.batch_size) >= int(n_rows):
            self.last_batch_indices = tuple(range(int(n_rows)))
            return None
        batch_size = max(1, int(self.config.batch_size))
        if bool(self.config.shuffle):
            idx = self._rng.choice(int(n_rows), size=batch_size, replace=False)
            self.last_batch_indices = tuple(int(v) for v in idx)
            return idx
        start = int(self.batch_cursor)
        stop = start + batch_size
        if stop > int(n_rows):
            self.epoch_index += 1
            if bool(self.config.drop_last):
                start = 0
                stop = batch_size
            else:
                stop = int(n_rows)
        idx = np.arange(start, stop, dtype=int)
        self.batch_cursor = 0 if stop >= int(n_rows) else stop
        self.last_batch_indices = tuple(int(v) for v in idx)
        return idx


def _torch_split_parameters(torch: Any, values: np.ndarray, shapes: tuple[tuple[int, ...], ...], device: Any) -> list[Any]:
    expected = int(sum(np.prod(shape) for shape in shapes))
    if values.shape[0] != expected:
        raise ValueError(f"parameter vector has {values.shape[0]} values but representation expects {expected}")
    params: list[Any] = []
    offset = 0
    for shape in shapes:
        size = int(np.prod(shape))
        block = values[offset : offset + size].reshape(shape)
        offset += size
        params.append(torch.tensor(block, dtype=torch.float32, device=device, requires_grad=True))
    return params


def _torch_activate(torch: Any, x: Any, activation: str) -> Any:
    key = str(activation or "relu").strip().lower()
    if key == "relu":
        return torch.relu(x)
    if key == "tanh":
        return torch.tanh(x)
    if key in {"sigmoid", "logistic"}:
        return torch.sigmoid(x)
    if key in {"identity", "linear", "none"}:
        return x
    raise ValueError(f"unsupported activation: {activation}")


def _jsonable_optimizer_state(state: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in dict(state).items():
        if hasattr(value, "tolist"):
            out[str(key)] = value.tolist()
        else:
            out[str(key)] = value
    return out


def _restore_optimizer_state(state: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in dict(state).items():
        if key in {"m", "v"}:
            out[key] = np.asarray(value, dtype=float)
        else:
            out[key] = value
    return out


