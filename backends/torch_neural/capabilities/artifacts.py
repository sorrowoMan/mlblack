from __future__ import annotations

from typing import Any

import numpy as np

from mlblack.backends.contracts import BackendCapabilityContract


class TorchArtifactsCapability:
    contract = BackendCapabilityContract(
        backend="torch",
        capability="artifacts",
        provides=("artifact.neural_graph.audit", "artifact.parameters.summary", "artifact.torch_model.describe"),
        methods={
            "artifact.neural_graph.audit": "audit_summary(model, data, max_rows) -> dict",
            "artifact.parameters.summary": "parameter_layout_summary(model) -> dict",
        },
        model_kinds=("torch.nn.Module",),
        supports_stateful_module=True,
    )

    def __init__(self, tensor: Any, autograd: Any) -> None:
        self.tensor = tensor
        self.autograd = autograd

    def audit_summary(self, model: Any, data: Any, *, max_rows: int = 4) -> dict[str, Any]:
        torch = self.tensor.torch()
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
        tokens = self.tensor.token_ids(data.X_train[: max(1, min(int(max_rows), int(data.X_train.shape[0])))], device=device)
        was_training = bool(getattr(model, "training", False))
        self.autograd.eval(model)
        with torch.no_grad():
            output = model(tokens, return_audit=True)
        if was_training:
            self.autograd.train(model)
        audit = dict(output.get("audit", {}) or {})
        attention = self.summarize_attention_maps(audit.get("attention_maps", ()))
        ffn = self.summarize_ffn_activations(audit.get("ffn_activations", ()))
        return {
            "schema": "mlblack.neural_graph.audit.v1",
            "sample_shape": tuple(int(v) for v in tokens.shape),
            "attention": attention,
            "ffn": ffn,
        }

    def summarize_attention_maps(self, maps: Any) -> dict[str, Any]:
        layers: list[dict[str, Any]] = []
        head_corr_values: list[float] = []
        for idx, item in enumerate(tuple(maps or ())):
            if item is None:
                continue
            arr = item.detach().cpu().numpy().astype(float)
            layer = {
                "layer": int(idx),
                "shape": tuple(int(v) for v in arr.shape),
                "mean": float(np.mean(arr)),
                "max": float(np.max(arr)),
                "min": float(np.min(arr)),
                "entropy": _attention_entropy(arr),
            }
            corr = _attention_head_corr(arr)
            if corr is not None:
                layer["mean_abs_offdiag_head_corr"] = float(corr)
                head_corr_values.append(float(corr))
            layers.append(layer)
        return {
            "num_layers": int(len(layers)),
            "layers": tuple(layers),
            "mean_abs_offdiag_head_corr": None if not head_corr_values else float(np.mean(head_corr_values)),
        }

    def summarize_ffn_activations(self, activations: Any) -> dict[str, Any]:
        layers: list[dict[str, Any]] = []
        sparsity_values: list[float] = []
        for idx, item in enumerate(tuple(activations or ())):
            if item is None:
                continue
            arr = item.detach().cpu().numpy().astype(float)
            sparsity = float(np.mean(np.abs(arr) <= 1e-8))
            sparsity_values.append(sparsity)
            layers.append(
                {
                    "layer": int(idx),
                    "shape": tuple(int(v) for v in arr.shape),
                    "mean": float(np.mean(arr)),
                    "std": float(np.std(arr)),
                    "max_abs": float(np.max(np.abs(arr))),
                    "sparsity": sparsity,
                }
            )
        return {
            "num_layers": int(len(layers)),
            "layers": tuple(layers),
            "mean_sparsity": None if not sparsity_values else float(np.mean(sparsity_values)),
        }


def _attention_entropy(arr: np.ndarray) -> float:
    clipped = np.clip(arr, 1e-12, 1.0)
    return float(np.mean(-np.sum(clipped * np.log(clipped), axis=-1)))


def _attention_head_corr(arr: np.ndarray) -> float | None:
    if arr.ndim != 4 or arr.shape[1] < 2:
        return None
    flat = np.transpose(arr, (1, 0, 2, 3)).reshape(arr.shape[1], -1)
    if np.any(np.std(flat, axis=1) <= 1e-12):
        return None
    corr = np.corrcoef(flat)
    mask = ~np.eye(corr.shape[0], dtype=bool)
    return float(np.mean(np.abs(corr[mask])))


__all__ = ["TorchArtifactsCapability"]
