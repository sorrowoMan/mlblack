from __future__ import annotations

from typing import Any

import numpy as np

from mlblack.backends.contracts import BackendCapabilityContract


class NumpyArtifactsCapability:
    contract = BackendCapabilityContract(
        backend="numpy",
        capability="artifacts",
        provides=("artifact.parameters.summary", "artifact.numpy_model.describe"),
        methods={
            "artifact.parameters.summary": "parameter_layout_summary(model) -> dict",
            "artifact.numpy_model.describe": "describe_model(model) -> dict",
        },
        model_kinds=("NumpyMLPPointModel",),
        supports_stateful_module=False,
    )

    def parameter_layout_summary(self, model: Any) -> dict[str, Any]:
        if not hasattr(model, "parameter_shapes"):
            raise TypeError("numpy artifact summary requires model.parameter_shapes()")
        shapes = tuple(tuple(int(v) for v in shape) for shape in model.parameter_shapes())
        names: list[str] = []
        for idx in range(len(shapes) // 2):
            names.append(f"mlp.layers.{idx}.weight")
            names.append(f"mlp.layers.{idx}.bias")
        total = int(sum(np.prod(shape) for shape in shapes))
        return {"names": tuple(names), "shapes": shapes, "total_size": total}

    def describe_model(self, model: Any) -> dict[str, Any]:
        metadata = dict(getattr(model, "metadata", {}) or {})
        return {
            "model_type": type(model).__name__,
            "metadata": metadata,
            "parameter_layout": self.parameter_layout_summary(model),
        }


__all__ = ["NumpyArtifactsCapability"]
