from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Sequence

import numpy as np

from pipeline import create_pipeline
from core.artifacts.artifact_persistence import ArtifactPersistenceBase
from core.models.torch_model import TorchMLPRegressor

try:
    import torch
except Exception as exc:  # pragma: no cover - environment guard
    raise ImportError(
        "PyTorch is required for TorchMLPSurrogateArtifact. Install torch before loading/predicting."
    ) from exc


def _as_2d(arr: np.ndarray) -> np.ndarray:
    x = np.asarray(arr, dtype=float)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    return x


@dataclass
class TorchMLPSurrogateArtifact(ArtifactPersistenceBase):
    """Surrogate artifact for torch MLP models."""

    artifact_id: str
    input_dim: int
    output_dim: int
    hidden_dims: Sequence[int]
    activation: str
    dropout: float
    model_state: Mapping[str, Any]
    x_mean: np.ndarray
    x_std: np.ndarray
    residual_std: np.ndarray
    feature_names: Sequence[str]
    target_names: Sequence[str]
    pipeline_name: str = "identity"
    pipeline_state: Dict[str, Any] = field(default_factory=dict)
    ood_z_threshold: float = 4.0
    input_feature_indices: Sequence[int] | None = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    _model: Any = field(default=None, init=False, repr=False)

    def _transform(self, X: np.ndarray) -> np.ndarray:
        x = _as_2d(X)
        pipe = create_pipeline(self.pipeline_name, self.pipeline_state)
        xt = pipe.transform(x)
        if self.input_feature_indices is None:
            return xt
        idx = np.asarray(tuple(self.input_feature_indices), dtype=int)
        return np.asarray(xt[:, idx], dtype=float)

    def _get_model(self):
        if self._model is None:
            model = TorchMLPRegressor(
                int(self.input_dim),
                int(self.output_dim),
                hidden_dims=tuple(int(h) for h in self.hidden_dims),
                activation=str(self.activation),
                dropout=float(self.dropout),
            )
            state = {k: v.detach().cpu() if hasattr(v, "detach") else v for k, v in dict(self.model_state).items()}
            model.load_state_dict(state, strict=True)
            model.eval()
            self._model = model
        return self._model

    def predict(self, X: np.ndarray) -> np.ndarray:
        x = self._transform(X)
        xt = torch.as_tensor(x, dtype=torch.float32)
        model = self._get_model()
        with torch.no_grad():
            y = model(xt).detach().cpu().numpy()
        return np.asarray(y, dtype=float)

    def uncertainty(self, X: np.ndarray) -> np.ndarray:
        x = self._transform(X)
        rs = np.asarray(self.residual_std, dtype=float).reshape(1, -1)
        return np.repeat(rs, x.shape[0], axis=0)

    def validity(self, X: np.ndarray) -> np.ndarray:
        x = self._transform(X)
        z = np.abs((x - self.x_mean) / np.maximum(self.x_std, 1e-8))
        max_z = np.max(z, axis=1)
        score = 1.0 - (max_z / float(self.ood_z_threshold))
        return np.clip(score, 0.0, 1.0)

    def save(self, out_dir: str) -> None:
        path = self._ensure_dir(out_dir)

        model_bundle = {
            "input_dim": int(self.input_dim),
            "output_dim": int(self.output_dim),
            "hidden_dims": [int(h) for h in self.hidden_dims],
            "activation": str(self.activation),
            "dropout": float(self.dropout),
            "state_dict": {
                k: v.detach().cpu() if hasattr(v, "detach") else v for k, v in dict(self.model_state).items()
            },
        }
        self._save_torch(path, "model.pt", model_bundle)

        self._save_npz(
            path,
            "artifact_stats.npz",
            x_mean=np.asarray(self.x_mean, dtype=float),
            x_std=np.asarray(self.x_std, dtype=float),
            residual_std=np.asarray(self.residual_std, dtype=float),
        )

        self._save_meta(
            path,
            self._common_meta(
                artifact_type="torch_mlp",
                input_feature_indices=(
                    None
                    if self.input_feature_indices is None
                    else [int(v) for v in tuple(self.input_feature_indices)]
                ),
            ),
        )

    @classmethod
    def load(cls, out_dir: str) -> "TorchMLPSurrogateArtifact":
        path = cls._ensure_dir(out_dir)
        model_bundle = cls._load_torch(path, "model.pt", map_location="cpu")
        stats = cls._load_npz(path, "artifact_stats.npz")
        meta = cls._load_meta(path)

        return cls(
            artifact_id=str(meta["artifact_id"]),
            input_dim=int(model_bundle["input_dim"]),
            output_dim=int(model_bundle["output_dim"]),
            hidden_dims=tuple(int(h) for h in model_bundle.get("hidden_dims", [])),
            activation=str(model_bundle.get("activation", "relu")),
            dropout=float(model_bundle.get("dropout", 0.0)),
            model_state=dict(model_bundle["state_dict"]),
            x_mean=np.asarray(stats["x_mean"], dtype=float),
            x_std=np.asarray(stats["x_std"], dtype=float),
            residual_std=np.asarray(stats["residual_std"], dtype=float),
            feature_names=tuple(meta.get("feature_names", [])),
            target_names=tuple(meta.get("target_names", [])),
            pipeline_name=str(meta.get("pipeline_name", "identity")),
            pipeline_state=dict(meta.get("pipeline_state", {})),
            ood_z_threshold=float(meta.get("ood_z_threshold", 4.0)),
            input_feature_indices=(
                None
                if meta.get("input_feature_indices") is None
                else tuple(int(v) for v in tuple(meta.get("input_feature_indices", ())))
            ),
            metadata=dict(meta.get("metadata", {})),
        )
