from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Sequence

import numpy as np

from pipeline import create_pipeline
from core.artifacts.artifact_persistence import ArtifactPersistenceBase


def _as_2d(arr: np.ndarray) -> np.ndarray:
    x = np.asarray(arr, dtype=float)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    return x


@dataclass
class LinearSurrogateArtifact(ArtifactPersistenceBase):
    """Minimal surrogate artifact produced by core trainer."""

    artifact_id: str
    coef: np.ndarray  # shape: (D, M)
    intercept: np.ndarray  # shape: (M,)
    x_mean: np.ndarray  # shape: (D,)
    x_std: np.ndarray  # shape: (D,)
    residual_std: np.ndarray  # shape: (M,)
    feature_names: Sequence[str]
    target_names: Sequence[str]
    pipeline_name: str = "identity"
    pipeline_state: Dict[str, Any] = field(default_factory=dict)
    ood_z_threshold: float = 4.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def _transform(self, X: np.ndarray) -> np.ndarray:
        x = _as_2d(X)
        pipe = create_pipeline(self.pipeline_name, self.pipeline_state)
        return pipe.transform(x)

    def predict(self, X: np.ndarray) -> np.ndarray:
        x = self._transform(X)
        y = x @ self.coef + self.intercept
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

        self._save_npz(
            path,
            "artifact.npz",
            coef=self.coef,
            intercept=self.intercept,
            x_mean=self.x_mean,
            x_std=self.x_std,
            residual_std=self.residual_std,
        )

        self._save_meta(
            path,
            self._common_meta(artifact_type="linear"),
        )

    @classmethod
    def load(cls, out_dir: str) -> "LinearSurrogateArtifact":
        path = cls._ensure_dir(out_dir)
        payload = cls._load_npz(path, "artifact.npz")
        meta = cls._load_meta(path)
        return cls(
            artifact_id=str(meta["artifact_id"]),
            coef=np.asarray(payload["coef"], dtype=float),
            intercept=np.asarray(payload["intercept"], dtype=float),
            x_mean=np.asarray(payload["x_mean"], dtype=float),
            x_std=np.asarray(payload["x_std"], dtype=float),
            residual_std=np.asarray(payload["residual_std"], dtype=float),
            feature_names=tuple(meta.get("feature_names", [])),
            target_names=tuple(meta.get("target_names", [])),
            pipeline_name=str(meta.get("pipeline_name", "identity")),
            pipeline_state=dict(meta.get("pipeline_state", {})),
            ood_z_threshold=float(meta.get("ood_z_threshold", 4.0)),
            metadata=dict(meta.get("metadata", {})),
        )
