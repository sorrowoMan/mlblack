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
class XGBoostMultiOutputModelWrapper:
    """Pickle-friendly multi-output wrapper with a stable predict interface."""

    estimators: Sequence[Any]

    def __post_init__(self) -> None:
        self.estimators = tuple(self.estimators)

    @property
    def n_features_in_(self) -> int | None:
        if not self.estimators:
            return None
        value = getattr(self.estimators[0], "n_features_in_", None)
        return None if value is None else int(value)

    def predict(self, X: np.ndarray) -> np.ndarray:
        x = _as_2d(X)
        cols = [np.asarray(est.predict(x), dtype=float).reshape(-1, 1) for est in self.estimators]
        if not cols:
            return np.zeros((int(x.shape[0]), 0), dtype=float)
        return np.asarray(np.concatenate(cols, axis=1), dtype=float)


@dataclass
class XGBoostSurrogateArtifact(ArtifactPersistenceBase):
    """Surrogate artifact for xgboost models."""

    artifact_id: str
    model: Any
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

    def _transform(self, X: np.ndarray) -> np.ndarray:
        x = _as_2d(X)
        pipe = create_pipeline(self.pipeline_name, self.pipeline_state)
        xt = pipe.transform(x)
        if self.input_feature_indices is None:
            return xt
        idx = np.asarray(tuple(self.input_feature_indices), dtype=int)
        return np.asarray(xt[:, idx], dtype=float)

    def predict(self, X: np.ndarray) -> np.ndarray:
        x = self._transform(X)
        y = np.asarray(self.model.predict(x), dtype=float)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        return y

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

        self._save_pickle(path, "model.pkl", self.model)

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
                artifact_type="xgboost",
                input_feature_indices=(
                    None
                    if self.input_feature_indices is None
                    else [int(v) for v in tuple(self.input_feature_indices)]
                ),
            ),
        )

    @classmethod
    def load(cls, out_dir: str) -> "XGBoostSurrogateArtifact":
        path = cls._ensure_dir(out_dir)
        model = cls._load_pickle(path, "model.pkl")

        stats = cls._load_npz(path, "artifact_stats.npz")
        meta = cls._load_meta(path)

        return cls(
            artifact_id=str(meta["artifact_id"]),
            model=model,
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
