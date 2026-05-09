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
class TreeEnsembleSurrogateArtifact(ArtifactPersistenceBase):
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
    model_family: str = "tree_ensemble"
    ensemble_kind: str = "random_forest"
    uncertainty_mode: str = "ensemble_std"
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

    def _ensemble_uncertainty(self, x: np.ndarray) -> np.ndarray | None:
        estimators = tuple(getattr(self.model, "estimators_", ()) or ())
        if len(estimators) <= 1:
            return None
        estimator_features = tuple(getattr(self.model, "estimators_features_", ()) or ())
        preds = []
        for i, est in enumerate(estimators):
            local_x = x
            if i < len(estimator_features):
                local_idx = np.asarray(tuple(estimator_features[i]), dtype=int)
                local_x = np.asarray(x[:, local_idx], dtype=float)
            y = np.asarray(est.predict(local_x), dtype=float)
            if y.ndim == 1:
                y = y.reshape(-1, 1)
            preds.append(y)
        if not preds:
            return None
        stack = np.stack(preds, axis=0)
        weights = np.asarray(getattr(self.model, "estimator_weights_", ()), dtype=float).reshape(-1)
        if weights.size == stack.shape[0] and np.sum(weights) > 0.0:
            norm = weights / np.sum(weights)
            mean = np.tensordot(norm, stack, axes=(0, 0))
            var = np.tensordot(norm, (stack - mean) ** 2, axes=(0, 0))
            return np.sqrt(np.maximum(var, 0.0))
        return np.asarray(np.std(stack, axis=0, ddof=1), dtype=float)

    def uncertainty(self, X: np.ndarray) -> np.ndarray:
        x = self._transform(X)
        mode = str(self.uncertainty_mode or "ensemble_std").strip().lower()
        if mode == "ensemble_std":
            ensemble_std = self._ensemble_uncertainty(x)
            if ensemble_std is not None:
                return ensemble_std
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
                artifact_type="tree_ensemble",
                model_family=str(self.model_family),
                ensemble_kind=str(self.ensemble_kind),
                uncertainty_mode=str(self.uncertainty_mode),
                input_feature_indices=(
                    None
                    if self.input_feature_indices is None
                    else [int(v) for v in tuple(self.input_feature_indices)]
                ),
            ),
        )

    @classmethod
    def load(cls, out_dir: str) -> "TreeEnsembleSurrogateArtifact":
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
            model_family=str(meta.get("model_family", "tree_ensemble")),
            ensemble_kind=str(meta.get("ensemble_kind", "random_forest")),
            uncertainty_mode=str(meta.get("uncertainty_mode", "ensemble_std")),
            input_feature_indices=(
                None
                if meta.get("input_feature_indices") is None
                else tuple(int(v) for v in tuple(meta.get("input_feature_indices", ())))
            ),
            metadata=dict(meta.get("metadata", {})),
        )


__all__ = ["TreeEnsembleSurrogateArtifact"]
