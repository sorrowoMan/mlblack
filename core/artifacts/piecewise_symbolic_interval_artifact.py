from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from core.artifacts.artifact_persistence import ArtifactPersistenceBase
from core.artifacts.symbolic_interval_artifact import SymbolicIntervalSurrogateArtifact
from core.symbolic.artifact_schema import build_piecewise_symbolic_interval_artifact_schema


def _as_2d(arr: np.ndarray) -> np.ndarray:
    x = np.asarray(arr, dtype=float)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    return x


def _canonical_key(bits: Sequence[int]) -> str:
    return "|".join(str(int(v)) for v in bits)


@dataclass
class PiecewiseSymbolicIntervalSurrogateArtifact(ArtifactPersistenceBase):
    """Piecewise interval artifact routing by binary gate features."""

    artifact_id: str
    global_artifact: SymbolicIntervalSurrogateArtifact
    local_artifacts: Mapping[str, SymbolicIntervalSurrogateArtifact]
    gate_feature_names: Sequence[str]
    blend_kappa: float = 512.0
    regime_counts: Mapping[str, int] = field(default_factory=dict)
    feature_names: Sequence[str] = field(default_factory=tuple)
    target_names: Sequence[str] = field(default_factory=tuple)
    pipeline_name: str = "identity"
    pipeline_state: Mapping[str, Any] = field(default_factory=dict)
    ood_z_threshold: float = 4.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.feature_names:
            self.feature_names = tuple(self.global_artifact.feature_names)
        if not self.target_names:
            self.target_names = tuple(self.global_artifact.target_names)
        self.pipeline_name = str(self.global_artifact.pipeline_name)
        self.pipeline_state = dict(self.global_artifact.pipeline_state)
        self.ood_z_threshold = float(self.global_artifact.ood_z_threshold)
        self.blend_kappa = float(max(1e-8, float(self.blend_kappa)))
        self.local_artifacts = {
            str(k): v for k, v in dict(self.local_artifacts).items() if isinstance(v, SymbolicIntervalSurrogateArtifact)
        }
        self.regime_counts = {str(k): int(max(0, int(v))) for k, v in dict(self.regime_counts).items()}
        self.gate_feature_names = tuple(str(v) for v in self.gate_feature_names)
        if not isinstance(self.metadata, dict):
            self.metadata = dict(self.metadata or {})
        schema = self.symbolic_artifact_schema()
        self.metadata["symbolic_artifact_schema"] = dict(schema)
        self.metadata["symbolic_complexity_metrics"] = dict(schema.get("complexity_metrics", {}))
        self.metadata["symbolic_head_semantics"] = dict(schema.get("head_semantics", {}))

    def _gate_indices(self) -> tuple[int, ...]:
        names = tuple(str(v) for v in self.feature_names)
        out = [int(names.index(g)) for g in self.gate_feature_names if g in names]
        return tuple(out)

    def _gate_key_by_row(self, X: np.ndarray) -> tuple[str, ...]:
        x = _as_2d(X)
        gate_idx = self._gate_indices()
        if len(gate_idx) == 0:
            return tuple("GLOBAL" for _ in range(int(x.shape[0])))
        gate = np.asarray(x[:, list(gate_idx)], dtype=float)
        return tuple(_canonical_key(tuple(int(v > 0.5) for v in row)) for row in gate)

    def _blend_alpha(self, key: str) -> float:
        n = int(self.regime_counts.get(str(key), 0))
        return float(n) / float(n + float(self.blend_kappa))

    def predict_interval(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = _as_2d(X)
        glo_lo, glo_hi = self.global_artifact.predict_interval(x)
        out_lo = np.asarray(glo_lo, dtype=float).copy()
        out_hi = np.asarray(glo_hi, dtype=float).copy()

        keys = np.asarray(self._gate_key_by_row(x), dtype=object)
        if keys.size == 0:
            return out_lo, out_hi

        for key in sorted(set(str(v) for v in keys.tolist())):
            local = self.local_artifacts.get(str(key))
            if local is None:
                continue
            mask = keys == str(key)
            if not bool(np.any(mask)):
                continue
            loc_lo, loc_hi = local.predict_interval(x[mask, :])
            alpha = float(np.clip(self._blend_alpha(str(key)), 0.0, 1.0))
            out_lo[mask, :] = alpha * np.asarray(loc_lo, dtype=float) + (1.0 - alpha) * out_lo[mask, :]
            out_hi[mask, :] = alpha * np.asarray(loc_hi, dtype=float) + (1.0 - alpha) * out_hi[mask, :]

        return out_lo, out_hi

    def predict(self, X: np.ndarray) -> np.ndarray:
        lo, hi = self.predict_interval(X)
        return 0.5 * (np.asarray(lo, dtype=float) + np.asarray(hi, dtype=float))

    def uncertainty(self, X: np.ndarray) -> np.ndarray:
        lo, hi = self.predict_interval(X)
        return np.maximum(np.asarray(hi, dtype=float) - np.asarray(lo, dtype=float), 1e-8)

    def validity(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self.global_artifact.validity(X), dtype=float)

    def symbolic_artifact_schema(self) -> dict[str, Any]:
        global_schema = dict(self.global_artifact.metadata.get("symbolic_artifact_schema", {}))
        if not global_schema:
            global_schema = self.global_artifact.symbolic_artifact_schema()

        local_schemas: dict[str, dict[str, Any]] = {}
        for key, artifact in sorted(self.local_artifacts.items(), key=lambda item: item[0]):
            schema = dict(artifact.metadata.get("symbolic_artifact_schema", {}))
            if not schema:
                schema = artifact.symbolic_artifact_schema()
            local_schemas[str(key)] = schema

        return build_piecewise_symbolic_interval_artifact_schema(
            artifact_type="piecewise_symbolic_torch_interval",
            artifact_id=str(self.artifact_id),
            feature_names=tuple(self.feature_names),
            gate_feature_names=tuple(self.gate_feature_names),
            blend_kappa=float(self.blend_kappa),
            regime_counts={str(k): int(v) for k, v in self.regime_counts.items()},
            metadata=dict(self.metadata),
            global_schema=global_schema,
            local_schemas=local_schemas,
        )

    def save(self, out_dir: str) -> None:
        path = self._ensure_dir(out_dir)

        global_dir = path / "global_artifact"
        self.global_artifact.save(str(global_dir))

        locals_dir = path / "local_artifacts"
        locals_dir.mkdir(parents=True, exist_ok=True)

        local_dirs: dict[str, str] = {}
        for key in sorted(self.local_artifacts.keys()):
            safe = str(key).replace("|", "_")
            rel_dir = Path("local_artifacts") / f"gate_{safe}"
            local_dirs[str(key)] = str(rel_dir)
            self.local_artifacts[str(key)].save(str(path / rel_dir))

        meta = self._common_meta(
            artifact_type="piecewise_symbolic_torch_interval",
            gate_feature_names=list(self.gate_feature_names),
            blend_kappa=float(self.blend_kappa),
            regime_counts={str(k): int(v) for k, v in self.regime_counts.items()},
            local_dirs=local_dirs,
        )
        self._save_meta(path, meta)
        self._save_json(path, "symbolic_schema.json", self.symbolic_artifact_schema())

    @classmethod
    def load(cls, out_dir: str) -> "PiecewiseSymbolicIntervalSurrogateArtifact":
        path = cls._ensure_dir(out_dir)
        meta = cls._load_meta(path)

        global_art = SymbolicIntervalSurrogateArtifact.load(str(path / "global_artifact"))

        local_map: dict[str, SymbolicIntervalSurrogateArtifact] = {}
        for key, rel_dir in dict(meta.get("local_dirs", {})).items():
            local_map[str(key)] = SymbolicIntervalSurrogateArtifact.load(str(path / str(rel_dir)))

        return cls(
            artifact_id=str(meta.get("artifact_id", "")),
            global_artifact=global_art,
            local_artifacts=local_map,
            gate_feature_names=tuple(meta.get("gate_feature_names", [])),
            blend_kappa=float(meta.get("blend_kappa", 512.0)),
            regime_counts={str(k): int(v) for k, v in dict(meta.get("regime_counts", {})).items()},
            feature_names=tuple(meta.get("feature_names", tuple(global_art.feature_names))),
            target_names=tuple(meta.get("target_names", tuple(global_art.target_names))),
            pipeline_name=str(meta.get("pipeline_name", "identity")),
            pipeline_state=dict(meta.get("pipeline_state", {})),
            ood_z_threshold=float(meta.get("ood_z_threshold", 4.0)),
            metadata=dict(meta.get("metadata", {})),
        )
