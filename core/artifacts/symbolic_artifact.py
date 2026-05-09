from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Sequence

import numpy as np

from pipeline import create_pipeline
from core.artifacts.artifact_persistence import ArtifactPersistenceBase
from core.symbolic.artifact_schema import build_symbolic_point_artifact_schema
from core.symbolic.symbolic_dsl import expression_to_string, evaluate_genome_numpy


def _as_2d(arr: np.ndarray) -> np.ndarray:
    x = np.asarray(arr, dtype=float)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    return x


def _replace_feature_tokens(expr: str, feature_names: Sequence[str]) -> str:
    names = tuple(str(n) for n in feature_names)

    def repl(match: re.Match[str]) -> str:
        idx = int(match.group(1))
        if 0 <= idx < len(names):
            return names[idx]
        return match.group(0)

    return re.sub(r"\bx(\d+)\b", repl, str(expr))


@dataclass
class SymbolicSurrogateArtifact(ArtifactPersistenceBase):
    """Surrogate artifact for symbolic torch trainer."""

    artifact_id: str
    genome: Sequence[Mapping[str, Any]]
    parameter_values: Mapping[str, float]
    readout_weight: np.ndarray  # (T, M)
    readout_bias: np.ndarray  # (M,)
    x_mean: np.ndarray
    x_std: np.ndarray
    residual_std: np.ndarray
    feature_names: Sequence[str]
    target_names: Sequence[str]
    pipeline_name: str = "identity"
    pipeline_state: Dict[str, Any] = field(default_factory=dict)
    ood_z_threshold: float = 4.0
    epsilon: float = 1e-6
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.metadata, dict):
            self.metadata = dict(self.metadata or {})
        schema = self.symbolic_artifact_schema()
        self.metadata["symbolic_artifact_schema"] = dict(schema)
        self.metadata["symbolic_complexity_metrics"] = dict(schema.get("complexity_metrics", {}))
        self.metadata["symbolic_head_semantics"] = dict(schema.get("head_semantics", {}))

    def _transform(self, X: np.ndarray) -> np.ndarray:
        x = _as_2d(X)
        pipe = create_pipeline(self.pipeline_name, self.pipeline_state)
        return pipe.transform(x)

    def _basis(self, X: np.ndarray) -> np.ndarray:
        if len(tuple(self.genome)) == 0:
            x = _as_2d(X)
            return np.zeros((int(x.shape[0]), 0), dtype=float)
        return evaluate_genome_numpy(
            self.genome,
            X,
            param_values=self.parameter_values,
            eps=float(self.epsilon),
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        x = self._transform(X)
        phi = self._basis(x)
        y = phi @ np.asarray(self.readout_weight, dtype=float) + np.asarray(self.readout_bias, dtype=float)
        y = np.asarray(y, dtype=float)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        return y

    def uncertainty(self, X: np.ndarray) -> np.ndarray:
        x = self._transform(X)
        rs = np.asarray(self.residual_std, dtype=float).reshape(1, -1)
        return np.repeat(rs, int(x.shape[0]), axis=0)

    def validity(self, X: np.ndarray) -> np.ndarray:
        x = self._transform(X)
        z = np.abs((x - self.x_mean) / np.maximum(self.x_std, 1e-8))
        max_z = np.max(z, axis=1)
        score = 1.0 - (max_z / float(self.ood_z_threshold))
        return np.clip(score, 0.0, 1.0)

    def expression(self, *, target_index: int = 0, precision: int = 6, use_feature_names: bool = False) -> str:
        t = int(target_index)
        w = np.asarray(self.readout_weight, dtype=float)
        b = np.asarray(self.readout_bias, dtype=float)

        if w.ndim != 2:
            raise ValueError("readout_weight must be 2D")
        if t < 0 or t >= int(w.shape[1]):
            raise ValueError(f"target_index out of range: {t}")

        pieces: list[str] = []
        for i, term in enumerate(self.genome):
            coeff = float(w[i, t])
            if abs(coeff) < 1e-12:
                continue
            term_expr = expression_to_string(
                term["expr"],
                param_values=dict(self.parameter_values),
                precision=int(precision),
            )
            if bool(use_feature_names):
                term_expr = _replace_feature_tokens(term_expr, self.feature_names)
            pieces.append(f"({coeff:.{int(precision)}g})*({term_expr})")

        pieces.append(f"({float(b[t]):.{int(precision)}g})")
        return " + ".join(pieces)

    def expressions(self, *, precision: int = 6, use_feature_names: bool = False) -> dict[str, str]:
        w = np.asarray(self.readout_weight, dtype=float)
        target_dim = int(w.shape[1]) if w.ndim == 2 else 1

        names = tuple(self.target_names) if self.target_names else tuple(f"y{i}" for i in range(target_dim))
        out: dict[str, str] = {}
        for i in range(target_dim):
            key = str(names[i]) if i < len(names) else f"y{i}"
            out[key] = self.expression(
                target_index=i,
                precision=int(precision),
                use_feature_names=bool(use_feature_names),
            )
        return out

    def symbolic_artifact_schema(self) -> dict[str, Any]:
        return build_symbolic_point_artifact_schema(
            artifact_type="symbolic_torch",
            artifact_id=str(self.artifact_id),
            genome=tuple(self.genome),
            parameter_values=dict(self.parameter_values),
            readout_weight=np.asarray(self.readout_weight, dtype=float),
            readout_bias=np.asarray(self.readout_bias, dtype=float),
            feature_names=tuple(self.feature_names),
            target_names=tuple(self.target_names),
            residual_std=np.asarray(self.residual_std, dtype=float),
            metadata=dict(self.metadata),
            final_expression=self.expressions(precision=12, use_feature_names=True),
            normalized_expression=self.expressions(precision=12, use_feature_names=False),
        )

    def save(self, out_dir: str) -> None:
        path = self._ensure_dir(out_dir)

        self._save_npz(
            path,
            "artifact_stats.npz",
            readout_weight=np.asarray(self.readout_weight, dtype=float),
            readout_bias=np.asarray(self.readout_bias, dtype=float),
            x_mean=np.asarray(self.x_mean, dtype=float),
            x_std=np.asarray(self.x_std, dtype=float),
            residual_std=np.asarray(self.residual_std, dtype=float),
        )

        meta = self._common_meta(
            artifact_type="symbolic_torch",
            genome=list(self.genome),
            parameter_values={str(k): float(v) for k, v in dict(self.parameter_values).items()},
            epsilon=float(self.epsilon),
        )
        self._save_meta(path, meta)

        formulas_raw = self.expressions(precision=12, use_feature_names=False)
        formulas_named = self.expressions(precision=12, use_feature_names=True)

        self._save_json(
            path,
            "formulas.json",
            {
                "raw": formulas_raw,
                "feature_named": formulas_named,
            },
        )
        self._save_json(path, "symbolic_schema.json", self.symbolic_artifact_schema())

        lines: list[str] = []
        for target_name, expr in formulas_named.items():
            lines.append(f"{target_name} = {expr}")
            lines.append("")

        self._save_text(path, "formula.txt", "\n".join(lines).strip() + "\n")

    @classmethod
    def load(cls, out_dir: str) -> "SymbolicSurrogateArtifact":
        path = cls._ensure_dir(out_dir)
        stats = cls._load_npz(path, "artifact_stats.npz")
        meta = cls._load_meta(path)

        return cls(
            artifact_id=str(meta["artifact_id"]),
            genome=tuple(meta.get("genome", [])),
            parameter_values=dict(meta.get("parameter_values", {})),
            readout_weight=np.asarray(stats["readout_weight"], dtype=float),
            readout_bias=np.asarray(stats["readout_bias"], dtype=float),
            x_mean=np.asarray(stats["x_mean"], dtype=float),
            x_std=np.asarray(stats["x_std"], dtype=float),
            residual_std=np.asarray(stats["residual_std"], dtype=float),
            feature_names=tuple(meta.get("feature_names", [])),
            target_names=tuple(meta.get("target_names", [])),
            pipeline_name=str(meta.get("pipeline_name", "identity")),
            pipeline_state=dict(meta.get("pipeline_state", {})),
            ood_z_threshold=float(meta.get("ood_z_threshold", 4.0)),
            epsilon=float(meta.get("epsilon", 1e-6)),
            metadata=dict(meta.get("metadata", {})),
        )
