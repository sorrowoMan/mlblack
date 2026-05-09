from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Sequence

import numpy as np

from pipeline import create_pipeline
from core.artifacts.artifact_persistence import ArtifactPersistenceBase
from core.symbolic.artifact_schema import build_symbolic_interval_artifact_schema
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


def _expression_from_parts(
    genome: Sequence[Mapping[str, Any]],
    parameter_values: Mapping[str, float],
    readout_weight: np.ndarray,
    readout_bias: np.ndarray,
    *,
    target_index: int,
    precision: int,
    feature_names: Sequence[str],
    use_feature_names: bool,
) -> str:
    w = np.asarray(readout_weight, dtype=float)
    b = np.asarray(readout_bias, dtype=float)

    if w.ndim != 2:
        raise ValueError("readout_weight must be 2D")
    t = int(target_index)
    if t < 0 or t >= int(w.shape[1]):
        raise ValueError(f"target_index out of range: {t}")

    pieces: list[str] = []
    for i, term in enumerate(genome):
        coeff = float(w[i, t])
        if abs(coeff) < 1e-12:
            continue
        term_expr = expression_to_string(
            term["expr"],
            param_values=dict(parameter_values),
            precision=int(precision),
        )
        if bool(use_feature_names):
            term_expr = _replace_feature_tokens(term_expr, feature_names)
        pieces.append(f"({coeff:.{int(precision)}g})*({term_expr})")

    pieces.append(f"({float(b[t]):.{int(precision)}g})")
    return " + ".join(pieces)


@dataclass
class SymbolicIntervalSurrogateArtifact(ArtifactPersistenceBase):
    """Surrogate artifact for symbolic interval trainer (lower/upper bounds)."""

    artifact_id: str
    lower_quantile: float
    upper_quantile: float

    genome_low: Sequence[Mapping[str, Any]]
    parameter_values_low: Mapping[str, float]
    readout_weight_low: np.ndarray
    readout_bias_low: np.ndarray

    genome_high: Sequence[Mapping[str, Any]]
    parameter_values_high: Mapping[str, float]
    readout_weight_high: np.ndarray
    readout_bias_high: np.ndarray

    x_mean: np.ndarray
    x_std: np.ndarray
    residual_std: np.ndarray
    calibration_margin: np.ndarray
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

    def _basis_low(self, X: np.ndarray) -> np.ndarray:
        return evaluate_genome_numpy(
            self.genome_low,
            X,
            param_values=self.parameter_values_low,
            eps=float(self.epsilon),
        )

    def _basis_high(self, X: np.ndarray) -> np.ndarray:
        return evaluate_genome_numpy(
            self.genome_high,
            X,
            param_values=self.parameter_values_high,
            eps=float(self.epsilon),
        )

    def predict_interval(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = self._transform(X)

        phi_l = self._basis_low(x)
        low = phi_l @ np.asarray(self.readout_weight_low, dtype=float) + np.asarray(self.readout_bias_low, dtype=float)

        phi_h = self._basis_high(x)
        high = phi_h @ np.asarray(self.readout_weight_high, dtype=float) + np.asarray(self.readout_bias_high, dtype=float)

        low = np.asarray(low, dtype=float)
        high = np.asarray(high, dtype=float)
        if low.ndim == 1:
            low = low.reshape(-1, 1)
        if high.ndim == 1:
            high = high.reshape(-1, 1)

        lo = np.minimum(low, high)
        hi = np.maximum(low, high)

        margin = np.asarray(self.calibration_margin, dtype=float).reshape(1, -1)
        if margin.shape[1] == 1 and lo.shape[1] > 1:
            margin = np.repeat(margin, lo.shape[1], axis=1)

        lo = lo - margin
        hi = hi + margin
        return lo, hi

    def predict(self, X: np.ndarray) -> np.ndarray:
        lo, hi = self.predict_interval(X)
        return 0.5 * (lo + hi)

    def uncertainty(self, X: np.ndarray) -> np.ndarray:
        lo, hi = self.predict_interval(X)
        return np.maximum(hi - lo, 1e-8)

    def validity(self, X: np.ndarray) -> np.ndarray:
        x = self._transform(X)
        z = np.abs((x - self.x_mean) / np.maximum(self.x_std, 1e-8))
        max_z = np.max(z, axis=1)
        score = 1.0 - (max_z / float(self.ood_z_threshold))
        return np.clip(score, 0.0, 1.0)

    def expression(
        self,
        *,
        bound: str = "center",  # low | high | center
        target_index: int = 0,
        precision: int = 6,
        use_feature_names: bool = False,
    ) -> str:
        key = str(bound).strip().lower()
        if key == "low":
            return _expression_from_parts(
                self.genome_low,
                self.parameter_values_low,
                self.readout_weight_low,
                self.readout_bias_low,
                target_index=int(target_index),
                precision=int(precision),
                feature_names=self.feature_names,
                use_feature_names=bool(use_feature_names),
            )
        if key == "high":
            return _expression_from_parts(
                self.genome_high,
                self.parameter_values_high,
                self.readout_weight_high,
                self.readout_bias_high,
                target_index=int(target_index),
                precision=int(precision),
                feature_names=self.feature_names,
                use_feature_names=bool(use_feature_names),
            )
        if key == "center":
            low_expr = self.expression(
                bound="low",
                target_index=int(target_index),
                precision=int(precision),
                use_feature_names=bool(use_feature_names),
            )
            high_expr = self.expression(
                bound="high",
                target_index=int(target_index),
                precision=int(precision),
                use_feature_names=bool(use_feature_names),
            )
            return f"0.5*(({low_expr}) + ({high_expr}))"
        raise ValueError("bound must be one of: low, high, center")

    def expressions(self, *, precision: int = 6, use_feature_names: bool = False) -> dict[str, dict[str, str]]:
        weights = np.asarray(self.readout_weight_low, dtype=float)
        target_dim = int(weights.shape[1]) if weights.ndim == 2 else 1
        target_labels = tuple(self.target_names) if self.target_names else tuple(f"y{i}" for i in range(target_dim))

        out: dict[str, dict[str, str]] = {}
        for target_index in range(target_dim):
            target_name = str(target_labels[target_index]) if target_index < len(target_labels) else f"y{target_index}"
            out[target_name] = {
                "low": self.expression(
                    bound="low",
                    target_index=target_index,
                    precision=int(precision),
                    use_feature_names=bool(use_feature_names),
                ),
                "high": self.expression(
                    bound="high",
                    target_index=target_index,
                    precision=int(precision),
                    use_feature_names=bool(use_feature_names),
                ),
                "center": self.expression(
                    bound="center",
                    target_index=target_index,
                    precision=int(precision),
                    use_feature_names=bool(use_feature_names),
                ),
            }
        return out

    def symbolic_artifact_schema(self) -> dict[str, Any]:
        return build_symbolic_interval_artifact_schema(
            artifact_type="symbolic_torch_interval",
            artifact_id=str(self.artifact_id),
            genome_low=tuple(self.genome_low),
            parameter_values_low=dict(self.parameter_values_low),
            readout_weight_low=np.asarray(self.readout_weight_low, dtype=float),
            readout_bias_low=np.asarray(self.readout_bias_low, dtype=float),
            genome_high=tuple(self.genome_high),
            parameter_values_high=dict(self.parameter_values_high),
            readout_weight_high=np.asarray(self.readout_weight_high, dtype=float),
            readout_bias_high=np.asarray(self.readout_bias_high, dtype=float),
            feature_names=tuple(self.feature_names),
            target_names=tuple(self.target_names),
            residual_std=np.asarray(self.residual_std, dtype=float),
            calibration_margin=np.asarray(self.calibration_margin, dtype=float),
            lower_quantile=float(self.lower_quantile),
            upper_quantile=float(self.upper_quantile),
            metadata=dict(self.metadata),
            final_expression=self.expressions(precision=12, use_feature_names=True),
            normalized_expression=self.expressions(precision=12, use_feature_names=False),
        )

    def save(self, out_dir: str) -> None:
        path = self._ensure_dir(out_dir)

        self._save_npz(
            path,
            "artifact_stats.npz",
            readout_weight_low=np.asarray(self.readout_weight_low, dtype=float),
            readout_bias_low=np.asarray(self.readout_bias_low, dtype=float),
            readout_weight_high=np.asarray(self.readout_weight_high, dtype=float),
            readout_bias_high=np.asarray(self.readout_bias_high, dtype=float),
            x_mean=np.asarray(self.x_mean, dtype=float),
            x_std=np.asarray(self.x_std, dtype=float),
            residual_std=np.asarray(self.residual_std, dtype=float),
            calibration_margin=np.asarray(self.calibration_margin, dtype=float),
        )

        meta = self._common_meta(
            artifact_type="symbolic_torch_interval",
            lower_quantile=float(self.lower_quantile),
            upper_quantile=float(self.upper_quantile),
            genome_low=list(self.genome_low),
            parameter_values_low={str(k): float(v) for k, v in dict(self.parameter_values_low).items()},
            genome_high=list(self.genome_high),
            parameter_values_high={str(k): float(v) for k, v in dict(self.parameter_values_high).items()},
            epsilon=float(self.epsilon),
        )
        self._save_meta(path, meta)

        names = tuple(self.target_names) if self.target_names else ("y0",)
        formulas_raw: dict[str, dict[str, str]] = {}
        formulas_named: dict[str, dict[str, str]] = {}

        lines: list[str] = []
        for i, tname in enumerate(names):
            key = str(tname)
            low_raw = self.expression(bound="low", target_index=i, precision=12, use_feature_names=False)
            high_raw = self.expression(bound="high", target_index=i, precision=12, use_feature_names=False)
            center_raw = self.expression(bound="center", target_index=i, precision=12, use_feature_names=False)

            low_named = self.expression(bound="low", target_index=i, precision=12, use_feature_names=True)
            high_named = self.expression(bound="high", target_index=i, precision=12, use_feature_names=True)
            center_named = self.expression(bound="center", target_index=i, precision=12, use_feature_names=True)

            formulas_raw[key] = {"low": low_raw, "high": high_raw, "center": center_raw}
            formulas_named[key] = {"low": low_named, "high": high_named, "center": center_named}

            lines.append(f"{key}_low = {low_named}")
            lines.append(f"{key}_high = {high_named}")
            lines.append(f"{key}_center = {center_named}")
            lines.append("")

        self._save_json(
            path,
            "formulas.json",
            {"raw": formulas_raw, "feature_named": formulas_named},
        )
        self._save_json(path, "symbolic_schema.json", self.symbolic_artifact_schema())
        self._save_text(path, "formula.txt", "\n".join(lines).strip() + "\n")

    @classmethod
    def load(cls, out_dir: str) -> "SymbolicIntervalSurrogateArtifact":
        path = cls._ensure_dir(out_dir)
        stats = cls._load_npz(path, "artifact_stats.npz")
        meta = cls._load_meta(path)

        if "calibration_margin" in stats:
            margin = np.asarray(stats["calibration_margin"], dtype=float)
        else:
            margin = np.zeros((len(tuple(meta.get("target_names", []))) or 1,), dtype=float)

        return cls(
            artifact_id=str(meta["artifact_id"]),
            lower_quantile=float(meta.get("lower_quantile", 0.1)),
            upper_quantile=float(meta.get("upper_quantile", 0.9)),
            genome_low=tuple(meta.get("genome_low", [])),
            parameter_values_low=dict(meta.get("parameter_values_low", {})),
            readout_weight_low=np.asarray(stats["readout_weight_low"], dtype=float),
            readout_bias_low=np.asarray(stats["readout_bias_low"], dtype=float),
            genome_high=tuple(meta.get("genome_high", [])),
            parameter_values_high=dict(meta.get("parameter_values_high", {})),
            readout_weight_high=np.asarray(stats["readout_weight_high"], dtype=float),
            readout_bias_high=np.asarray(stats["readout_bias_high"], dtype=float),
            x_mean=np.asarray(stats["x_mean"], dtype=float),
            x_std=np.asarray(stats["x_std"], dtype=float),
            residual_std=np.asarray(stats["residual_std"], dtype=float),
            calibration_margin=np.asarray(margin, dtype=float).reshape(-1),
            feature_names=tuple(meta.get("feature_names", [])),
            target_names=tuple(meta.get("target_names", [])),
            pipeline_name=str(meta.get("pipeline_name", "identity")),
            pipeline_state=dict(meta.get("pipeline_state", {})),
            ood_z_threshold=float(meta.get("ood_z_threshold", 4.0)),
            epsilon=float(meta.get("epsilon", 1e-6)),
            metadata=dict(meta.get("metadata", {})),
        )
