from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from mlblack.pipeline.symbolic import safe_corr

from .artifacts import OrthogonalBasisSetArtifact


@dataclass(frozen=True)
class BasisConsensusConfig:
    min_support_ratio: float = 0.5
    include_value_overlap: bool = True
    max_report_terms: int = 64
    high_value_overlap_threshold: float = 0.995
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BasisConsensusReport:
    artifact_count: int
    atom_count: int
    consensus_terms: tuple[Mapping[str, Any], ...]
    expression_frequency: tuple[Mapping[str, Any], ...]
    selected_index_frequency: tuple[Mapping[str, Any], ...]
    artifact_overlap: tuple[tuple[float, ...], ...]
    semantic_overlap: Mapping[str, Any]
    value_overlap: Mapping[str, Any]
    config: Mapping[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "artifact_count": int(self.artifact_count),
            "atom_count": int(self.atom_count),
            "consensus_terms": [dict(row) for row in self.consensus_terms],
            "expression_frequency": [dict(row) for row in self.expression_frequency],
            "selected_index_frequency": [dict(row) for row in self.selected_index_frequency],
            "artifact_overlap": [list(row) for row in self.artifact_overlap],
            "semantic_overlap": dict(self.semantic_overlap),
            "value_overlap": dict(self.value_overlap),
            "config": dict(self.config),
        }


class SymbolicBasisConsensusAnalyzer:
    """Artifact/report capability for basis consensus and overlap."""

    name = "symbolic_basis_consensus_analyzer"
    context_requires = ("artifact.symbolic_basis_ref",)
    context_optional = ("data.X_train", "basis.metrics")
    context_provides = ("basis.consensus", "basis.overlap_report", "artifact.report")
    context_mutates = ()
    context_cache = ()
    requires_metrics = ()
    metrics_fallback = "strict"
    context_notes = "Builds consensus and overlap reports from Stage 1 symbolic basis artifacts."

    def __init__(self, config: BasisConsensusConfig | None = None) -> None:
        self.config = config or BasisConsensusConfig()

    def analyze(
        self,
        artifacts: Sequence[OrthogonalBasisSetArtifact],
        *,
        X: np.ndarray | None = None,
    ) -> BasisConsensusReport:
        items = tuple(artifacts or ())
        expression_rows: dict[str, dict[str, Any]] = {}
        index_rows: dict[str, dict[str, Any]] = {}
        artifact_key_sets: list[set[str]] = []
        atom_count = 0
        for artifact_pos, artifact in enumerate(items):
            artifact_id = str(artifact.artifact_id or artifact.name or artifact_pos)
            keys: set[str] = set()
            for atom_pos, term in enumerate(artifact.basis_genome):
                expr = dict(term.get("expr", {}) or {})
                expr_key = _expr_key(expr)
                keys.add(expr_key)
                atom_count += 1
                row = expression_rows.setdefault(
                    expr_key,
                    {
                        "expression": expr_key,
                        "support": 0,
                        "artifact_ids": [],
                        "atom_positions": [],
                        "selected_indices": [],
                    },
                )
                row["support"] = int(row["support"]) + 1
                row["artifact_ids"].append(artifact_id)
                row["atom_positions"].append(int(atom_pos))
                if atom_pos < len(artifact.selected_indices):
                    row["selected_indices"].append(int(artifact.selected_indices[atom_pos]))
                    idx_key = str(int(artifact.selected_indices[atom_pos]))
                    idx_row = index_rows.setdefault(idx_key, {"selected_index": int(idx_key), "support": 0, "artifact_ids": []})
                    idx_row["support"] = int(idx_row["support"]) + 1
                    idx_row["artifact_ids"].append(artifact_id)
            artifact_key_sets.append(keys)

        artifact_count = int(len(items))
        min_support = max(1, int(np.ceil(float(self.config.min_support_ratio) * max(1, artifact_count))))
        expression_frequency = tuple(
            _with_support_ratio(row, artifact_count)
            for row in sorted(expression_rows.values(), key=lambda row: (-int(row["support"]), str(row["expression"])))
        )[: max(0, int(self.config.max_report_terms))]
        selected_index_frequency = tuple(
            _with_support_ratio(row, artifact_count)
            for row in sorted(index_rows.values(), key=lambda row: (-int(row["support"]), int(row["selected_index"])))
        )[: max(0, int(self.config.max_report_terms))]
        consensus_terms = tuple(row for row in expression_frequency if int(row["support"]) >= min_support)
        artifact_overlap = _artifact_overlap_matrix(artifact_key_sets)
        semantic_overlap = _semantic_overlap_report(items, max_report_terms=int(self.config.max_report_terms))
        value_overlap = (
            _value_overlap_report(
                items,
                X,
                max_report_terms=int(self.config.max_report_terms),
                high_overlap_threshold=float(self.config.high_value_overlap_threshold),
            )
            if bool(self.config.include_value_overlap) and X is not None and len(items) > 0
            else {"enabled": bool(self.config.include_value_overlap), "available": False}
        )
        return BasisConsensusReport(
            artifact_count=artifact_count,
            atom_count=int(atom_count),
            consensus_terms=consensus_terms,
            expression_frequency=expression_frequency,
            selected_index_frequency=selected_index_frequency,
            artifact_overlap=artifact_overlap,
            semantic_overlap=semantic_overlap,
            value_overlap=value_overlap,
            config={
                "min_support_ratio": float(self.config.min_support_ratio),
                "include_value_overlap": bool(self.config.include_value_overlap),
                "max_report_terms": int(self.config.max_report_terms),
                "high_value_overlap_threshold": float(self.config.high_value_overlap_threshold),
                "metadata": dict(self.config.metadata),
            },
        )

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "config": {
                "min_support_ratio": float(self.config.min_support_ratio),
                "include_value_overlap": bool(self.config.include_value_overlap),
                "max_report_terms": int(self.config.max_report_terms),
                "high_value_overlap_threshold": float(self.config.high_value_overlap_threshold),
                "metadata": dict(self.config.metadata),
            },
        }


def _expr_key(expr: Mapping[str, Any]) -> str:
    try:
        from mlblack.models.symbolic_normalization import expression_equivalence_key

        return expression_equivalence_key(expr)
    except Exception:
        return str(dict(expr))


def _with_support_ratio(row: Mapping[str, Any], artifact_count: int) -> dict[str, Any]:
    out = dict(row)
    out["support_ratio"] = float(int(out.get("support", 0)) / float(max(1, artifact_count)))
    out["artifact_ids"] = sorted(set(str(v) for v in out.get("artifact_ids", [])))
    if "selected_indices" in out:
        out["selected_indices"] = sorted(set(int(v) for v in out.get("selected_indices", [])))
    return out


def _artifact_overlap_matrix(rows: Sequence[set[str]]) -> tuple[tuple[float, ...], ...]:
    matrix: list[tuple[float, ...]] = []
    for left in rows:
        row: list[float] = []
        for right in rows:
            union = left | right
            row.append(1.0 if not union else float(len(left & right) / len(union)))
        matrix.append(tuple(row))
    return tuple(matrix)


def _semantic_overlap_report(artifacts: Sequence[OrthogonalBasisSetArtifact], *, max_report_terms: int) -> dict[str, Any]:
    family_rows: dict[str, dict[str, Any]] = {}
    feature_rows: dict[str, dict[str, Any]] = {}
    artifact_family_sets: list[set[str]] = []
    artifact_feature_sets: list[set[str]] = []
    for artifact_index, artifact in enumerate(artifacts):
        artifact_id = str(artifact.artifact_id or artifact.name or artifact_index)
        families: set[str] = set()
        features: set[str] = set()
        for term in tuple(artifact.selected_terms):
            family = str(term.get("family", term.get("activation_family", "")) or "")
            if family:
                families.add(family)
                row = family_rows.setdefault(family, {"family": family, "support": 0, "artifact_ids": []})
                row["support"] = int(row["support"]) + 1
                row["artifact_ids"].append(artifact_id)
            for feature in tuple(term.get("features", ()) or ()):
                feature_key = str(int(feature))
                features.add(feature_key)
                row = feature_rows.setdefault(feature_key, {"feature": int(feature), "support": 0, "artifact_ids": []})
                row["support"] = int(row["support"]) + 1
                row["artifact_ids"].append(artifact_id)
        artifact_family_sets.append(families)
        artifact_feature_sets.append(features)
    artifact_count = int(len(tuple(artifacts)))
    family_frequency = tuple(
        _with_artifact_support(row, artifact_count)
        for row in sorted(family_rows.values(), key=lambda row: (-int(row["support"]), str(row["family"])))
    )[: max(0, int(max_report_terms))]
    feature_frequency = tuple(
        _with_artifact_support(row, artifact_count)
        for row in sorted(feature_rows.values(), key=lambda row: (-int(row["support"]), int(row["feature"])))
    )[: max(0, int(max_report_terms))]
    return {
        "family_frequency": [dict(row) for row in family_frequency],
        "feature_frequency": [dict(row) for row in feature_frequency],
        "artifact_family_overlap": [list(row) for row in _artifact_overlap_matrix(artifact_family_sets)],
        "artifact_feature_overlap": [list(row) for row in _artifact_overlap_matrix(artifact_feature_sets)],
    }


def _with_artifact_support(row: Mapping[str, Any], artifact_count: int) -> dict[str, Any]:
    out = dict(row)
    artifact_ids = sorted(set(str(v) for v in out.get("artifact_ids", [])))
    out["artifact_ids"] = artifact_ids
    out["artifact_support"] = int(len(artifact_ids))
    out["support_ratio"] = float(len(artifact_ids) / float(max(1, artifact_count)))
    return out


def _value_overlap_report(
    artifacts: Sequence[OrthogonalBasisSetArtifact],
    X: np.ndarray | None,
    *,
    max_report_terms: int,
    high_overlap_threshold: float,
) -> dict[str, Any]:
    if X is None:
        return {"enabled": True, "available": False}
    x = np.asarray(X, dtype=float)
    summaries: list[dict[str, Any]] = []
    atom_rows: list[dict[str, Any]] = []
    atom_values: list[np.ndarray] = []
    max_cross_abs_corr = 0.0
    for idx, artifact in enumerate(artifacts):
        try:
            Z = np.asarray(artifact.transform(x), dtype=float)
        except Exception as exc:
            summaries.append({"artifact_index": int(idx), "available": False, "error": repr(exc)})
            continue
        names = artifact.basis_feature_names()
        for atom_idx in range(int(Z.shape[1])):
            values = np.asarray(Z[:, atom_idx], dtype=float).reshape(-1)
            atom_values.append(values)
            atom_rows.append(
                {
                    "artifact_index": int(idx),
                    "artifact_id": str(artifact.artifact_id or artifact.name or idx),
                    "atom_index": int(atom_idx),
                    "name": str(names[atom_idx]) if atom_idx < len(names) else f"basis_{atom_idx}",
                }
            )
        corr = _corr_matrix(Z)
        off_diag = corr - np.eye(corr.shape[0])
        max_abs_corr = float(np.max(np.abs(off_diag))) if off_diag.size else 0.0
        max_cross_abs_corr = max(max_cross_abs_corr, max_abs_corr)
        summaries.append(
            {
                "artifact_index": int(idx),
                "available": True,
                "basis_count": int(Z.shape[1]),
                "rank": int(np.linalg.matrix_rank(Z - np.mean(Z, axis=0, keepdims=True))),
                "max_abs_corr": max_abs_corr,
                "mean_abs_corr": float(np.mean(np.abs(off_diag))) if off_diag.size else 0.0,
            }
        )
    atom_matrix, high_pairs = _atom_overlap(atom_values, atom_rows, threshold=float(high_overlap_threshold), max_pairs=int(max_report_terms))
    if atom_matrix.size:
        matrix_without_diag = atom_matrix - np.eye(atom_matrix.shape[0])
        max_cross_abs_corr = max(max_cross_abs_corr, float(np.max(np.abs(matrix_without_diag))) if matrix_without_diag.size else 0.0)
    return {
        "enabled": True,
        "available": True,
        "max_cross_abs_corr": float(max_cross_abs_corr),
        "artifact_summaries": summaries,
        "atom_index": atom_rows[: max(0, int(max_report_terms))],
        "atom_overlap_matrix": atom_matrix.tolist(),
        "high_overlap_pairs": high_pairs,
        "high_value_overlap_threshold": float(high_overlap_threshold),
    }


def _corr_matrix(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2:
        matrix = matrix.reshape(matrix.shape[0], -1)
    if matrix.shape[1] <= 1:
        return np.eye(max(1, matrix.shape[1]))
    centered = matrix - np.mean(matrix, axis=0, keepdims=True)
    cols = []
    for idx in range(centered.shape[1]):
        col = centered[:, idx]
        denom = float(np.linalg.norm(col))
        cols.append(col / denom if denom > 1e-12 else np.zeros_like(col))
    normalized = np.column_stack(cols)
    corr = normalized.T @ normalized
    return np.asarray(np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0), dtype=float)


def _atom_overlap(
    atom_values: Sequence[np.ndarray],
    atom_rows: Sequence[Mapping[str, Any]],
    *,
    threshold: float,
    max_pairs: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    if not atom_values:
        return np.zeros((0, 0), dtype=float), []
    matrix = np.eye(len(atom_values), dtype=float)
    high_pairs: list[dict[str, Any]] = []
    for i in range(len(atom_values)):
        for j in range(i + 1, len(atom_values)):
            corr = abs(safe_corr(np.asarray(atom_values[i], dtype=float), np.asarray(atom_values[j], dtype=float)))
            matrix[i, j] = corr
            matrix[j, i] = corr
            if corr >= float(threshold) and len(high_pairs) < int(max_pairs):
                high_pairs.append(
                    {
                        "left": dict(atom_rows[i]),
                        "right": dict(atom_rows[j]),
                        "abs_corr": float(corr),
                    }
                )
    return matrix, high_pairs


__all__ = ["BasisConsensusConfig", "BasisConsensusReport", "SymbolicBasisConsensusAnalyzer"]
