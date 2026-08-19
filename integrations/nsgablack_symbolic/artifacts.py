from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from typing import Any, Mapping, Sequence

import numpy as np

from mlblack.core.types import UnknownState
from mlblack.models.symbolic import (
    SymbolicBasisSetModel,
    expression_complexity,
    expression_to_string,
    feature_expr,
    parameterize_expression,
)
from mlblack.models.symbolic_normalization import expression_equivalence_key, expression_family_signature
from mlblack.pipeline.data_views import NumericDataView
from mlblack.pipeline.symbolic import CandidateTerm, FunctionPool, safe_corr
from mlblack.representations import SymbolicBasisSetConfig, SymbolicBasisSetRepresentation


SYMBOLIC_ARTIFACT_SCHEMA_NAME = "symbolic_artifact"
SYMBOLIC_ARTIFACT_SCHEMA_KEY = "symbolic_artifact_v2"
SYMBOLIC_ARTIFACT_SCHEMA_VERSION = 2
SYMBOLIC_ARTIFACT_FIELDS: tuple[str, ...] = (
    "final_expression",
    "normalized_expression",
    "canonical_expression",
    "feature_usage",
    "term_contributions",
    "complexity_metrics",
    "stability_metrics",
    "candidate_lineage",
    "simplification_trace",
    "truth_contract_recovery",
    "family_recovery",
    "phase_equivalence_recovery",
    "orthogonal_search_objective",
    "heterogeneous_lane_consensus",
    "equivalence_expression_handling",
    "interference_feature_handling",
    "periodic_equivalence_disambiguation",
    "regional_correction_basis",
    "head_semantics",
    "regime_structure",
    "basis_structure",
    "assembler_structure",
    "piecewise_gate_basis",
    "basis_consensus",
    "basis_overlap_report",
    "search_policy_report",
    "evaluation_report",
    "branch_report",
)


@dataclass(frozen=True)
class SymbolicArtifactSchema:
    """Lightweight typed symbolic artifact surface.

    The old repository had a very wide symbolic artifact schema. This new
    object preserves that boundary without making trainer/problem layers carry
    private debug dictionaries.
    """

    payload: Mapping[str, Any] = field(default_factory=dict)
    schema_name: str = SYMBOLIC_ARTIFACT_SCHEMA_NAME
    schema_key: str = SYMBOLIC_ARTIFACT_SCHEMA_KEY
    schema_version: int = SYMBOLIC_ARTIFACT_SCHEMA_VERSION

    def as_dict(self) -> dict[str, Any]:
        payload = _jsonable_mapping(self.payload)
        return {
            "schema_name": str(self.schema_name),
            "schema_key": str(self.schema_key),
            "schema_version": int(self.schema_version),
            "created_at": str(payload.get("created_at") or _utc_now()),
            **payload,
        }


@dataclass(frozen=True)
class OrthogonalBasisSetArtifact:
    """Stage 1 output boundary consumed by Stage 2."""

    name: str
    input_dim: int
    feature_names: tuple[str, ...]
    basis_genome: tuple[dict[str, Any], ...]
    fitted_state: tuple[float, ...]
    selected_indices: tuple[int, ...] = tuple()
    selected_terms: tuple[dict[str, Any], ...] = tuple()
    metrics: Mapping[str, Any] = field(default_factory=dict)
    objectives: tuple[float, ...] = tuple()
    constraints: tuple[float, ...] = tuple()
    source_record: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    artifact_id: str = ""
    created_at: str = ""

    @classmethod
    def from_stage1_record(
        cls,
        record: Any,
        *,
        input_dim: int,
        feature_names: Sequence[str],
        parameterize_terms: bool = True,
        name: str = "orthogonal_basis_set",
        metadata: Mapping[str, Any] | None = None,
    ) -> "OrthogonalBasisSetArtifact":
        selected_terms = tuple(dict(term) for term in getattr(record, "selected_terms", tuple()))
        genome: list[dict[str, Any]] = []
        for pos, term in enumerate(selected_terms):
            expr = dict(term.get("expr", {}))
            if bool(parameterize_terms):
                expr = parameterize_expression(expr, prefix=f"basis{pos}")
            genome.append({"name": f"basis_{pos}_{term.get('name', pos)}", "expr": expr})
        payload_for_id = {
            "type": "orthogonal_basis_set",
            "selected_indices": list(getattr(record, "selected_indices", tuple())),
            "objectives": list(getattr(record, "objectives", tuple())),
            "basis_genome": genome,
        }
        return cls(
            name=str(name),
            input_dim=int(input_dim),
            feature_names=tuple(str(v) for v in feature_names),
            basis_genome=tuple(genome),
            fitted_state=tuple(float(v) for v in getattr(record, "fitted_state", tuple())),
            selected_indices=tuple(int(v) for v in getattr(record, "selected_indices", tuple())),
            selected_terms=selected_terms,
            metrics=dict(getattr(record, "metrics", {}) or {}),
            objectives=tuple(float(v) for v in getattr(record, "objectives", tuple())),
            constraints=tuple(float(v) for v in getattr(record, "constraints", tuple())),
            source_record=record.as_dict() if hasattr(record, "as_dict") else {},
            metadata=dict(metadata or {}),
            artifact_id=_artifact_id("orthogonal_basis_set", payload_for_id),
            created_at=_utc_now(),
        )

    def model(self) -> SymbolicBasisSetModel:
        representation = SymbolicBasisSetRepresentation(
            SymbolicBasisSetConfig(
                input_dim=int(self.input_dim),
                genome=tuple(self.basis_genome),
                name=str(self.name),
                feature_names=tuple(self.feature_names),
            )
        )
        return representation.decode(UnknownState(values=np.asarray(self.fitted_state, dtype=float)))

    def transform(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self.model().transform(X), dtype=float)

    def basis_feature_names(self) -> tuple[str, ...]:
        model = self.model()
        names: list[str] = []
        for index, expr in enumerate(model.expression_strings()):
            text = str(expr).strip() or f"basis_{index}"
            names.append(f"basis_{index}:{text}")
        return tuple(names)

    def to_basis_data_view(self, data: NumericDataView) -> NumericDataView:
        train = self.transform(data.X_train)
        valid = None if data.X_valid is None else self.transform(data.X_valid)
        return NumericDataView(
            X_train=train,
            y_train=data.y_train,
            X_valid=valid,
            y_valid=data.y_valid,
            feature_names=self.basis_feature_names(),
            target_name=data.target_name,
            metadata={
                **dict(data.metadata),
                "basis_artifact": self.describe(include_record=False),
                "source_feature_names": list(data.effective_feature_names),
            },
        )

    def as_function_pool(self, X: np.ndarray, *, y: np.ndarray | None = None) -> FunctionPool:
        Z = self.transform(X)
        target = np.asarray(y, dtype=float).reshape(-1) if y is not None else np.std(Z, axis=1).reshape(-1)
        names = self.basis_feature_names()
        terms: list[CandidateTerm] = []
        for idx, name in enumerate(names):
            values = np.asarray(Z[:, idx], dtype=float).reshape(-1)
            expr = feature_expr(idx)
            terms.append(
                CandidateTerm(
                    name=str(name),
                    expr=expr,
                    values=values,
                    complexity=float(expression_complexity(self.basis_genome[idx]["expr"])),
                    family="basis_atom",
                    activation_family="basis_atom",
                    features=(int(idx),),
                    prior_corr=abs(safe_corr(values, target)),
                    metadata={
                        "source": "orthogonal_basis_artifact",
                        "selected_index": self.selected_indices[idx] if idx < len(self.selected_indices) else idx,
                    },
                )
            )
        return FunctionPool(tuple(terms), metadata={"source": "orthogonal_basis_artifact", "basis_name": self.name})

    def schema(self) -> SymbolicArtifactSchema:
        model = self.model()
        expressions = list(model.expression_strings())
        expression_map = {f"basis_{idx}": expr for idx, expr in enumerate(expressions)}
        normalized_map = _normalized_expression_map(self.metadata, fallback=expression_map)
        return SymbolicArtifactSchema(
            {
                "family": "symbolic",
                "artifact_type": "orthogonal_basis_set",
                "artifact_id": self.artifact_id or _artifact_id("orthogonal_basis_set", self.describe(include_record=False, include_schema=False)),
                "created_at": self.created_at or _utc_now(),
                "name": self.name,
                "stage": {
                    "name": "orthogonal_basis_search",
                    "outer_owner": "nsgablack",
                    "inner_owner": "mlblack",
                },
                "final_expression": expression_map,
                "normalized_expression": normalized_map,
                "canonical_expression": _canonical_expression_map(self.metadata),
                "feature_usage": _feature_usage_from_terms(self.selected_terms, feature_names=self.feature_names),
                "term_contributions": _term_contributions_from_terms(self.selected_terms, feature_names=self.feature_names),
                "basis_structure": {
                    "source": "nsgablack_symbolic.stage1",
                    "basis_count": int(len(self.basis_genome)),
                    "basis_terms": [
                        {
                            "name": str(term.get("name", f"basis_{idx}")),
                            "expression": expressions[idx] if idx < len(expressions) else expression_to_string(term["expr"]),
                            "selected_index": self.selected_indices[idx] if idx < len(self.selected_indices) else idx,
                        }
                        for idx, term in enumerate(self.basis_genome)
                    ],
                },
                "regime_structure": _default_regime_structure(stage="orthogonal_basis_search"),
                "assembler_structure": _default_assembler_structure(
                    stage="orthogonal_basis_search",
                    output_expression_count=int(len(self.basis_genome)),
                    basis_conditioned=False,
                    search_input_space="raw_feature_function_pool",
                ),
                "piecewise_gate_basis": _metadata_section(self.metadata, "piecewise_gate_basis", status="not_applicable"),
                "head_semantics": {
                    "output_kind": "basis_set",
                    "head_kind": "symbolic_basis_set",
                    "stage": "orthogonal_basis_search",
                },
                "parameterization": {
                    "parameter_count": int(len(self.fitted_state)),
                    "fitted_state": list(self.fitted_state),
                },
                "complexity_metrics": {
                    "expression_size": float(sum(expression_complexity(term["expr"]) for term in self.basis_genome)),
                    "term_count": int(len(self.basis_genome)),
                    "parameter_count": int(len(self.fitted_state)),
                },
                "stability_metrics": dict(self.metrics),
                "candidate_lineage": {
                    "selected_indices": list(self.selected_indices),
                    "selected_terms": [dict(term) for term in self.selected_terms],
                    "replay_record": dict(self.metadata.get("replay_record", {}))
                    if isinstance(self.metadata.get("replay_record"), Mapping)
                    else {},
                },
                "orthogonal_search_objective": {
                    "objectives": list(self.objectives),
                    "constraints": list(self.constraints),
                    "metrics": dict(self.metrics),
                },
                "heterogeneous_lane_consensus": _metadata_section(self.metadata, "heterogeneous_lane_consensus"),
                "equivalence_expression_handling": _metadata_section(self.metadata, "equivalence_expression_handling"),
                "interference_feature_handling": _metadata_section(self.metadata, "interference_feature_handling"),
                "periodic_equivalence_disambiguation": _metadata_section(self.metadata, "periodic_equivalence_disambiguation"),
                "regional_correction_basis": _metadata_section(self.metadata, "regional_correction_basis"),
                "basis_consensus": dict(self.metadata.get("basis_consensus", {}))
                if isinstance(self.metadata.get("basis_consensus"), Mapping)
                else {},
                "basis_overlap_report": dict(self.metadata.get("basis_overlap_report", {}))
                if isinstance(self.metadata.get("basis_overlap_report"), Mapping)
                else {},
                "search_policy_report": dict(self.metadata.get("candidate_score", {}))
                if isinstance(self.metadata.get("candidate_score"), Mapping)
                else {},
                "evaluation_report": dict(self.metadata.get("fold_report", {}))
                if isinstance(self.metadata.get("fold_report"), Mapping)
                else {},
                "branch_report": dict(self.metadata.get("branch_report", {}))
                if isinstance(self.metadata.get("branch_report"), Mapping)
                else {},
                "simplification_trace": list(self.metadata.get("simplification_trace", []))
                if isinstance(self.metadata.get("simplification_trace", []), Sequence)
                and not isinstance(self.metadata.get("simplification_trace", []), (str, bytes, bytearray))
                else [],
                "truth_contract_recovery": dict(self.metadata.get("truth_contract_recovery", {}))
                if isinstance(self.metadata.get("truth_contract_recovery"), Mapping)
                else {},
                "family_recovery": _family_recovery_section(self.metadata),
                "phase_equivalence_recovery": _phase_equivalence_recovery_section(self.metadata),
                "metadata": dict(self.metadata),
            }
        )

    def describe(self, *, include_record: bool = True, include_schema: bool = True) -> dict[str, Any]:
        out = {
            "name": self.name,
            "artifact_id": self.artifact_id,
            "created_at": self.created_at,
            "input_dim": int(self.input_dim),
            "feature_names": list(self.feature_names),
            "basis_genome": [dict(term) for term in self.basis_genome],
            "fitted_state": list(self.fitted_state),
            "selected_indices": list(self.selected_indices),
            "selected_terms": [dict(term) for term in self.selected_terms],
            "metrics": dict(self.metrics),
            "objectives": list(self.objectives),
            "constraints": list(self.constraints),
            "metadata": dict(self.metadata),
        }
        if include_schema:
            out["schema"] = self.schema().as_dict()
        if include_record:
            out["source_record"] = dict(self.source_record)
        return out


@dataclass(frozen=True)
class SymbolicTaskArtifact:
    """Stage 2 task-expression artifact."""

    name: str
    expression: dict[str, Any]
    fitted_state: tuple[float, ...]
    metrics: Mapping[str, Any]
    objectives: tuple[float, ...]
    constraints: tuple[float, ...]
    selected_indices: tuple[int, ...] = tuple()
    selected_terms: tuple[dict[str, Any], ...] = tuple()
    task_kind: str = "regression"
    head_kind: str = "point"
    objective_names: tuple[str, ...] = tuple()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    artifact_id: str = ""
    created_at: str = ""

    def schema(self) -> SymbolicArtifactSchema:
        artifact_id = self.artifact_id or _artifact_id(
            "basis_conditioned_task_expression",
            {
                "expression": self.expression,
                "selected_indices": list(self.selected_indices),
                "objectives": list(self.objectives),
                "task_kind": self.task_kind,
                "head_kind": self.head_kind,
            },
        )
        return SymbolicArtifactSchema(
            {
                "family": "symbolic",
                "artifact_type": "basis_conditioned_task_expression",
                "artifact_id": artifact_id,
                "created_at": self.created_at or _utc_now(),
                "name": self.name,
                "stage": {
                    "name": "basis_conditioned_symbolic_task",
                    "outer_owner": "nsgablack",
                    "inner_owner": "mlblack",
                },
                "task_semantics": {
                    "task_kind": str(self.task_kind),
                    "objective_names": list(self.objective_names),
                    "objectives": list(self.objectives),
                    "constraints": list(self.constraints),
                },
                "final_expression": {"target": expression_to_string(self.expression)},
                "normalized_expression": _normalized_expression_map(
                    self.metadata,
                    fallback={"target": expression_to_string(self.expression)},
                ),
                "canonical_expression": _canonical_expression_map(
                    self.metadata,
                    fallback_expressions={"target": self.expression},
                ),
                "feature_usage": _feature_usage_from_terms(self.selected_terms),
                "term_contributions": _term_contributions_from_terms(self.selected_terms),
                "head_semantics": {
                    "output_kind": _head_output_kind(self.head_kind),
                    "head_kind": str(self.head_kind),
                    "stage": "basis_conditioned_symbolic_task",
                },
                "parameterization": {
                    "parameter_count": int(len(self.fitted_state)),
                    "fitted_state": list(self.fitted_state),
                },
                "complexity_metrics": {
                    "expression_size": float(expression_complexity(self.expression)),
                    "parameter_count": int(len(self.fitted_state)),
                    "term_count": int(len(self.selected_terms)),
                },
                "stability_metrics": dict(self.metrics),
                "candidate_lineage": {
                    "selected_indices": list(self.selected_indices),
                    "selected_terms": [dict(term) for term in self.selected_terms],
                    "replay_record": dict(self.metadata.get("replay_record", {}))
                    if isinstance(self.metadata.get("replay_record"), Mapping)
                    else {},
                },
                "regime_structure": _metadata_section(self.metadata, "regime_structure", status="not_recorded"),
                "basis_structure": _metadata_section(
                    self.metadata,
                    "basis_structure",
                    status="reported",
                    source="stage1_basis_artifact",
                    basis_artifact_id=dict(self.metadata.get("basis_artifact", {}) or {}).get("artifact_id", ""),
                ),
                "assembler_structure": _default_assembler_structure(
                    stage="basis_conditioned_symbolic_task",
                    output_expression_count=1,
                    basis_conditioned=True,
                    search_input_space="stage1_basis_function_pool",
                ),
                "piecewise_gate_basis": _metadata_section(self.metadata, "piecewise_gate_basis", status="not_applicable"),
                "orthogonal_search_objective": dict(self.metadata.get("basis_metrics", {}))
                if isinstance(self.metadata.get("basis_metrics"), Mapping)
                else {},
                "heterogeneous_lane_consensus": _metadata_section(self.metadata, "heterogeneous_lane_consensus"),
                "equivalence_expression_handling": _metadata_section(self.metadata, "equivalence_expression_handling"),
                "interference_feature_handling": _metadata_section(self.metadata, "interference_feature_handling"),
                "periodic_equivalence_disambiguation": _metadata_section(self.metadata, "periodic_equivalence_disambiguation"),
                "regional_correction_basis": _metadata_section(self.metadata, "regional_correction_basis"),
                "basis_consensus": dict(self.metadata.get("basis_consensus", {}))
                if isinstance(self.metadata.get("basis_consensus"), Mapping)
                else {},
                "basis_overlap_report": dict(self.metadata.get("basis_overlap_report", {}))
                if isinstance(self.metadata.get("basis_overlap_report"), Mapping)
                else {},
                "search_policy_report": dict(self.metadata.get("candidate_score", {}))
                if isinstance(self.metadata.get("candidate_score"), Mapping)
                else {},
                "evaluation_report": dict(self.metadata.get("fold_report", {}))
                if isinstance(self.metadata.get("fold_report"), Mapping)
                else {},
                "branch_report": dict(self.metadata.get("branch_report", {}))
                if isinstance(self.metadata.get("branch_report"), Mapping)
                else {},
                "simplification_trace": list(self.metadata.get("simplification_trace", []))
                if isinstance(self.metadata.get("simplification_trace", []), Sequence)
                and not isinstance(self.metadata.get("simplification_trace", []), (str, bytes, bytearray))
                else [],
                "truth_contract_recovery": dict(self.metadata.get("truth_contract_recovery", {}))
                if isinstance(self.metadata.get("truth_contract_recovery"), Mapping)
                else {},
                "family_recovery": _family_recovery_section(self.metadata),
                "phase_equivalence_recovery": _phase_equivalence_recovery_section(self.metadata),
                "metadata": dict(self.metadata),
            }
        )

    def describe(self) -> dict[str, Any]:
        artifact_id = self.artifact_id or _artifact_id(
            "basis_conditioned_task_expression",
            {
                "expression": self.expression,
                "selected_indices": list(self.selected_indices),
                "objectives": list(self.objectives),
                "task_kind": self.task_kind,
                "head_kind": self.head_kind,
            },
        )
        return {
            "name": self.name,
            "artifact_id": artifact_id,
            "created_at": self.created_at or _utc_now(),
            "task_kind": self.task_kind,
            "head_kind": self.head_kind,
            "objective_names": list(self.objective_names),
            "expression": dict(self.expression),
            "expression_string": expression_to_string(self.expression),
            "fitted_state": list(self.fitted_state),
            "metrics": dict(self.metrics),
            "objectives": list(self.objectives),
            "constraints": list(self.constraints),
            "selected_indices": list(self.selected_indices),
            "selected_terms": [dict(term) for term in self.selected_terms],
            "schema": self.schema().as_dict(),
            "metadata": dict(self.metadata),
        }


def _jsonable_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): _jsonable(item) for key, item in dict(value).items()}


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return _jsonable_mapping(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_jsonable(item) for item in value]
    return str(value)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _artifact_id(kind: str, payload: Mapping[str, Any]) -> str:
    blob = json.dumps(_jsonable_mapping(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return f"sym:{kind}:{hashlib.sha1(blob.encode('utf-8')).hexdigest()[:16]}"


def _head_output_kind(head_kind: str) -> str:
    key = str(head_kind or "point").strip().lower()
    if key.startswith("interval"):
        return "interval"
    if key in {"binary_logistic", "softmax", "probability_calibration", "probability"}:
        return "probability"
    return "point"


def _feature_usage_from_terms(
    terms: Sequence[Mapping[str, Any]],
    *,
    feature_names: Sequence[str] = tuple(),
) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for term in tuple(terms or ()):
        for feature in tuple(term.get("features", ()) or ()):
            idx = int(feature)
            name = str(tuple(feature_names)[idx]) if 0 <= idx < len(tuple(feature_names)) else f"x{idx}"
            counts[name] = int(counts.get(name, 0) + 1)
    return {
        "feature_count": int(len(counts)),
        "features": [{"name": key, "count": int(value)} for key, value in sorted(counts.items())],
    }


def _term_contributions_from_terms(
    terms: Sequence[Mapping[str, Any]],
    *,
    feature_names: Sequence[str] = tuple(),
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for idx, term in enumerate(tuple(terms or ())):
        feature_labels: list[str] = []
        for feature in tuple(term.get("features", ()) or ()):
            feature_idx = int(feature)
            feature_labels.append(
                str(tuple(feature_names)[feature_idx]) if 0 <= feature_idx < len(tuple(feature_names)) else f"x{feature_idx}"
            )
        rows.append(
            {
                "term_index": int(idx),
                "name": str(term.get("name", f"term_{idx}")),
                "family": str(term.get("family", term.get("activation_family", "")) or ""),
                "activation_family": str(term.get("activation_family", "")),
                "features": feature_labels,
                "complexity": float(term.get("complexity", 0.0) or 0.0),
                "prior_corr": float(term.get("prior_corr", 0.0) or 0.0),
            }
        )
    return {"terms": rows, "term_count": int(len(rows))}


def _metadata_section(metadata: Mapping[str, Any], key: str, **defaults: Any) -> dict[str, Any]:
    raw = dict(metadata or {}).get(str(key))
    if isinstance(raw, Mapping):
        return {**{str(k): _jsonable(v) for k, v in defaults.items()}, **_jsonable_mapping(raw)}
    return {"status": "not_recorded", **{str(k): _jsonable(v) for k, v in defaults.items()}}


def _normalized_expression_map(metadata: Mapping[str, Any], *, fallback: Mapping[str, Any]) -> dict[str, Any]:
    raw = dict(metadata or {}).get("simplified_expressions")
    if not isinstance(raw, Mapping):
        return {str(k): _jsonable(v) for k, v in dict(fallback).items()}
    out: dict[str, Any] = {}
    for key, payload in dict(raw).items():
        if isinstance(payload, Mapping) and payload.get("expression_string") is not None:
            out[str(key)] = str(payload.get("expression_string"))
        else:
            out[str(key)] = _jsonable(payload)
    return out or {str(k): _jsonable(v) for k, v in dict(fallback).items()}


def _canonical_expression_map(
    metadata: Mapping[str, Any],
    *,
    fallback_expressions: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    raw = dict(metadata or {}).get("simplified_expressions")
    out: dict[str, Any] = {}
    if isinstance(raw, Mapping):
        for key, payload in dict(raw).items():
            row = dict(payload) if isinstance(payload, Mapping) else {}
            expr = dict(row.get("expression", {}) or {}) if isinstance(row.get("expression"), Mapping) else {}
            if expr:
                signature = dict(row.get("family_signature", {}) or {}) or expression_family_signature(expr)
                out[str(key)] = {
                    "canonical_key": str(row.get("canonical_key") or expression_equivalence_key(expr)),
                    "canonical_expression": str(row.get("canonical_expression") or row.get("expression_string") or ""),
                    "family_signature": signature,
                }
    for key, expr in dict(fallback_expressions or {}).items():
        if str(key) in out:
            continue
        signature = expression_family_signature(dict(expr))
        out[str(key)] = {
            "canonical_key": expression_equivalence_key(dict(expr)),
            "canonical_expression": str(signature.get("canonical_expression", "")),
            "family_signature": signature,
        }
    return out


def _family_recovery_section(metadata: Mapping[str, Any]) -> dict[str, Any]:
    truth = dict(metadata.get("truth_contract_recovery", {})) if isinstance(metadata.get("truth_contract_recovery"), Mapping) else {}
    family = dict(truth.get("family_recovery", {})) if isinstance(truth.get("family_recovery"), Mapping) else {}
    if family:
        return family
    if truth:
        return {
            "status": str(truth.get("status", "reported")),
            "family_recovery_score": float(truth.get("family_recovery_score", 0.0) or 0.0),
            "family_matched_contract_count": int(truth.get("family_matched_contract_count", 0) or 0),
        }
    return {"status": "not_recorded"}


def _phase_equivalence_recovery_section(metadata: Mapping[str, Any]) -> dict[str, Any]:
    truth = dict(metadata.get("truth_contract_recovery", {})) if isinstance(metadata.get("truth_contract_recovery"), Mapping) else {}
    periodic = (
        dict(metadata.get("periodic_equivalence_disambiguation", {}))
        if isinstance(metadata.get("periodic_equivalence_disambiguation"), Mapping)
        else {}
    )
    if truth or periodic:
        return {
            "status": str(truth.get("status", periodic.get("status", "reported"))),
            "phase_equivalence_recovery_score": float(truth.get("phase_equivalence_recovery_score", 0.0) or 0.0),
            "phase_equivalent_contract_count": int(truth.get("phase_equivalent_contract_count", 0) or 0),
            "phase_equivalence_policy": str(periodic.get("phase_equivalence_policy", "scored")),
            "periodic_term_count": int(periodic.get("periodic_term_count", 0) or 0),
        }
    return {"status": "not_recorded"}


def _default_regime_structure(*, stage: str) -> dict[str, Any]:
    return {
        "status": "not_applicable",
        "source": "nsgablack_symbolic",
        "mode": "global",
        "piecewise_enabled": False,
        "stage": str(stage),
    }


def _default_assembler_structure(
    *,
    stage: str,
    output_expression_count: int,
    basis_conditioned: bool,
    search_input_space: str,
) -> dict[str, Any]:
    return {
        "status": "reported",
        "source": "nsgablack_symbolic",
        "stage_protocol": str(stage),
        "assembler_mode": str(stage),
        "basis_conditioned": bool(basis_conditioned),
        "output_expression_count": int(output_expression_count),
        "search_input_space": str(search_input_space),
        "structure_head": "symbolic",
        "prediction_head": "deferred_to_head_semantics",
    }


def symbolic_artifact_schema_descriptor() -> dict[str, Any]:
    return {
        "schema_name": SYMBOLIC_ARTIFACT_SCHEMA_NAME,
        "schema_key": SYMBOLIC_ARTIFACT_SCHEMA_KEY,
        "schema_version": int(SYMBOLIC_ARTIFACT_SCHEMA_VERSION),
        "fields": list(SYMBOLIC_ARTIFACT_FIELDS),
    }


__all__ = [
    "OrthogonalBasisSetArtifact",
    "SYMBOLIC_ARTIFACT_SCHEMA_KEY",
    "SYMBOLIC_ARTIFACT_SCHEMA_NAME",
    "SYMBOLIC_ARTIFACT_SCHEMA_VERSION",
    "SYMBOLIC_ARTIFACT_FIELDS",
    "SymbolicArtifactSchema",
    "SymbolicTaskArtifact",
    "symbolic_artifact_schema_descriptor",
]
