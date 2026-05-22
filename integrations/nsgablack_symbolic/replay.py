from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class SymbolicCandidateReplayRecord:
    replay_id: str
    stage: str
    problem_name: str
    outer_candidate: tuple[float, ...]
    selected_indices: tuple[int, ...]
    selected_terms: tuple[Mapping[str, Any], ...]
    fitted_state: tuple[float, ...]
    objectives: tuple[float, ...]
    constraints: tuple[float, ...]
    metrics: Mapping[str, Any]
    signals: Mapping[str, Any]
    audit: Mapping[str, Any]
    replay_inputs: Mapping[str, Any]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "replay_id": str(self.replay_id),
            "stage": str(self.stage),
            "problem_name": str(self.problem_name),
            "outer_candidate": [float(v) for v in self.outer_candidate],
            "selected_indices": [int(v) for v in self.selected_indices],
            "selected_terms": [dict(row) for row in self.selected_terms],
            "fitted_state": [float(v) for v in self.fitted_state],
            "objectives": [float(v) for v in self.objectives],
            "constraints": [float(v) for v in self.constraints],
            "metrics": _jsonable_mapping(self.metrics),
            "signals": _jsonable_mapping(self.signals),
            "audit": _jsonable_mapping(self.audit),
            "replay_inputs": _jsonable_mapping(self.replay_inputs),
            "metadata": _jsonable_mapping(self.metadata),
        }


class SymbolicReplayRecordBuilder:
    """Builds JSON-compatible replay records for nested symbolic candidates."""

    name = "symbolic_replay_record_builder"
    context_requires = ("symbolic.candidate_score",)
    context_optional = ("symbolic.graph_cache", "symbolic.path_memory", "resource.context", "stage.audit")
    context_provides = ("symbolic.replay_record", "symbolic.candidate_lineage", "artifact.report")
    context_mutates = ()
    context_cache = ()
    requires_metrics = ()
    metrics_fallback = "strict"
    context_notes = "Compresses a nested symbolic evaluation record into a stable JSON replay/audit payload."

    def build(
        self,
        record: Any,
        *,
        stage: str,
        problem_name: str,
        resource_context: Mapping[str, Any] | None = None,
        extra_inputs: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> SymbolicCandidateReplayRecord:
        selected_terms = tuple(_term_summary(row) for row in tuple(getattr(record, "selected_terms", tuple()) or ()))
        audit = _audit_summary(getattr(record, "report", {}) or {})
        replay_inputs = {
            "selected_indices": [int(v) for v in tuple(getattr(record, "selected_indices", tuple()) or ())],
            "fitted_state": [float(v) for v in tuple(getattr(record, "fitted_state", tuple()) or ())],
            "resource_context": dict(resource_context or {}),
            **dict(extra_inputs or {}),
        }
        payload_for_id = {
            "stage": str(stage),
            "problem_name": str(problem_name),
            "outer_candidate": [float(v) for v in tuple(getattr(record, "outer_candidate", tuple()) or ())],
            "selected_indices": replay_inputs["selected_indices"],
            "objectives": [float(v) for v in tuple(getattr(record, "objectives", tuple()) or ())],
        }
        return SymbolicCandidateReplayRecord(
            replay_id=_stable_id(payload_for_id),
            stage=str(stage),
            problem_name=str(problem_name),
            outer_candidate=tuple(float(v) for v in tuple(getattr(record, "outer_candidate", tuple()) or ())),
            selected_indices=tuple(int(v) for v in tuple(getattr(record, "selected_indices", tuple()) or ())),
            selected_terms=selected_terms,
            fitted_state=tuple(float(v) for v in tuple(getattr(record, "fitted_state", tuple()) or ())),
            objectives=tuple(float(v) for v in tuple(getattr(record, "objectives", tuple()) or ())),
            constraints=tuple(float(v) for v in tuple(getattr(record, "constraints", tuple()) or ())),
            metrics=dict(getattr(record, "metrics", {}) or {}),
            signals=dict(getattr(record, "signals", {}) or {}),
            audit=audit,
            replay_inputs=replay_inputs,
            metadata=dict(metadata or {}),
        )

    def describe(self) -> dict[str, Any]:
        return {"name": self.name}


def _term_summary(term: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "name": str(term.get("name", "")),
        "family": str(term.get("family", term.get("activation_family", "")) or ""),
        "activation_family": str(term.get("activation_family", "")),
        "features": [int(v) for v in tuple(term.get("features", ()) or ())],
        "complexity": float(term.get("complexity", 0.0) or 0.0),
        "prior_corr": float(term.get("prior_corr", 0.0) or 0.0),
    }


def _audit_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(report or {})
    inner = dict(payload.get("inner_report", {}) or {})
    score = dict(payload.get("candidate_score", {}) or {})
    graph = dict(payload.get("graph_cache", {}) or {})
    path = payload.get("path_memory")
    return {
        "inner_report": {
            "run_name": inner.get("run_name"),
            "status": inner.get("status"),
            "steps": inner.get("steps"),
            "best_score": inner.get("best_score"),
            "best_metrics": dict(inner.get("best_metrics", {}) or {}),
        },
        "candidate_score": {
            "score": score.get("score"),
            "success": score.get("success"),
            "score_parts": dict(score.get("score_parts", {}) or {}),
            "metadata": dict(score.get("metadata", {}) or {}),
        },
        "graph_cache": {
            key: graph.get(key)
            for key in (
                "backend",
                "namespace",
                "value_hits",
                "value_misses",
                "derivative_hits",
                "derivative_misses",
                "value_entries",
                "derivative_entries",
                "db_path",
            )
            if key in graph
        },
        "path_memory": None if path is None else dict(path),
    }


def _stable_id(payload: Mapping[str, Any]) -> str:
    blob = json.dumps(_jsonable_mapping(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return f"sym:replay:{hashlib.sha1(blob.encode('utf-8')).hexdigest()[:16]}"


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


__all__ = ["SymbolicCandidateReplayRecord", "SymbolicReplayRecordBuilder"]
