from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from mlblack.models.symbolic import expression_complexity
from mlblack.models.symbolic_normalization import expression_equivalence_key
from mlblack.pipeline.symbolic import CandidateTerm, safe_corr


@dataclass(frozen=True)
class StructureGuardConfig:
    """Policy knobs for symbolic structure safety and reuse checks."""

    enabled: bool = True
    max_complexity: float = 24.0
    max_feature_reuse: int = 2
    max_duplicate_terms: int = 0
    redundancy_corr_threshold: float = 0.995
    min_value_stability_score: float = 0.12
    min_pole_safety_score: float = 0.12
    min_native_structure_score: float = 0.0
    complexity_penalty_weight: float = 0.01
    feature_reuse_penalty_weight: float = 0.25
    duplicate_penalty_weight: float = 0.5
    redundancy_penalty_weight: float = 0.25
    value_stability_penalty_weight: float = 0.5
    pole_safety_penalty_weight: float = 0.5
    native_structure_bonus_weight: float = 0.02
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StructureGuardReport:
    triggered: bool
    penalty: float
    bonus: float
    reasons: tuple[str, ...]
    score_parts: Mapping[str, float]
    metrics: Mapping[str, Any]
    constraints: tuple[float, ...]
    config: Mapping[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "triggered": bool(self.triggered),
            "penalty": float(self.penalty),
            "bonus": float(self.bonus),
            "reasons": list(self.reasons),
            "score_parts": {str(k): float(v) for k, v in self.score_parts.items()},
            "metrics": dict(self.metrics),
            "constraints": list(self.constraints),
            "config": dict(self.config),
        }


class SymbolicStructureGuard:
    """Outer-search structural guard for symbolic candidates.

    This is the new home for fine-grained old symbolic-search ideas such as
    seat guard, reuse isolation, native-structure score, pole safety, chart
    stability, and redundancy checks.
    """

    name = "symbolic_structure_guard"
    context_requires = ("symbolic.function_pool",)
    context_optional = ("symbolic.candidate_score", "data.X_train", "feedback.metrics", "resource.context")
    context_provides = ("symbolic.structure_guard", "symbolic.native_structure_score")
    context_mutates = ()
    context_cache = ()
    requires_metrics = ()
    metrics_fallback = "strict"
    context_notes = "Scores symbolic candidate structure safety, reuse, redundancy, pole safety and native-structure fit."

    def __init__(self, config: StructureGuardConfig | None = None) -> None:
        self.config = config or StructureGuardConfig()

    def evaluate(self, selected_terms: Sequence[CandidateTerm | Mapping[str, Any]]) -> StructureGuardReport:
        cfg = self.config
        if not bool(cfg.enabled):
            return self._report(False, 0.0, 0.0, (), {}, {}, ())
        terms = tuple(selected_terms or ())
        expr_keys = tuple(_term_expr_key(term) for term in terms)
        complexities = tuple(_term_complexity(term) for term in terms)
        features = tuple(_term_features(term) for term in terms)
        values = tuple(_term_values(term) for term in terms)
        families = tuple(_term_family(term) for term in terms)

        total_complexity = float(sum(complexities))
        duplicate_count = int(len(expr_keys) - len(set(expr_keys)))
        feature_reuse = _feature_reuse_counts(features)
        max_reuse = max(feature_reuse.values(), default=0)
        reuse_excess = float(sum(max(0, count - int(cfg.max_feature_reuse)) for count in feature_reuse.values()))
        redundancy_count = _redundancy_count(values, threshold=float(cfg.redundancy_corr_threshold))
        value_stability_scores = tuple(_value_stability_score(value) for value in values if value is not None)
        pole_safety_scores = tuple(_pole_safety_score(value) for value in values if value is not None)
        native_scores = tuple(_native_structure_score(family, complexity) for family, complexity in zip(families, complexities))

        min_value_stability = float(min(value_stability_scores)) if value_stability_scores else 1.0
        min_pole_safety = float(min(pole_safety_scores)) if pole_safety_scores else 1.0
        mean_native_score = float(np.mean(native_scores)) if native_scores else 0.0

        reasons: list[str] = []
        parts: dict[str, float] = {}
        complexity_excess = max(0.0, total_complexity - float(cfg.max_complexity))
        if complexity_excess > 0.0:
            reasons.append("complexity_budget")
        parts["complexity_penalty"] = float(cfg.complexity_penalty_weight) * complexity_excess

        duplicate_excess = max(0, duplicate_count - int(cfg.max_duplicate_terms))
        if duplicate_excess > 0:
            reasons.append("seat_duplicate")
        parts["duplicate_penalty"] = float(cfg.duplicate_penalty_weight) * float(duplicate_excess)

        if reuse_excess > 0.0:
            reasons.append("feature_reuse")
        parts["feature_reuse_penalty"] = float(cfg.feature_reuse_penalty_weight) * reuse_excess

        if redundancy_count > 0:
            reasons.append("redundant_values")
        parts["redundancy_penalty"] = float(cfg.redundancy_penalty_weight) * float(redundancy_count)

        value_deficit = max(0.0, float(cfg.min_value_stability_score) - min_value_stability)
        if value_deficit > 0.0:
            reasons.append("chart_value_instability")
        parts["value_stability_penalty"] = float(cfg.value_stability_penalty_weight) * value_deficit

        pole_deficit = max(0.0, float(cfg.min_pole_safety_score) - min_pole_safety)
        if pole_deficit > 0.0:
            reasons.append("pole_safety")
        parts["pole_safety_penalty"] = float(cfg.pole_safety_penalty_weight) * pole_deficit

        native_deficit = max(0.0, float(cfg.min_native_structure_score) - mean_native_score)
        if native_deficit > 0.0:
            reasons.append("native_structure_score")
        parts["native_structure_penalty"] = native_deficit
        bonus = float(cfg.native_structure_bonus_weight) * mean_native_score

        penalty = float(sum(parts.values()))
        metrics = {
            "structure.total_complexity": total_complexity,
            "structure.duplicate_count": duplicate_count,
            "structure.max_feature_reuse": max_reuse,
            "structure.feature_reuse_excess": reuse_excess,
            "structure.redundancy_count": redundancy_count,
            "structure.min_value_stability_score": min_value_stability,
            "structure.min_pole_safety_score": min_pole_safety,
            "structure.mean_native_structure_score": mean_native_score,
            "structure.families": list(families),
            "structure.feature_reuse": {str(k): int(v) for k, v in feature_reuse.items()},
        }
        constraints = (
            complexity_excess,
            float(duplicate_excess),
            reuse_excess,
            float(redundancy_count),
            value_deficit,
            pole_deficit,
            native_deficit,
        )
        return self._report(bool(reasons), penalty, bonus, tuple(reasons), parts, metrics, constraints)

    def _report(
        self,
        triggered: bool,
        penalty: float,
        bonus: float,
        reasons: tuple[str, ...],
        parts: Mapping[str, float],
        metrics: Mapping[str, Any],
        constraints: tuple[float, ...],
    ) -> StructureGuardReport:
        cfg = self.config
        return StructureGuardReport(
            triggered=bool(triggered),
            penalty=float(penalty),
            bonus=float(bonus),
            reasons=tuple(reasons),
            score_parts=dict(parts),
            metrics=dict(metrics),
            constraints=tuple(float(v) for v in constraints),
            config={
                "enabled": bool(cfg.enabled),
                "max_complexity": float(cfg.max_complexity),
                "max_feature_reuse": int(cfg.max_feature_reuse),
                "max_duplicate_terms": int(cfg.max_duplicate_terms),
                "redundancy_corr_threshold": float(cfg.redundancy_corr_threshold),
                "min_value_stability_score": float(cfg.min_value_stability_score),
                "min_pole_safety_score": float(cfg.min_pole_safety_score),
                "min_native_structure_score": float(cfg.min_native_structure_score),
                "metadata": dict(cfg.metadata),
            },
        )

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "config": self._report(False, 0.0, 0.0, (), {}, {}, ()).config}


def _term_expr_key(term: CandidateTerm | Mapping[str, Any]) -> str:
    if isinstance(term, CandidateTerm):
        return term.key()
    expr = dict(term.get("expr", {}) or {}) if isinstance(term, Mapping) else {}
    if expr:
        return expression_equivalence_key(expr)
    return str(term)


def _term_complexity(term: CandidateTerm | Mapping[str, Any]) -> float:
    if isinstance(term, CandidateTerm):
        return float(term.complexity)
    if term.get("complexity") is not None:
        return float(term.get("complexity") or 0.0)
    expr = dict(term.get("expr", {}) or {})
    return float(expression_complexity(expr)) if expr else 1.0


def _term_features(term: CandidateTerm | Mapping[str, Any]) -> tuple[int, ...]:
    if isinstance(term, CandidateTerm):
        return tuple(int(v) for v in term.features)
    return tuple(int(v) for v in tuple(term.get("features", ()) or ()))


def _term_values(term: CandidateTerm | Mapping[str, Any]) -> np.ndarray | None:
    raw = term.values if isinstance(term, CandidateTerm) else term.get("values")
    if raw is None:
        return None
    try:
        arr = np.asarray(raw, dtype=float).reshape(-1)
    except Exception:
        return None
    return arr if arr.size else None


def _term_family(term: CandidateTerm | Mapping[str, Any]) -> str:
    if isinstance(term, CandidateTerm):
        return str(term.activation_family or term.family)
    return str(term.get("activation_family", term.get("family", "")) or "")


def _feature_reuse_counts(features: Sequence[Sequence[int]]) -> dict[int, int]:
    counts: dict[int, int] = {}
    for row in features:
        for value in set(int(v) for v in row):
            counts[value] = counts.get(value, 0) + 1
    return counts


def _redundancy_count(values: Sequence[np.ndarray | None], *, threshold: float) -> int:
    usable = [np.asarray(value, dtype=float).reshape(-1) for value in values if value is not None]
    count = 0
    for i, left in enumerate(usable):
        for right in usable[i + 1 :]:
            if left.shape[0] == right.shape[0] and abs(safe_corr(left, right)) >= float(threshold):
                count += 1
    return int(count)


def _value_stability_score(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        return 0.0
    finite = arr[np.isfinite(arr)]
    finite_ratio = float(finite.size) / float(arr.size)
    if finite.size == 0:
        return 0.0
    q10, q50, q90 = (float(v) for v in np.quantile(finite, [0.1, 0.5, 0.9]))
    spread = abs(q90 - q10)
    scale = max(1e-12, abs(q50), float(np.std(finite)), 1.0)
    spread_penalty = min(10.0, spread / scale)
    variation_penalty = min(10.0, float(np.std(np.diff(finite))) / scale) if finite.size >= 3 else 0.0
    score = finite_ratio / (1.0 + 0.35 * spread_penalty + 0.45 * variation_penalty)
    return float(np.clip(score, 0.0, 1.0))


def _pole_safety_score(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        return 0.0
    finite = arr[np.isfinite(arr)]
    finite_ratio = float(finite.size) / float(arr.size)
    if finite.size == 0:
        return 0.0
    abs_values = np.abs(finite)
    q50 = float(np.quantile(abs_values, 0.5))
    q95 = float(np.quantile(abs_values, 0.95))
    q99 = float(np.quantile(abs_values, 0.99))
    tail_ratio = q99 / max(q50, 1.0)
    tail_penalty = min(20.0, tail_ratio / 10.0)
    saturation_penalty = 1.0 if q95 > 1e6 else 0.0
    score = finite_ratio / (1.0 + tail_penalty + saturation_penalty)
    return float(np.clip(score, 0.0, 1.0))


def _native_structure_score(family: str, complexity: float) -> float:
    key = str(family or "").strip().lower()
    score = 0.35
    if key in {"seed", "basis_atom"}:
        score = 1.0
    elif key in {"poly", "trig", "bounded", "saturation", "radial"}:
        score = 0.75
    elif key in {"safe_log", "safe_exp", "safe_ratio", "interaction_ratio", "interaction_rational"}:
        score = 0.6
    elif "gate" in key or "conditional" in key:
        score = 0.55
    elif "interaction" in key or "linear_combo" in key:
        score = 0.7
    score -= 0.02 * max(0.0, float(complexity) - 4.0)
    return float(np.clip(score, 0.0, 1.25))


__all__ = ["StructureGuardConfig", "StructureGuardReport", "SymbolicStructureGuard"]
