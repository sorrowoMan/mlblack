from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from mlblack.models.symbolic import expression_complexity
from mlblack.models.symbolic_normalization import expression_equivalence_key
from mlblack.pipeline.symbolic import CandidateTerm, safe_corr

from .overfit_guard import OverfitGuardConfig, OverfitGuardReport, SymbolicOverfitGuard
from .path_memory import SymbolicPathMemory
from .structure_guard import StructureGuardConfig, SymbolicStructureGuard


@dataclass(frozen=True)
class CandidateScoreConfig:
    objective_weight: float = 1.0
    constraint_penalty: float = 1000.0
    prior_corr_bonus: float = 0.04
    complexity_penalty: float = 7e-4
    gradient_alignment_bonus: float = 0.02
    novelty_bonus: float = 0.01
    path_accept_bonus: float = 0.05
    duplicate_penalty: float = 0.5
    overfit_penalty_weight: float = 1.0
    structure_guard_penalty_weight: float = 1.0
    success_quantile_score: float = 1.0
    structure_guard_config: StructureGuardConfig = field(default_factory=StructureGuardConfig)
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CandidateScoreReport:
    score: float
    scalar_objective: float
    score_parts: Mapping[str, float]
    selected_expr_keys: tuple[str, ...]
    selected_term_names: tuple[str, ...]
    prior_summary: Mapping[str, Any]
    overfit_guard: Mapping[str, Any]
    structure_guard: Mapping[str, Any]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        overfit = dict(self.overfit_guard)
        return bool(float(self.score) <= float(self.metadata.get("success_quantile_score", 1.0)) and not bool(overfit.get("triggered", False)))

    def as_dict(self) -> dict[str, Any]:
        return {
            "score": float(self.score),
            "scalar_objective": float(self.scalar_objective),
            "score_parts": {str(k): float(v) for k, v in self.score_parts.items()},
            "selected_expr_keys": list(self.selected_expr_keys),
            "selected_term_names": list(self.selected_term_names),
            "prior_summary": dict(self.prior_summary),
            "overfit_guard": dict(self.overfit_guard),
            "structure_guard": dict(self.structure_guard),
            "success": bool(self.success),
            "metadata": dict(self.metadata),
        }


class SymbolicCandidateScorer:
    """Candidate scoring policy for nsgablack symbolic outer search."""

    name = "symbolic_candidate_scorer"
    context_requires = ("feedback.objectives", "symbolic.function_pool")
    context_optional = (
        "feedback.constraints",
        "feedback.metrics",
        "feedback.gradients",
        "symbolic.path_memory",
        "symbolic.overfit_guard",
        "symbolic.structure_guard",
    )
    context_provides = ("symbolic.candidate_score",)
    context_mutates = ("symbolic.path_memory",)
    context_cache = ("symbolic.path_memory",)
    requires_metrics = ()
    metrics_fallback = "strict"
    context_notes = "Combines objective, constraints, path memory, overfit guard and candidate metadata into one audit score."

    def __init__(
        self,
        config: CandidateScoreConfig | None = None,
        *,
        path_memory: SymbolicPathMemory | None = None,
        overfit_guard: SymbolicOverfitGuard | None = None,
        structure_guard: SymbolicStructureGuard | None = None,
    ) -> None:
        self.config = config or CandidateScoreConfig()
        self.path_memory = path_memory
        self.overfit_guard = overfit_guard or SymbolicOverfitGuard(
            OverfitGuardConfig(penalty_weight=float(self.config.overfit_penalty_weight))
        )
        self.structure_guard = structure_guard or SymbolicStructureGuard(self.config.structure_guard_config)

    def score(
        self,
        *,
        objectives: Sequence[float] | np.ndarray,
        constraints: Sequence[float] | np.ndarray = (),
        selected_terms: Sequence[CandidateTerm | Mapping[str, Any]] = (),
        metrics: Mapping[str, Any] | None = None,
        gradient_scores: Sequence[float] | None = None,
        expression: Mapping[str, Any] | None = None,
        path_memory: SymbolicPathMemory | None = None,
        record_memory: bool = True,
    ) -> CandidateScoreReport:
        cfg = self.config
        obj = np.asarray(objectives, dtype=float).reshape(-1)
        cons = np.asarray(constraints, dtype=float).reshape(-1)
        violation = float(np.sum(np.maximum(cons, 0.0))) if cons.size else 0.0
        scalar_objective = float(np.sum(obj)) + float(cfg.constraint_penalty) * violation

        terms = tuple(selected_terms or ())
        expr_keys = tuple(_term_expr_key(term) for term in terms)
        names = tuple(_term_name(term, idx) for idx, term in enumerate(terms))
        complexities = [_term_complexity(term) for term in terms]
        prior_corrs = [_term_prior_corr(term) for term in terms]
        complexity = float(sum(complexities))
        mean_prior_corr = float(np.mean(np.abs(prior_corrs))) if prior_corrs else 0.0
        duplicate_count = float(len(expr_keys) - len(set(expr_keys)))
        novelty = 1.0
        memory = path_memory or self.path_memory
        prior_summary: Mapping[str, Any] = {"count": 0, "mean_accept_rate": 0.5, "mean_seen": 0.0, "items": []}
        if memory is not None and expr_keys:
            prior_summary = memory.prior_summary(expr_keys)
            mean_seen = float(prior_summary.get("mean_seen", 0.0) or 0.0)
            novelty = float(1.0 / (1.0 + mean_seen))

        grad_alignment = _gradient_alignment(terms, gradient_scores)
        guard_report = self.overfit_guard.evaluate(metrics or {}).as_dict()
        overfit_penalty = float(guard_report.get("penalty", 0.0) or 0.0)
        structure_report = self.structure_guard.evaluate(terms).as_dict()
        structure_penalty = float(structure_report.get("penalty", 0.0) or 0.0)
        structure_bonus = float(structure_report.get("bonus", 0.0) or 0.0)
        path_accept_bonus = float(cfg.path_accept_bonus) * (float(prior_summary.get("mean_accept_rate", 0.5)) - 0.5)

        parts = {
            "objective": float(cfg.objective_weight) * scalar_objective,
            "constraint_penalty": float(cfg.constraint_penalty) * violation,
            "prior_corr_bonus": -float(cfg.prior_corr_bonus) * mean_prior_corr,
            "complexity_penalty": float(cfg.complexity_penalty) * complexity,
            "gradient_alignment_bonus": -float(cfg.gradient_alignment_bonus) * grad_alignment,
            "novelty_bonus": -float(cfg.novelty_bonus) * novelty,
            "path_accept_bonus": -float(path_accept_bonus),
            "duplicate_penalty": float(cfg.duplicate_penalty) * duplicate_count,
            "overfit_penalty": float(cfg.overfit_penalty_weight) * overfit_penalty,
            "structure_guard_penalty": float(cfg.structure_guard_penalty_weight) * structure_penalty,
            "structure_native_bonus": -structure_bonus,
        }
        score = float(sum(parts.values()))
        if expression is not None and not expr_keys:
            expr_keys = (expression_equivalence_key(expression),)
            names = ("expression",)

        report = CandidateScoreReport(
            score=score,
            scalar_objective=scalar_objective,
            score_parts=parts,
            selected_expr_keys=expr_keys,
            selected_term_names=names,
            prior_summary=prior_summary,
            overfit_guard=guard_report,
            structure_guard=structure_report,
            metadata={
                "success_quantile_score": float(cfg.success_quantile_score),
                "complexity": complexity,
                "mean_prior_corr": mean_prior_corr,
                "gradient_alignment": grad_alignment,
                "duplicate_count": duplicate_count,
                "structure_guard_triggered": bool(structure_report.get("triggered", False)),
                "config": {
                    "objective_weight": float(cfg.objective_weight),
                    "constraint_penalty": float(cfg.constraint_penalty),
                    "prior_corr_bonus": float(cfg.prior_corr_bonus),
                    "complexity_penalty": float(cfg.complexity_penalty),
                    "gradient_alignment_bonus": float(cfg.gradient_alignment_bonus),
                    "novelty_bonus": float(cfg.novelty_bonus),
                    "path_accept_bonus": float(cfg.path_accept_bonus),
                    "duplicate_penalty": float(cfg.duplicate_penalty),
                    "overfit_penalty_weight": float(cfg.overfit_penalty_weight),
                    "structure_guard_penalty_weight": float(cfg.structure_guard_penalty_weight),
                    "metadata": dict(cfg.metadata),
                },
            },
        )
        if record_memory and memory is not None:
            for key in expr_keys:
                memory.record_expr_outcome(
                    key,
                    selected_score=score,
                    delta_score=-score,
                    success=report.success,
                )
        return report

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "config": {
                "objective_weight": float(self.config.objective_weight),
                "constraint_penalty": float(self.config.constraint_penalty),
                "prior_corr_bonus": float(self.config.prior_corr_bonus),
                "complexity_penalty": float(self.config.complexity_penalty),
                "gradient_alignment_bonus": float(self.config.gradient_alignment_bonus),
                "novelty_bonus": float(self.config.novelty_bonus),
                "path_accept_bonus": float(self.config.path_accept_bonus),
                "duplicate_penalty": float(self.config.duplicate_penalty),
                "overfit_penalty_weight": float(self.config.overfit_penalty_weight),
                "structure_guard_penalty_weight": float(self.config.structure_guard_penalty_weight),
                "success_quantile_score": float(self.config.success_quantile_score),
                "metadata": dict(self.config.metadata),
            },
            "path_memory": None if self.path_memory is None else self.path_memory.describe(),
            "overfit_guard": self.overfit_guard.describe(),
            "structure_guard": self.structure_guard.describe(),
        }


def _term_expr_key(term: CandidateTerm | Mapping[str, Any]) -> str:
    if isinstance(term, CandidateTerm):
        return term.key()
    if "expr" in term:
        expr = dict(term.get("expr", {}) or {})
        if expr:
            return expression_equivalence_key(expr)
    return str(term)


def _term_name(term: CandidateTerm | Mapping[str, Any], index: int) -> str:
    if isinstance(term, CandidateTerm):
        return str(term.name)
    return str(term.get("name", f"term_{index}"))


def _term_complexity(term: CandidateTerm | Mapping[str, Any]) -> float:
    if isinstance(term, CandidateTerm):
        return float(term.complexity)
    if term.get("complexity") is not None:
        return float(term.get("complexity") or 0.0)
    if "expr" in term:
        return float(expression_complexity(dict(term["expr"])))
    return 1.0


def _term_prior_corr(term: CandidateTerm | Mapping[str, Any]) -> float:
    if isinstance(term, CandidateTerm):
        return float(term.prior_corr)
    return float(term.get("prior_corr", 0.0) or 0.0)


def _gradient_alignment(terms: Sequence[CandidateTerm | Mapping[str, Any]], gradient_scores: Sequence[float] | None) -> float:
    if gradient_scores is None:
        return 0.0
    scores = np.asarray(tuple(gradient_scores), dtype=float).reshape(-1)
    if scores.size == 0:
        return 0.0
    values: list[float] = []
    for term in terms:
        features = term.features if isinstance(term, CandidateTerm) else tuple(term.get("features", ()) or ())
        feature_scores = [abs(float(scores[int(idx)])) for idx in features if 0 <= int(idx) < scores.shape[0]]
        if feature_scores:
            values.append(float(np.mean(feature_scores)))
    return float(np.mean(values)) if values else 0.0


__all__ = [
    "CandidateScoreConfig",
    "CandidateScoreReport",
    "SymbolicCandidateScorer",
]
