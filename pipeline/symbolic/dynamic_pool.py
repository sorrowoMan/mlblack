from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from mlblack.models.symbolic import binary_expr, const_expr, feature_expr, unary_expr

from .function_space import CandidateTerm, FunctionPool, FunctionSpace, FunctionSpaceProvider, dedupe_terms, safe_corr
from .grammar import DynamicActivationConfig, resolve_dynamic_activation_kwargs
from .pool_pipeline import FunctionPoolPipeline, FunctionPoolPipelineConfig
from .primitives import PrimitiveRegistry, default_primitive_registry


@dataclass(frozen=True)
class DynamicPoolConfig:
    residual_corr_threshold: float = 0.08
    gradient_score_threshold: float = 0.08
    budget_low_ratio: float = 0.15
    expand_unary_families: tuple[str, ...] = (
        "poly",
        "trig",
        "bounded",
        "saturation",
        "radial",
        "safe_log",
        "safe_exp",
        "safe_ratio",
    )
    expand_pair_families: tuple[str, ...] = (
        "interaction_basic",
        "interaction_poly",
        "interaction_compose",
        "interaction_ratio",
        "interaction_saturation",
        "interaction_radial",
        "interaction_rational",
        "linear_combo",
    )
    gate_quantiles: tuple[float, ...] = (0.25, 0.5, 0.75)
    gate_top_k: int = 4
    dynamic_top_k_features: int = 6
    dynamic_pair_top_k: int = 24
    max_added_terms: int = 64
    max_active_terms: int | None = None
    min_prior_corr: float = 0.0
    redundancy_corr_threshold: float = 0.995
    family_budget: Mapping[str, int] = field(default_factory=dict)
    activation: DynamicActivationConfig = field(default_factory=DynamicActivationConfig)


@dataclass(frozen=True)
class PoolSignal:
    residual_trigger: bool
    gradient_trigger: bool
    gate_trigger: bool
    budget_low: bool
    feature_scores: tuple[float, ...]
    focus_features: tuple[int, ...]
    reasons: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "residual_trigger": bool(self.residual_trigger),
            "gradient_trigger": bool(self.gradient_trigger),
            "gate_trigger": bool(self.gate_trigger),
            "budget_low": bool(self.budget_low),
            "feature_scores": list(self.feature_scores),
            "focus_features": list(self.focus_features),
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class DynamicPoolUpdate:
    pool: FunctionPool
    added_terms: tuple[CandidateTerm, ...]
    pruned_terms: tuple[CandidateTerm, ...]
    signal: PoolSignal
    before_count: int
    after_count: int
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "before_count": int(self.before_count),
            "after_count": int(self.after_count),
            "added_count": int(len(self.added_terms)),
            "pruned_count": int(len(self.pruned_terms)),
            "signal": self.signal.as_dict(),
            "added_terms": [term.describe(include_values=False) for term in self.added_terms],
            "pruned_terms": [term.describe(include_values=False) for term in self.pruned_terms],
            "metadata": dict(self.metadata),
        }


class DynamicFunctionPoolPipeline(FunctionSpaceProvider):
    """Residual/gradient/budget/gate aware symbolic pool updater."""

    name = "symbolic_dynamic_function_pool_pipeline"
    context_requires = ("data.X_train", "data.feature_names", "symbolic.function_pool")
    context_optional = (
        "data.y_train",
        "feedback.residuals",
        "feedback.gradients",
        "symbolic.gradient_signal",
        "signal.budget.remaining_ratio",
        "signal.gate.enabled",
        "resource.context",
    )
    context_provides = ("symbolic.function_pool", "symbolic.pool_delta", "symbolic.function_space")
    context_mutates = ("symbolic.function_pool",)
    context_cache = ("symbolic.function_pool",)
    context_notes = "Expands/prunes symbolic FunctionPool from residual, gradient, budget and gate signals."

    def __init__(
        self,
        config: DynamicPoolConfig | None = None,
        registry: PrimitiveRegistry | None = None,
    ) -> None:
        self.config = config or DynamicPoolConfig()
        self.registry = registry or default_primitive_registry()

    def update(
        self,
        X: np.ndarray,
        *,
        base_pool: FunctionPool | None = None,
        y: np.ndarray | None = None,
        residuals: np.ndarray | None = None,
        gradient_scores: Sequence[float] | None = None,
        budget_remaining_ratio: float | None = None,
        gate_scores: Sequence[float] | None = None,
        feature_names: Sequence[str] | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> DynamicPoolUpdate:
        ctx = dict(context or {})
        x = np.asarray(X, dtype=float)
        if x.ndim != 2:
            raise ValueError("X must be 2D")
        names = tuple(feature_names or tuple(f"x{i}" for i in range(x.shape[1])))
        if base_pool is None:
            base_space = FunctionPoolPipeline(registry=self.registry).build(x, y=y, feature_names=names, context=ctx)
            base_pool = base_space.pool

        signal = self._resolve_signal(
            x,
            y=y,
            residuals=residuals,
            gradient_scores=gradient_scores,
            budget_remaining_ratio=budget_remaining_ratio,
            gate_scores=gate_scores,
        )
        before_terms = tuple(base_pool.terms)
        dynamic_terms: list[CandidateTerm] = []
        if signal.residual_trigger or signal.gradient_trigger:
            activation = resolve_dynamic_activation_kwargs(self.config.activation)
            dynamic_space = FunctionPoolPipeline(
                FunctionPoolPipelineConfig(
                    mode="dynamic",
                    unary_families=self.config.expand_unary_families,
                    pair_families=self.config.expand_pair_families,
                    include_unary=True,
                    include_pairs=True,
                    top_k_features=max(1, int(self.config.dynamic_top_k_features)),
                    pair_top_k=max(0, int(self.config.dynamic_pair_top_k)),
                    max_terms=max(1, int(self.config.max_added_terms)),
                    family_budget=self.config.family_budget or dict(activation.get("family_budget", {})),
                    recursive_depth=int(activation.get("recursive_depth", 2)),
                    recursive_seed_top_k=int(activation.get("recursive_seed_top_k", 3)),
                    recursive_pair_seed_top_k=int(activation.get("recursive_pair_seed_top_k", 2)),
                    recursive_max_complexity=float(activation.get("recursive_max_complexity", 9.5)),
                ),
                registry=self.registry,
            ).build(
                x,
                y=y,
                residuals=residuals,
                gradient_scores=signal.feature_scores,
                feature_names=names,
                context=ctx,
            )
            dynamic_terms.extend(dynamic_space.pool.terms)

        if signal.gate_trigger:
            target = _target_from(y=y, residuals=residuals, X=x)
            dynamic_terms.extend(_gate_terms(x, names, target, signal.focus_features, self.config))

        candidate_terms = list(dedupe_terms((*before_terms, *dynamic_terms)))
        filtered = _filter_by_min_corr(candidate_terms, min_prior_corr=float(self.config.min_prior_corr))
        pruned = _prune_redundant(filtered, threshold=float(self.config.redundancy_corr_threshold))
        if signal.budget_low and pruned:
            limit = self.config.max_active_terms or max(4, int(round(len(before_terms) * 0.6)))
            pruned = _top_terms(pruned, limit)
        elif self.config.max_active_terms is not None:
            pruned = _top_terms(pruned, int(self.config.max_active_terms))

        before_keys = {term.key() for term in before_terms}
        after_keys = {term.key() for term in pruned}
        added = tuple(term for term in pruned if term.key() not in before_keys)
        removed = tuple(term for term in before_terms if term.key() not in after_keys)
        pool = FunctionPool(
            tuple(pruned),
            metadata={
                **dict(base_pool.metadata),
                "dynamic_update": True,
                "dynamic_reasons": list(signal.reasons),
                "context_signal_pool": ctx.get("signal.pool"),
            },
        )
        return DynamicPoolUpdate(
            pool=pool,
            added_terms=added,
            pruned_terms=removed,
            signal=signal,
            before_count=len(before_terms),
            after_count=len(pruned),
            metadata={"pipeline": type(self).__name__},
        )

    def build(self, X: np.ndarray, **kwargs: Any) -> FunctionSpace:
        update = self.update(X, **kwargs)
        names = tuple(kwargs.get("feature_names") or tuple(f"x{i}" for i in range(np.asarray(X).shape[1])))
        return FunctionSpace(
            pool=update.pool,
            feature_names=names,
            registry_summary=self.registry.describe(),
            metadata={"dynamic_update": update.as_dict()},
        )

    def _resolve_signal(
        self,
        X: np.ndarray,
        *,
        y: np.ndarray | None,
        residuals: np.ndarray | None,
        gradient_scores: Sequence[float] | None,
        budget_remaining_ratio: float | None,
        gate_scores: Sequence[float] | None,
    ) -> PoolSignal:
        target = _target_from(y=y, residuals=residuals, X=X)
        residual_scores = np.asarray([abs(safe_corr(X[:, i], target)) for i in range(X.shape[1])], dtype=float)
        gradient_arr = None if gradient_scores is None else np.asarray(tuple(gradient_scores), dtype=float).reshape(-1)
        if gradient_arr is not None and gradient_arr.shape[0] != X.shape[1]:
            gradient_arr = None
        combined = residual_scores.copy()
        if gradient_arr is not None:
            combined = np.maximum(combined, np.abs(gradient_arr))
        residual_trigger = bool(residuals is not None and np.max(residual_scores, initial=0.0) >= float(self.config.residual_corr_threshold))
        gradient_trigger = bool(gradient_arr is not None and np.max(np.abs(gradient_arr), initial=0.0) >= float(self.config.gradient_score_threshold))
        budget_low = bool(budget_remaining_ratio is not None and float(budget_remaining_ratio) <= float(self.config.budget_low_ratio))
        gate_arr = None if gate_scores is None else np.asarray(tuple(gate_scores), dtype=float).reshape(-1)
        gate_trigger = bool(gate_arr is not None and gate_arr.size > 0 and np.max(np.abs(gate_arr), initial=0.0) > 0.0)
        focus_count = max(1, int(self.config.dynamic_top_k_features))
        focus = tuple(int(i) for i in np.argsort(-combined)[:focus_count])
        reasons: list[str] = []
        if residual_trigger:
            reasons.append("residual")
        if gradient_trigger:
            reasons.append("gradient")
        if gate_trigger:
            reasons.append("gate")
        if budget_low:
            reasons.append("budget_low")
        if not reasons:
            reasons.append("no_expand")
        return PoolSignal(
            residual_trigger=residual_trigger,
            gradient_trigger=gradient_trigger,
            gate_trigger=gate_trigger,
            budget_low=budget_low,
            feature_scores=tuple(float(v) for v in combined),
            focus_features=focus,
            reasons=tuple(reasons),
        )


def _target_from(*, y: np.ndarray | None, residuals: np.ndarray | None, X: np.ndarray) -> np.ndarray:
    if residuals is not None:
        return np.asarray(residuals, dtype=float).reshape(-1)
    if y is not None:
        return np.asarray(y, dtype=float).reshape(-1)
    return np.std(np.asarray(X, dtype=float), axis=1).reshape(-1)


def _gate_terms(
    X: np.ndarray,
    feature_names: Sequence[str],
    target: np.ndarray,
    focus_features: Sequence[int],
    config: DynamicPoolConfig,
) -> tuple[CandidateTerm, ...]:
    x = np.asarray(X, dtype=float)
    out: list[CandidateTerm] = []
    for feature_idx in tuple(focus_features)[: max(1, int(config.gate_top_k))]:
        col = np.asarray(x[:, int(feature_idx)], dtype=float).reshape(-1)
        finite = col[np.isfinite(col)]
        if finite.size < 4:
            continue
        for q in config.gate_quantiles:
            quantile = float(q)
            if not 0.0 < quantile < 1.0:
                continue
            cut = float(np.quantile(finite, quantile))
            expr = _soft_gate_expr(int(feature_idx), cut=cut, slope=8.0)
            values = _soft_gate_values(col, cut=cut, slope=8.0)
            out.append(
                CandidateTerm(
                    name=f"gate_soft({feature_names[int(feature_idx)]}>{cut:.4g})",
                    expr=expr,
                    values=values,
                    complexity=3.5,
                    family="gate_soft",
                    activation_family="gate",
                    features=(int(feature_idx),),
                    prior_corr=abs(safe_corr(values, target)),
                    metadata={"cut": cut, "quantile": quantile},
                )
            )
    return tuple(out)


def _soft_gate_expr(feature_idx: int, *, cut: float, slope: float) -> dict[str, Any]:
    shifted = binary_expr("sub", feature_expr(feature_idx), const_expr(float(cut)))
    scaled = binary_expr("mul", const_expr(float(slope)), shifted)
    return binary_expr("mul", const_expr(0.5), binary_expr("add", const_expr(1.0), unary_expr("tanh", scaled)))


def _soft_gate_values(values: np.ndarray, *, cut: float, slope: float) -> np.ndarray:
    z = np.asarray(values, dtype=float).reshape(-1) - float(cut)
    return np.asarray(0.5 * (1.0 + np.tanh(float(slope) * z)), dtype=float).reshape(-1)


def _filter_by_min_corr(terms: Sequence[CandidateTerm], *, min_prior_corr: float) -> list[CandidateTerm]:
    if min_prior_corr <= 0.0:
        return list(terms)
    return [term for term in terms if abs(float(term.prior_corr)) >= float(min_prior_corr) or term.activation_family == "seed"]


def _prune_redundant(terms: Sequence[CandidateTerm], *, threshold: float) -> list[CandidateTerm]:
    ordered = sorted(terms, key=lambda term: (-abs(float(term.prior_corr)), float(term.complexity), term.name))
    kept: list[CandidateTerm] = []
    for term in ordered:
        values = np.asarray(term.values, dtype=float).reshape(-1)
        redundant = False
        for prev in kept:
            if abs(safe_corr(values, np.asarray(prev.values, dtype=float).reshape(-1))) >= float(threshold):
                redundant = True
                break
        if not redundant:
            kept.append(term)
    return kept


def _top_terms(terms: Sequence[CandidateTerm], limit: int) -> list[CandidateTerm]:
    ordered = sorted(terms, key=lambda term: (-abs(float(term.prior_corr)), float(term.complexity), term.name))
    return list(ordered[: max(0, int(limit))])
