from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from mlblack.pipeline.conditional import ConditionalPrimitiveSpec

from .function_space import CandidateTerm, FunctionPool, FunctionSpace, FunctionSpaceProvider, dedupe_terms, safe_corr
from .grammar import (
    GrammarCandidate,
    generate_pair_candidates,
    generate_recursive_pair_candidates,
    generate_recursive_unary_candidates,
    generate_unary_candidates,
    lower_conditional_primitive_specs,
    make_seed_candidate,
)
from .primitives import PrimitiveRegistry, default_primitive_registry, feature_expr


@dataclass(frozen=True)
class FunctionPoolPipelineConfig:
    """Controls materialization of the active symbolic function pool."""

    mode: str = "initial"
    unary_families: tuple[str, ...] | None = None
    pair_families: tuple[str, ...] | None = None
    include_unary: bool = True
    include_pairs: bool = True
    include_conditional: bool = True
    top_k_features: int | None = None
    pair_top_k: int = 16
    max_terms: int | None = None
    family_budget: Mapping[str, int] = field(default_factory=dict)
    primitive_params: Mapping[str, float] = field(default_factory=dict)
    conditional_specs: tuple[ConditionalPrimitiveSpec, ...] = tuple()
    recursive_depth: int = 1
    recursive_seed_top_k: int = 3
    recursive_pair_seed_top_k: int = 2
    recursive_max_complexity: float | None = None


class FunctionPoolPipeline(FunctionSpaceProvider):
    """Symbolic search-space pipeline.

    It reads data/signals and materializes the currently active FunctionPool.
    Outer optimizers consume the pool as representation/search context.
    """

    name = "symbolic_function_pool_pipeline"

    def __init__(
        self,
        config: FunctionPoolPipelineConfig | None = None,
        registry: PrimitiveRegistry | None = None,
    ) -> None:
        self.config = config or FunctionPoolPipelineConfig()
        self.registry = registry or default_primitive_registry()

    def build(
        self,
        X: np.ndarray,
        *,
        y: np.ndarray | None = None,
        feature_names: Sequence[str] | None = None,
        residuals: np.ndarray | None = None,
        gradient_scores: Sequence[float] | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> FunctionSpace:
        ctx = dict(context or {})
        x = np.asarray(X, dtype=float)
        if x.ndim != 2:
            raise ValueError("X must be 2D")
        names = tuple(feature_names or tuple(f"x{i}" for i in range(x.shape[1])))
        if len(names) != x.shape[1]:
            raise ValueError("feature_names length must match X columns")

        target = _resolve_target(y=y, residuals=residuals, X=x)
        feature_order = _rank_features(x, target=target, gradient_scores=gradient_scores)
        if self.config.top_k_features is not None:
            feature_order = feature_order[: max(1, int(self.config.top_k_features))]

        grammar_items: list[GrammarCandidate] = []
        seeds = _seed_grammar_candidates(x, names, target)
        grammar_items.extend(seeds)
        if self.config.include_unary:
            grammar_items.extend(_unary_candidates(self.registry, x, names, feature_order, config=self.config))
        if self.config.include_pairs:
            grammar_items.extend(_pair_candidates(self.registry, x, names, feature_order, config=self.config))
        if int(self.config.recursive_depth) >= 2:
            grammar_items.extend(_recursive_candidates(self.registry, x, names, target, feature_order, config=self.config))
        if self.config.include_conditional and self.config.conditional_specs:
            grammar_items.extend(
                lower_conditional_primitive_specs(
                    self.config.conditional_specs,
                    feature_names=names,
                    X=x,
                )
            )

        terms = [_term_from_grammar(item, target) for item in grammar_items]
        deduped = list(dedupe_terms(terms))
        budgeted = _apply_family_budget(deduped, self.config.family_budget)
        if self.config.max_terms is not None:
            budgeted = sorted(budgeted, key=lambda term: (-abs(float(term.prior_corr)), float(term.complexity), term.name))
            budgeted = budgeted[: max(0, int(self.config.max_terms))]

        pool = FunctionPool(
            tuple(budgeted),
            metadata={
                "mode": str(self.config.mode),
                "source": type(self).__name__,
                "context_signal_pool": ctx.get("signal.pool"),
                "recursive_depth": int(self.config.recursive_depth),
                "conditional_specs": [spec.describe() for spec in self.config.conditional_specs],
            },
        )
        return FunctionSpace(
            pool=pool,
            feature_names=names,
            registry_summary=self.registry.describe(),
            metadata={
                "pipeline": type(self).__name__,
                "n_input_features": int(x.shape[1]),
                "target_source": "residuals" if residuals is not None else ("y" if y is not None else "feature_std"),
            },
        )

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "config": {
                "mode": self.config.mode,
                "unary_families": None if self.config.unary_families is None else list(self.config.unary_families),
                "pair_families": None if self.config.pair_families is None else list(self.config.pair_families),
                "include_unary": bool(self.config.include_unary),
                "include_pairs": bool(self.config.include_pairs),
                "include_conditional": bool(self.config.include_conditional),
                "top_k_features": self.config.top_k_features,
                "pair_top_k": int(self.config.pair_top_k),
                "max_terms": self.config.max_terms,
                "family_budget": dict(self.config.family_budget),
                "primitive_params": dict(self.config.primitive_params),
                "conditional_specs": [spec.describe() for spec in self.config.conditional_specs],
                "recursive_depth": int(self.config.recursive_depth),
                "recursive_seed_top_k": int(self.config.recursive_seed_top_k),
                "recursive_pair_seed_top_k": int(self.config.recursive_pair_seed_top_k),
                "recursive_max_complexity": self.config.recursive_max_complexity,
            },
            "contract": self.get_contract().describe(),
        }


def _resolve_target(*, y: np.ndarray | None, residuals: np.ndarray | None, X: np.ndarray) -> np.ndarray:
    if residuals is not None:
        target = np.asarray(residuals, dtype=float).reshape(-1)
    elif y is not None:
        target = np.asarray(y, dtype=float).reshape(-1)
    else:
        target = np.std(np.asarray(X, dtype=float), axis=1).reshape(-1)
    if target.shape[0] != X.shape[0]:
        raise ValueError("target/residual length must match X rows")
    return target


def _rank_features(
    X: np.ndarray,
    *,
    target: np.ndarray,
    gradient_scores: Sequence[float] | None,
) -> list[int]:
    if gradient_scores is not None:
        scores = np.asarray(tuple(gradient_scores), dtype=float).reshape(-1)
        if scores.shape[0] == X.shape[1]:
            return [int(i) for i in np.argsort(-np.abs(scores))]
    corr_scores = np.asarray([abs(safe_corr(X[:, i], target)) for i in range(X.shape[1])], dtype=float)
    return [int(i) for i in np.argsort(-corr_scores)]


def _seed_grammar_candidates(X: np.ndarray, feature_names: Sequence[str], target: np.ndarray) -> list[GrammarCandidate]:
    _ = target
    return [
        make_seed_candidate(
            name=str(name),
            expr=feature_expr(i),
            values=np.asarray(X[:, i], dtype=float).reshape(-1),
            features=(int(i),),
            family="seed",
            activation_family="seed",
        )
        for i, name in enumerate(feature_names)
    ]


def _term_from_grammar(item: GrammarCandidate, target: np.ndarray) -> CandidateTerm:
    values = np.asarray(item.values, dtype=float).reshape(-1)
    return CandidateTerm(
        name=str(item.name),
        expr=dict(item.expr),
        values=values,
        complexity=float(item.complexity),
        family=str(item.family),
        activation_family=str(item.activation_family),
        features=tuple(int(v) for v in item.features),
        prior_corr=abs(safe_corr(values, target)),
        metadata={"source": "symbolic_grammar"},
    )


def _primitive_params_for_feature(X: np.ndarray, feature_idx: int, config: FunctionPoolPipelineConfig) -> dict[str, float]:
    params = {str(k): float(v) for k, v in dict(config.primitive_params).items()}
    col = np.asarray(X[:, int(feature_idx)], dtype=float).reshape(-1)
    finite = col[np.isfinite(col)]
    spread = 1.0
    if finite.size:
        spread = float(np.quantile(finite, 0.9) - np.quantile(finite, 0.1))
    params.setdefault("scale", float(max(1.0, spread)))
    params.setdefault("eps", 1e-3)
    return params


def _primitive_params_for_pair(X: np.ndarray, left_idx: int, right_idx: int, config: FunctionPoolPipelineConfig) -> dict[str, float]:
    left = _primitive_params_for_feature(X, int(left_idx), config)
    right = _primitive_params_for_feature(X, int(right_idx), config)
    params = {str(k): float(v) for k, v in dict(config.primitive_params).items()}
    params.setdefault("scale", float(max(left.get("scale", 1.0), right.get("scale", 1.0), 1.0)))
    params.setdefault("eps", float(max(left.get("eps", 1e-3), right.get("eps", 1e-3), 1e-8)))
    return params


def _unary_candidates(
    registry: PrimitiveRegistry,
    X: np.ndarray,
    feature_names: Sequence[str],
    feature_order: Sequence[int],
    *,
    config: FunctionPoolPipelineConfig,
) -> list[GrammarCandidate]:
    out: list[GrammarCandidate] = []
    for feature_idx in feature_order:
        idx = int(feature_idx)
        out.extend(
            generate_unary_candidates(
                registry=registry,
                base_expr=feature_expr(idx),
                base_values=np.asarray(X[:, idx], dtype=float).reshape(-1),
                base_label=str(feature_names[idx]),
                feature_ids=(idx,),
                params=_primitive_params_for_feature(X, idx, config),
                mode=str(config.mode),
                active_families=config.unary_families,
            )
        )
    return out


def _pair_candidates(
    registry: PrimitiveRegistry,
    X: np.ndarray,
    feature_names: Sequence[str],
    feature_order: Sequence[int],
    *,
    config: FunctionPoolPipelineConfig,
) -> list[GrammarCandidate]:
    out: list[GrammarCandidate] = []
    rules = registry.iter_pair_rules(mode=config.mode, families=config.pair_families)
    if not rules:
        return out
    pair_count = 0
    order = tuple(int(i) for i in feature_order)
    for pos_i, i in enumerate(order):
        for j in order[pos_i + 1 :]:
            out.extend(
                generate_pair_candidates(
                    registry=registry,
                    left_expr=feature_expr(i),
                    left_values=np.asarray(X[:, i], dtype=float).reshape(-1),
                    left_label=str(feature_names[i]),
                    right_expr=feature_expr(j),
                    right_values=np.asarray(X[:, j], dtype=float).reshape(-1),
                    right_label=str(feature_names[j]),
                    feature_ids=(int(i), int(j)),
                    params=_primitive_params_for_pair(X, i, j, config),
                    mode=str(config.mode),
                    active_families=config.pair_families,
                )
            )
            pair_count += 1
            if pair_count >= int(config.pair_top_k):
                return out
    return out


def _top_candidates(items: Sequence[GrammarCandidate], target: np.ndarray, top_k: int) -> list[GrammarCandidate]:
    ranked = sorted(
        list(items),
        key=lambda item: (-abs(safe_corr(np.asarray(item.values, dtype=float).reshape(-1), target)), float(item.complexity), item.name),
    )
    return ranked[: max(0, int(top_k))]


def _recursive_candidates(
    registry: PrimitiveRegistry,
    X: np.ndarray,
    feature_names: Sequence[str],
    target: np.ndarray,
    feature_order: Sequence[int],
    *,
    config: FunctionPoolPipelineConfig,
) -> list[GrammarCandidate]:
    out: list[GrammarCandidate] = []
    seed_cache: dict[int, tuple[GrammarCandidate, ...]] = {}

    def feature_seed_pool(idx: int) -> tuple[GrammarCandidate, ...]:
        key = int(idx)
        cached = seed_cache.get(key)
        if cached is not None:
            return cached
        base = make_seed_candidate(
            name=str(feature_names[key]),
            expr=feature_expr(key),
            values=np.asarray(X[:, key], dtype=float).reshape(-1),
            features=(key,),
            family="seed",
            activation_family="seed",
        )
        unary = generate_unary_candidates(
            registry=registry,
            base_expr=feature_expr(key),
            base_values=np.asarray(X[:, key], dtype=float).reshape(-1),
            base_label=str(feature_names[key]),
            feature_ids=(key,),
            params=_primitive_params_for_feature(X, key, config),
            mode=str(config.mode),
            active_families=config.unary_families,
        )
        top_unary = _top_candidates(unary, target, int(config.recursive_pair_seed_top_k))
        seed_cache[key] = tuple([base, *top_unary])
        return seed_cache[key]

    for idx in tuple(int(i) for i in feature_order):
        seeds = feature_seed_pool(idx)
        out.extend(
            _top_candidates(
                generate_recursive_unary_candidates(
                    registry=registry,
                    seeds=_top_candidates(seeds, target, int(config.recursive_seed_top_k)),
                    params=_primitive_params_for_feature(X, idx, config),
                    mode=str(config.mode),
                    active_families=config.unary_families,
                    max_complexity=config.recursive_max_complexity,
                ),
                target,
                int(config.recursive_seed_top_k),
            )
        )

    pair_count = 0
    order = tuple(int(i) for i in feature_order)
    for pos_i, i in enumerate(order):
        for j in order[pos_i + 1 :]:
            out.extend(
                _top_candidates(
                    generate_recursive_pair_candidates(
                        registry=registry,
                        left_seeds=feature_seed_pool(i),
                        right_seeds=feature_seed_pool(j),
                        params=_primitive_params_for_pair(X, i, j, config),
                        mode=str(config.mode),
                        active_families=config.pair_families,
                        max_complexity=config.recursive_max_complexity,
                    ),
                    target,
                    int(max(1, config.recursive_pair_seed_top_k * 3)),
                )
            )
            pair_count += 1
            if pair_count >= int(config.pair_top_k):
                return out
    return out


def _apply_family_budget(terms: Sequence[CandidateTerm], family_budget: Mapping[str, int]) -> list[CandidateTerm]:
    if not family_budget:
        return list(terms)
    used: dict[str, int] = {}
    out: list[CandidateTerm] = []
    ordered = sorted(terms, key=lambda term: (-abs(float(term.prior_corr)), float(term.complexity), term.name))
    for term in ordered:
        budget = family_budget.get(term.activation_family, family_budget.get(term.family))
        if budget is not None:
            count = used.get(term.activation_family, 0)
            if count >= int(budget):
                continue
            used[term.activation_family] = count + 1
        out.append(term)
    return out
