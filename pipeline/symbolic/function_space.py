from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from mlblack.core.contracts import ComponentContract, ContractMixin
from mlblack.models.symbolic_normalization import expression_equivalence_key


@dataclass(frozen=True)
class CandidateTerm:
    """Materialized symbolic candidate term produced by a function-space pipeline."""

    name: str
    expr: dict[str, Any]
    values: np.ndarray
    complexity: float
    family: str
    activation_family: str
    features: tuple[int, ...]
    prior_corr: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def key(self) -> str:
        try:
            return expression_equivalence_key(self.expr)
        except Exception:
            return json.dumps(self.expr, sort_keys=True)

    def as_genome_term(self) -> dict[str, Any]:
        return {"name": self.name, "expr": dict(self.expr)}

    def describe(self, *, include_values: bool = False) -> dict[str, Any]:
        payload = {
            "name": self.name,
            "expr": dict(self.expr),
            "complexity": float(self.complexity),
            "family": self.family,
            "activation_family": self.activation_family,
            "features": list(self.features),
            "prior_corr": float(self.prior_corr),
            "metadata": dict(self.metadata),
        }
        if include_values:
            payload["values"] = np.asarray(self.values, dtype=float).reshape(-1).tolist()
        return payload


@dataclass(frozen=True)
class FunctionPool:
    """Current active symbolic candidate pool."""

    terms: tuple[CandidateTerm, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def genome(self) -> tuple[dict[str, Any], ...]:
        return tuple(term.as_genome_term() for term in self.terms)

    def by_family(self, family: str) -> tuple[CandidateTerm, ...]:
        key = str(family)
        return tuple(term for term in self.terms if term.family == key or term.activation_family == key)

    def top_by_prior_corr(self, n: int) -> "FunctionPool":
        ordered = sorted(self.terms, key=lambda term: (-abs(float(term.prior_corr)), float(term.complexity), term.name))
        return FunctionPool(tuple(ordered[: max(0, int(n))]), metadata={**dict(self.metadata), "selected_by": "prior_corr"})

    def describe(self, *, include_values: bool = False) -> dict[str, Any]:
        return {
            "n_terms": int(len(self.terms)),
            "families": sorted({term.family for term in self.terms}),
            "activation_families": sorted({term.activation_family for term in self.terms}),
            "terms": [term.describe(include_values=include_values) for term in self.terms],
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class FunctionSpace:
    """Declarative symbolic search-space plus a materialized active pool."""

    pool: FunctionPool
    feature_names: tuple[str, ...]
    registry_summary: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def describe(self, *, include_values: bool = False) -> dict[str, Any]:
        return {
            "feature_names": list(self.feature_names),
            "registry_summary": dict(self.registry_summary),
            "pool": self.pool.describe(include_values=include_values),
            "metadata": dict(self.metadata),
        }


class FunctionSpaceProvider(ContractMixin):
    """Contract marker for components that produce symbolic function pools."""

    name = "symbolic_function_space_provider"
    context_requires = ("data.X_train", "data.feature_names")
    context_optional = ("data.y_train", "feedback.residuals", "feedback.gradients", "resource.context", "signal.pool")
    context_provides = ("symbolic.function_space", "symbolic.function_pool", "symbolic.primitive_registry")
    context_mutates = ()
    context_cache = ("symbolic.function_pool",)
    requires_metrics = ()
    metrics_fallback = "strict"
    context_notes = "Builds symbolic search-space candidates as a pipeline output, not as a trainer."
    contract = ComponentContract(
        name=name,
        requires=("data.X_train", "data.feature_names"),
        optional=("data.y_train", "feedback.residuals", "feedback.gradients", "resource.context", "signal.pool"),
        provides=("symbolic.function_space", "symbolic.function_pool", "symbolic.primitive_registry"),
        cache=("symbolic.function_pool",),
        supports_batch=False,
        supports_resume=True,
        metadata={"layer": "pipeline", "family": "symbolic"},
    )


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=float).reshape(-1)
    y = np.asarray(b, dtype=float).reshape(-1)
    if x.shape[0] != y.shape[0] or x.shape[0] == 0:
        return 0.0
    xc = x - float(np.mean(x))
    yc = y - float(np.mean(y))
    denom = float(np.linalg.norm(xc) * np.linalg.norm(yc)) + 1e-12
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(xc, yc) / denom)


def dedupe_terms(terms: Sequence[CandidateTerm]) -> tuple[CandidateTerm, ...]:
    seen: set[str] = set()
    out: list[CandidateTerm] = []
    for term in terms:
        key = term.key()
        if key in seen:
            continue
        seen.add(key)
        out.append(term)
    return tuple(out)
