"""Stable optimization-method assembly without exposing repository ownership."""

from __future__ import annotations

from typing import Any

from nsgablack.adapters.gaussian_search import (
    GaussianSearchAdapter,
    GaussianSearchConfig,
)
from nsgablack.adapters.gradient_optimizer import (
    GradientOptimizerAdapter,
    GradientOptimizerConfig,
)
from nsgablack.adapters.fixed_candidate import FixedCandidateAdapter


def build_optimization_adapter(method: str, **parameters: Any) -> Any:
    """Resolve a stable method ID to its nsgablack strategy implementation."""

    method_id = _method_id(method)
    values = dict(parameters)
    if method_id.startswith("gradient."):
        state_gateway = values.pop("state_gateway", None)
        prefer_provider_transition = bool(
            values.pop("prefer_provider_transition", False)
        )
        if "max_grad_norm" in values and "max_gradient_norm" not in values:
            values["max_gradient_norm"] = values.pop("max_grad_norm")
        return GradientOptimizerAdapter(
            GradientOptimizerConfig.from_method(method_id, **values),
            state_gateway=state_gateway,
            prefer_provider_transition=prefer_provider_transition,
        )
    if method_id == "search.random_gaussian":
        return GaussianSearchAdapter(GaussianSearchConfig(**values))
    if method_id == "evaluation.fixed":
        if values:
            raise TypeError("evaluation.fixed does not accept parameters")
        return FixedCandidateAdapter()
    raise AssertionError(f"unhandled optimization method: {method_id}")


def _method_id(value: str) -> str:
    normalized = str(value or "").strip().lower()
    aliases = {
        "sgd": "gradient.sgd",
        "gd": "gradient.sgd",
        "gradient_descent": "gradient.sgd",
        "adam": "gradient.adam",
        "adamw": "gradient.adamw",
        "random": "search.random_gaussian",
        "random_search": "search.random_gaussian",
        "gaussian": "search.random_gaussian",
        "gaussian_search": "search.random_gaussian",
        "fixed": "evaluation.fixed",
        "fixed_candidate": "evaluation.fixed",
    }
    normalized = aliases.get(normalized, normalized)
    allowed = {
        "gradient.sgd",
        "gradient.adam",
        "gradient.adamw",
        "search.random_gaussian",
        "evaluation.fixed",
    }
    if normalized not in allowed:
        raise ValueError(
            "unknown optimization method; expected gradient.sgd, gradient.adam, "
            "gradient.adamw, search.random_gaussian, or evaluation.fixed"
        )
    return normalized


__all__ = ["build_optimization_adapter"]
