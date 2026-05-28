# -*- coding: utf-8 -*-
"""Problem config: TSNEProblem registry."""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict


@dataclass(frozen=True)
class ProblemSpec:
    key: str
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProblemRegistry:
    registry: tuple[ProblemSpec, ...] = ()


def get_problem_registry() -> ProblemRegistry:
    return ProblemRegistry(
        registry=(ProblemSpec(key="tsne", params={}),)
    )


ProblemBuilder = Callable[[Dict[str, Any]], object]
_PROBLEM_BUILDERS: Dict[str, ProblemBuilder] = {}


def register_problem_builder(key: str, builder: ProblemBuilder) -> None:
    _PROBLEM_BUILDERS[str(key).strip().lower()] = builder


def build_problem(registry: ProblemRegistry, key: str) -> object:
    for spec in registry.registry:
        if spec.key == key:
            builder = _PROBLEM_BUILDERS.get(key)
            if builder:
                return builder(dict(spec.params or {}))
    raise ValueError(f"Problem key not found: {key}")


def _register_builtin() -> None:
    from problem.tsne_problem import TSNEProblem
    register_problem_builder("tsne", lambda p: TSNEProblem())

_register_builtin()
