# -*- coding: utf-8 -*-
"""Problem-layer configuration: Problem registry + builder."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional


@dataclass(frozen=True)
class ProblemSpec:
    key: str
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProblemRegistry:
    registry: tuple[ProblemSpec, ...] = ()


def get_problem_registry() -> ProblemRegistry:
    return ProblemRegistry(
        registry=(
            ProblemSpec(key="example", params={}),
        )
    )


ProblemBuilder = Callable[[Dict[str, Any]], object]
_PROBLEM_BUILDERS: Dict[str, ProblemBuilder] = {}


def register_problem_builder(key: str, builder: ProblemBuilder) -> None:
    _PROBLEM_BUILDERS[str(key).strip().lower()] = builder


def _find_spec(registry: ProblemRegistry, key: str) -> ProblemSpec:
    lookup = str(key).strip().lower()
    for spec in tuple(registry.registry or ()):
        if str(spec.key).strip().lower() == lookup:
            return spec
    raise ValueError(f"Problem key not registered: {key}")


def _build_problem_from_spec(spec: ProblemSpec) -> object:
    key = str(spec.key).strip().lower()
    builder = _PROBLEM_BUILDERS.get(key)
    if builder is None:
        raise ValueError(f"Unknown problem key: {spec.key}")
    return builder(dict(spec.params or {}))


def build_problem(registry: ProblemRegistry, key: str) -> object:
    spec = _find_spec(registry, key)
    return _build_problem_from_spec(spec)


def _register_builtin_problems() -> None:
    from problem.example_problem import ExampleRegressionProblem
    def _example_builder(params: Dict[str, Any]) -> ExampleRegressionProblem:
        return ExampleRegressionProblem()
    register_problem_builder("example", _example_builder)

_register_builtin_problems()
