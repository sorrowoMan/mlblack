from __future__ import annotations

from my_project.known_relation_symbolic.problem.registry import (
    KNOWN_RELATION_BENCHMARKS,
    get_known_relation_benchmark,
    known_relation_benchmark_keys,
)


def build_problem_catalog() -> dict[str, object]:
    return {
        "scenario_keys": known_relation_benchmark_keys(),
        "benchmarks": dict(KNOWN_RELATION_BENCHMARKS),
    }


__all__ = [
    "KNOWN_RELATION_BENCHMARKS",
    "build_problem_catalog",
    "get_known_relation_benchmark",
    "known_relation_benchmark_keys",
]
