from __future__ import annotations

from core.symbolic.benchmark.bundle_pipeline import build_symbolic_benchmark_bundle

from my_project.known_relation_symbolic.problem.registry import get_known_relation_benchmark
from my_project.known_relation_symbolic.problem.specs import KnownRelationBenchmarkDefinition


def build_known_relation_bundle(
    *,
    benchmark_key: str,
    n_total: int,
    train_ratio: float,
    noise_std: float,
    seed: int,
):
    definition = get_known_relation_benchmark(benchmark_key)
    return build_symbolic_benchmark_bundle(
        definition=definition,
        n_total=int(n_total),
        train_ratio=float(train_ratio),
        noise_std=float(noise_std),
        seed=int(seed),
    )


__all__ = [
    "KnownRelationBenchmarkDefinition",
    "build_known_relation_bundle",
]
