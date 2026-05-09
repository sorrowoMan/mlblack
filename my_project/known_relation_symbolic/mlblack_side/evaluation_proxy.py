from __future__ import annotations

from core.symbolic.benchmark.outer_proxy import (
    SymbolicOuterEvaluationProxyConfig,
    SymbolicScenarioEvaluationProxy,
)

from my_project.known_relation_symbolic.pipeline.bundle import build_known_relation_bundle


KnownRelationEvaluationProxyConfig = SymbolicOuterEvaluationProxyConfig


def _known_relation_bundle_builder(
    scenario_key: str,
    n_total: int,
    train_ratio: float,
    noise_std: float,
    seed: int,
):
    return build_known_relation_bundle(
        benchmark_key=str(scenario_key),
        n_total=int(n_total),
        train_ratio=float(train_ratio),
        noise_std=float(noise_std),
        seed=int(seed),
    )


class KnownRelationEvaluationProxy(SymbolicScenarioEvaluationProxy):
    """Known-relation project instance of the generic symbolic scenario evaluation proxy."""

    def __init__(self, config: KnownRelationEvaluationProxyConfig | None = None) -> None:
        super().__init__(bundle_builder=_known_relation_bundle_builder, config=config)


__all__ = ["KnownRelationEvaluationProxy", "KnownRelationEvaluationProxyConfig"]
