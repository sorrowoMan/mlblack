from __future__ import annotations

from dataclasses import asdict
from typing import Any

from my_project.known_relation_symbolic.config import KnownRelationSymbolicBuildConfig
from my_project.known_relation_symbolic.mlblack_side import (
    KnownRelationEvaluationProxy,
    KnownRelationEvaluationProxyConfig,
)
from my_project.known_relation_symbolic.problem import known_relation_benchmark_keys


def build_known_relation_symbolic_components(
    cfg: KnownRelationSymbolicBuildConfig | None = None,
) -> dict[str, Any]:
    resolved = cfg or KnownRelationSymbolicBuildConfig()
    proxy = KnownRelationEvaluationProxy(
        KnownRelationEvaluationProxyConfig(
            n_total=int(resolved.n_total),
            train_ratio=float(resolved.train_ratio),
            noise_std=float(resolved.noise_std),
            seed=int(resolved.seed),
        )
    )
    return {
        "config": resolved,
        "config_dict": asdict(resolved),
        "scenario_keys": known_relation_benchmark_keys(),
        "evaluation_proxy": proxy,
        "outer_solver_backend": str(resolved.outer_solver_backend),
    }


__all__ = ["KnownRelationSymbolicBuildConfig", "build_known_relation_symbolic_components"]
