from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class SymbolicOrthogonalNestedCaseConfig:
    output_dir: str = str(Path(__file__).resolve().parents[1] / "runs")
    seed: int = 11
    n_samples: int = 96
    valid_fraction: float = 0.25

    stage1_basis_size: int = 3
    stage1_pool_max_terms: int = 40
    stage1_generations: int = 2
    stage1_pop_size: int = 6
    stage1_offspring_size: int = 6
    stage1_inner_steps: int = 10
    stage1_inner_population_size: int = 8

    stage2_task_terms: int = 3
    stage2_pool_max_terms: int = 48
    stage2_generations: int = 2
    stage2_pop_size: int = 6
    stage2_offspring_size: int = 6
    stage2_inner_steps: int = 20
    stage2_inner_population_size: int = 10
    stage2_learning_rate: float = 0.03
    stage2_task_kind: str = "regression"
    stage2_head_kind: str = "point"

    mutation_sigma: float = 0.75
    crossover_rate: float = 0.85
    enable_path_memory: bool = False
    enable_graph_cache: bool = True
    resource_context: Mapping[str, Any] = field(default_factory=lambda: {"threads": 1, "device": "cpu"})

    def output_root(self, suite_id: str) -> Path:
        return Path(self.output_dir).expanduser().resolve() / str(suite_id)
