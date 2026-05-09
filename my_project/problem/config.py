from __future__ import annotations

from my_project.config.schema import ProblemConfig
from my_project.problem.example_problem import ProblemContext


def build_problem_context(cfg: ProblemConfig) -> ProblemContext:
    return ProblemContext(data_csv=str(cfg.data_csv), target_col=str(cfg.target_col))
