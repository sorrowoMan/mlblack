from __future__ import annotations

"""Stable public import surface for solver scaffold assembly."""

from nowcasting_work_ci.nsgablack_side.build_solver import (  # noqa: F401
    NowcastingSolverBuildConfig,
    build_nowcasting_solver,
)

__all__ = ["NowcastingSolverBuildConfig", "build_nowcasting_solver"]
