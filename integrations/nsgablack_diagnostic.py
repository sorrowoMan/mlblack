"""Diagnostic Case assembly on the canonical NSGABlack control plane."""

from __future__ import annotations

from typing import Any, Mapping

from nsgablack.core import BudgetController

from mlblack.core.diagnostic import (
    DiagnosticProblem,
    DiagnosticRepresentation,
    DiagnosticRunner,
)
from mlblack.integrations.nsgablack_control import build_learning_solver
from mlblack.integrations.nsgablack_optimization import build_optimization_adapter


def build_diagnostic_solver(
    runner: DiagnosticRunner,
    *,
    name: str,
    resource_context: Mapping[str, Any] | None = None,
) -> Any:
    """Build a one-evaluation diagnostic without a private runner lifecycle."""

    solver = build_learning_solver(
        problem=DiagnosticProblem(runner, name=f"{name}_problem"),
        representation=DiagnosticRepresentation(),
        adapter=build_optimization_adapter("evaluation.fixed"),
        run_name=str(name),
        resource_context=resource_context,
    )
    solver.register_controller(
        BudgetController(max_generations=1, name=f"{name}.one_shot")
    )
    return solver


__all__ = ["build_diagnostic_solver"]
