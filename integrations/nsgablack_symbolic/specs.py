from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class SymbolicStagePlan:
    """JSON-compatible stage plan consumed by nsgablack-side orchestration."""

    name: str
    objective_kind: str
    function_pool: Mapping[str, Any] = field(default_factory=dict)
    outer_solver: Mapping[str, Any] = field(default_factory=dict)
    inner_training: Mapping[str, Any] = field(default_factory=dict)
    resource_policy: Mapping[str, Any] = field(default_factory=dict)
    report_fields: tuple[str, ...] = tuple()

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "objective_kind": self.objective_kind,
            "function_pool": dict(self.function_pool),
            "outer_solver": dict(self.outer_solver),
            "inner_training": dict(self.inner_training),
            "resource_policy": dict(self.resource_policy),
            "report_fields": list(self.report_fields),
        }


@dataclass(frozen=True)
class SymbolicOrthogonalNestedPlan:
    """Two-stage symbolic orthogonal plan declaration.

    Stage 1 searches/fits an orthogonal symbolic basis set.
    Stage 2 searches/fits task expressions over the Stage 1 basis atoms.
    """

    basis_stage: SymbolicStagePlan
    task_stage: SymbolicStagePlan
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "basis_stage": self.basis_stage.as_dict(),
            "task_stage": self.task_stage.as_dict(),
            "metadata": dict(self.metadata),
        }

