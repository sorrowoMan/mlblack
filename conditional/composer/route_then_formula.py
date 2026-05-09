from __future__ import annotations

from dataclasses import dataclass

from conditional.composer.base import ComposedConditionalTask
from conditional.composer.spec import RouteThenFormulaSpec


@dataclass(frozen=True)
class RouteThenFormulaComposer:
    spec: RouteThenFormulaSpec

    def compose(self) -> ComposedConditionalTask:
        return ComposedConditionalTask(
            name="route_then_formula",
            mode="route_then_formula",
            router_policy=self.spec.router_policy,
            metadata={
                "branch_formula_name": str(self.spec.branch_formula_name),
                "share_candidate_pool": bool(self.spec.share_candidate_pool),
            },
        )


__all__ = ["RouteThenFormulaComposer"]
