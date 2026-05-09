from __future__ import annotations

from dataclasses import dataclass

from conditional.composer.base import ComposedConditionalTask
from conditional.composer.spec import RoutePlusPrimitivesSpec


@dataclass(frozen=True)
class RoutePlusPrimitivesComposer:
    spec: RoutePlusPrimitivesSpec

    def compose(self) -> ComposedConditionalTask:
        return ComposedConditionalTask(
            name="route_plus_primitives",
            mode="route_plus_primitives",
            router_policy=self.spec.router_policy,
            primitives=tuple(self.spec.primitives),
            metadata={
                "branch_formula_name": str(self.spec.branch_formula_name),
                "primitive_count": int(len(self.spec.primitives)),
            },
        )


__all__ = ["RoutePlusPrimitivesComposer"]
