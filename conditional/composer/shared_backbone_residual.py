from __future__ import annotations

from dataclasses import dataclass

from conditional.composer.base import ComposedConditionalTask
from conditional.composer.spec import SharedBackboneResidualSpec


@dataclass(frozen=True)
class SharedBackboneResidualComposer:
    spec: SharedBackboneResidualSpec

    def compose(self) -> ComposedConditionalTask:
        return ComposedConditionalTask(
            name="shared_backbone_regime_residual",
            mode="shared_backbone_regime_residual",
            router_policy=self.spec.router_policy,
            metadata={
                "backbone_name": str(self.spec.backbone_name),
                "residual_name": str(self.spec.residual_name),
                "residual_target": str(self.spec.residual_target),
                "share_candidate_pool": bool(self.spec.share_candidate_pool),
            },
        )


__all__ = ["SharedBackboneResidualComposer"]
