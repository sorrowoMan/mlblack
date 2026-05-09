from __future__ import annotations

from dataclasses import dataclass, field

from conditional.primitives import ConditionalPrimitiveSpec
from conditional.router import RouterPolicyAdapter


@dataclass(frozen=True)
class RouteThenFormulaSpec:
    router_policy: RouterPolicyAdapter
    branch_formula_name: str = "symbolic_formula"
    share_candidate_pool: bool = True


@dataclass(frozen=True)
class SharedBackboneResidualSpec:
    router_policy: RouterPolicyAdapter
    backbone_name: str = "shared_backbone"
    residual_name: str = "regime_residual"
    residual_target: str = "residual"
    share_candidate_pool: bool = True


@dataclass(frozen=True)
class RoutePlusPrimitivesSpec:
    router_policy: RouterPolicyAdapter
    primitives: tuple[ConditionalPrimitiveSpec, ...] = field(default_factory=tuple)
    branch_formula_name: str = "symbolic_formula"


__all__ = [
    "RoutePlusPrimitivesSpec",
    "RouteThenFormulaSpec",
    "SharedBackboneResidualSpec",
]
