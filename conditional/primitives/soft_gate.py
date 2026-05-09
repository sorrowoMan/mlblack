from __future__ import annotations

from dataclasses import dataclass

from conditional.primitives.base import ConditionalPrimitiveSpec


@dataclass(frozen=True)
class SoftGatePrimitive:
    feature_name: str
    cut: float
    slope: float = 4.0
    multiplier_feature: str | None = None

    def to_spec(self) -> ConditionalPrimitiveSpec:
        source = [str(self.feature_name)]
        if self.multiplier_feature is not None:
            source.append(str(self.multiplier_feature))
        return ConditionalPrimitiveSpec(
            name=f"soft_gate_{self.feature_name}",
            family="gate_soft",
            source_features=tuple(source),
            parameters={
                "cut": float(self.cut),
                "slope": float(self.slope),
                "multiplier_feature": None if self.multiplier_feature is None else str(self.multiplier_feature),
            },
        )


__all__ = ["SoftGatePrimitive"]
