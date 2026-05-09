from __future__ import annotations

from dataclasses import dataclass

from conditional.primitives.base import ConditionalPrimitiveSpec


@dataclass(frozen=True)
class HingePrimitive:
    feature_name: str
    cut: float
    direction: str = "positive"
    multiplier_feature: str | None = None

    def to_spec(self) -> ConditionalPrimitiveSpec:
        source = [str(self.feature_name)]
        if self.multiplier_feature is not None:
            source.append(str(self.multiplier_feature))
        return ConditionalPrimitiveSpec(
            name=f"hinge_{self.direction}_{self.feature_name}",
            family="piecewise_hinge",
            source_features=tuple(source),
            parameters={
                "cut": float(self.cut),
                "direction": str(self.direction),
                "multiplier_feature": None if self.multiplier_feature is None else str(self.multiplier_feature),
            },
        )


__all__ = ["HingePrimitive"]
