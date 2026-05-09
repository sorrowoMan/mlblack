from __future__ import annotations

from dataclasses import dataclass

from conditional.primitives.base import ConditionalPrimitiveSpec


@dataclass(frozen=True)
class PiecewisePrimitive:
    feature_name: str
    cut: float
    left_mode: str = "identity"
    right_mode: str = "identity"
    multiplier_feature: str | None = None

    def to_spec(self) -> ConditionalPrimitiveSpec:
        source = [str(self.feature_name)]
        if self.multiplier_feature is not None:
            source.append(str(self.multiplier_feature))
        return ConditionalPrimitiveSpec(
            name=f"piecewise_{self.feature_name}",
            family="piecewise",
            source_features=tuple(source),
            parameters={
                "cut": float(self.cut),
                "left_mode": str(self.left_mode),
                "right_mode": str(self.right_mode),
                "multiplier_feature": None if self.multiplier_feature is None else str(self.multiplier_feature),
            },
        )


__all__ = ["PiecewisePrimitive"]
