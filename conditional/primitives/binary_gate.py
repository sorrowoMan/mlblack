from __future__ import annotations

from dataclasses import dataclass

from conditional.primitives.base import ConditionalPrimitiveSpec


@dataclass(frozen=True)
class BinaryGate:
    feature_name: str
    threshold: float = 0.5
    positive_value: float = 1.0
    negative_value: float = 0.0

    def to_spec(self) -> ConditionalPrimitiveSpec:
        return ConditionalPrimitiveSpec(
            name=f"binary_gate_{self.feature_name}",
            family="gate_binary",
            source_features=(str(self.feature_name),),
            parameters={
                "threshold": float(self.threshold),
                "positive_value": float(self.positive_value),
                "negative_value": float(self.negative_value),
            },
        )


__all__ = ["BinaryGate"]
