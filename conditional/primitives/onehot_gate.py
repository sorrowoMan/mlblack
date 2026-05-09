from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from conditional.primitives.base import ConditionalPrimitiveSpec


@dataclass(frozen=True)
class OneHotGate:
    feature_name: str
    categories: Sequence[str]

    def to_spec(self) -> ConditionalPrimitiveSpec:
        return ConditionalPrimitiveSpec(
            name=f"onehot_gate_{self.feature_name}",
            family="gate_onehot",
            source_features=(str(self.feature_name),),
            parameters={"categories": tuple(str(v) for v in self.categories)},
            output_mode="vector",
        )


__all__ = ["OneHotGate"]
