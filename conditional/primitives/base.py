from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, runtime_checkable


@dataclass(frozen=True)
class ConditionalPrimitiveSpec:
    name: str
    family: str
    source_features: tuple[str, ...]
    parameters: Mapping[str, Any] = field(default_factory=dict)
    output_mode: str = "scalar"
    learnable: bool = False


@runtime_checkable
class ConditionalPrimitive(Protocol):
    def to_spec(self) -> ConditionalPrimitiveSpec: ...


__all__ = ["ConditionalPrimitive", "ConditionalPrimitiveSpec"]
