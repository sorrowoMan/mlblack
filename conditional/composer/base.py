from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, runtime_checkable

from conditional.primitives import ConditionalPrimitiveSpec
from conditional.router import RouterPolicyAdapter


@dataclass(frozen=True)
class ComposedConditionalTask:
    name: str
    mode: str
    router_policy: RouterPolicyAdapter | None = None
    primitives: tuple[ConditionalPrimitiveSpec, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class ConditionalComposer(Protocol):
    def compose(self) -> ComposedConditionalTask: ...


__all__ = ["ComposedConditionalTask", "ConditionalComposer"]
