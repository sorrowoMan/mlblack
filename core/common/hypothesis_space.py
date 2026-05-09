from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Protocol


class HypothesisSpace(Protocol):
    """Model/hypothesis-space abstraction: defines y = f(W, X)."""

    name: str

    def forward(self, X: Any) -> Any:
        ...

    def parameters(self) -> Iterable[Any]:
        ...


@dataclass
class TorchModuleHypothesisSpace:
    """Thin wrapper to expose torch modules via a stable hypothesis-space API."""

    module: Any
    family: str
    name: str = "torch_module"

    def forward(self, X: Any) -> Any:
        return self.module(X)

    def parameters(self) -> Iterable[Any]:
        return self.module.parameters()
