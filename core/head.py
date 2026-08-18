from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

import numpy as np

from blackbase.contracts import ComponentContract, ContractMixin

BaseDecoder = Callable[[np.ndarray, Mapping[str, Any]], Any]


@dataclass(frozen=True)
class HeadBlock:
    """Contiguous parameter block consumed by an output head."""

    name: str
    start: int
    stop: int
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def values(self, vector: np.ndarray) -> np.ndarray:
        return np.asarray(vector, dtype=float).reshape(-1)[int(self.start) : int(self.stop)]

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "start": int(self.start),
            "stop": int(self.stop),
            "metadata": dict(self.metadata),
        }


class OutputHead(ContractMixin, ABC):
    """Decoder-side output semantics.

    A representation owns base decoding. The head owns how many base parameter
    blocks are needed and how decoded base models become the final prediction
    object consumed by the problem.
    """

    name = "output_head"
    output_kind = "unknown"
    context_requires = ('base_decoder', 'candidate.unknown_state')
    context_optional = ()
    context_provides = ('candidate.output',)
    context_mutates = ()
    context_cache = ()
    requires_metrics = ()
    metrics_fallback = "strict"
    context_notes = 'Reads context fields: base_decoder, candidate.unknown_state; provides candidate.output.'
    contract = ComponentContract(
        name=name,
        requires=("base_decoder", "candidate.unknown_state"),
        provides=("candidate.output",),
        supports_batch=True,
        supports_resume=True,
    )

    @abstractmethod
    def parameter_size(self, base_dimension: int) -> int:
        """Return total parameter count required by this head."""

    @abstractmethod
    def blocks(self, base_dimension: int) -> tuple[HeadBlock, ...]:
        """Return named base-parameter blocks."""

    @abstractmethod
    def decode(
        self,
        values: np.ndarray,
        *,
        base_dimension: int,
        base_decode: BaseDecoder,
        context: Mapping[str, Any] | None = None,
    ) -> Any:
        """Decode full state values into final output object."""

    def repair_values(self, values: np.ndarray, *, base_dimension: int) -> np.ndarray:
        expected = int(self.parameter_size(base_dimension))
        arr = np.asarray(values, dtype=float).reshape(-1)
        if arr.shape[0] != expected:
            fixed = np.zeros(expected, dtype=float)
            fixed[: min(expected, arr.shape[0])] = arr[: min(expected, arr.shape[0])]
            arr = fixed
        return np.nan_to_num(arr, nan=0.0, posinf=1e6, neginf=-1e6)

    def describe(self, *, base_dimension: int | None = None) -> dict[str, Any]:
        data: dict[str, Any] = {
            "name": self.name,
            "output_kind": self.output_kind,
            "contract": self.get_contract().describe(),
        }
        if base_dimension is not None:
            data["parameter_size"] = int(self.parameter_size(int(base_dimension)))
            data["blocks"] = [block.describe() for block in self.blocks(int(base_dimension))]
        return data
