from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from mlblack.pipeline.symbolic import CandidateTerm, FunctionPool


@dataclass(frozen=True)
class FunctionPoolIndexSearchSpace:
    """Index-coded outer search space over a symbolic FunctionPool."""

    function_pool: FunctionPool
    dimension: int
    allow_duplicates: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def bounds(self) -> dict[str, list[float]]:
        hi = float(max(0, len(self.function_pool.terms) - 1))
        return {f"x{i}": [0.0, hi] for i in range(int(self.dimension))}

    def decode_indices(self, candidate: Sequence[float] | np.ndarray) -> tuple[int, ...]:
        arr = np.asarray(candidate, dtype=float).reshape(int(self.dimension))
        hi = max(0, len(self.function_pool.terms) - 1)
        indices = tuple(int(np.clip(np.round(value), 0, hi)) for value in arr)
        if self.allow_duplicates:
            return indices
        seen: set[int] = set()
        out: list[int] = []
        for index in indices:
            if index in seen:
                for replacement in range(hi + 1):
                    if replacement not in seen:
                        index = int(replacement)
                        break
            seen.add(int(index))
            out.append(int(index))
        return tuple(out)

    def decode_terms(self, candidate: Sequence[float] | np.ndarray) -> tuple[CandidateTerm, ...]:
        return tuple(self.function_pool.terms[index] for index in self.decode_indices(candidate))

    def describe(self) -> dict[str, Any]:
        return {
            "name": "function_pool_index_search_space",
            "dimension": int(self.dimension),
            "bounds": self.bounds(),
            "allow_duplicates": bool(self.allow_duplicates),
            "function_pool": self.function_pool.describe(include_values=False),
            "metadata": dict(self.metadata),
        }


__all__ = ["FunctionPoolIndexSearchSpace"]
