from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from mlblack.core.contracts import ComponentContract
from mlblack.core.head import BaseDecoder, HeadBlock, OutputHead
from mlblack.models.symbolic import SymbolicBasisSetModel, SymbolicExpressionModel


@dataclass(frozen=True)
class SymbolicBasisSetHead(OutputHead):
    """Multi-symbol head that splits one state into expression blocks."""

    block_names: tuple[str, ...]
    block_dimensions: tuple[int, ...]
    output_name: str = "symbolic_basis_set"

    name = "symbolic_basis_set"
    output_kind = "symbolic_basis_set"
    context_requires = ("base_decoder", "candidate.unknown_state", "symbolic.genome")
    context_optional = ("symbolic.parameter_specs",)
    context_provides = ("candidate.symbolic_basis_model", "model.transform", "symbolic.basis_model")
    context_mutates = ()
    context_cache = ()
    requires_metrics = ()
    metrics_fallback = "strict"
    context_notes = "Splits a symbolic state into expression blocks and returns a SymbolicBasisSetModel."
    contract = ComponentContract(
        name=name,
        requires=("base_decoder", "candidate.unknown_state", "symbolic.genome"),
        optional=("symbolic.parameter_specs",),
        provides=("candidate.symbolic_basis_model", "model.transform", "symbolic.basis_model"),
        supports_batch=True,
        supports_resume=True,
        metadata={"head": "symbolic_basis_set"},
    )

    def __post_init__(self) -> None:
        if len(self.block_names) != len(self.block_dimensions):
            raise ValueError("block_names and block_dimensions must have the same length")
        if any(int(dim) < 0 for dim in self.block_dimensions):
            raise ValueError("block_dimensions must be non-negative")

    def parameter_size(self, base_dimension: int) -> int:
        _ = base_dimension
        return int(sum(int(dim) for dim in self.block_dimensions))

    def blocks(self, base_dimension: int) -> tuple[HeadBlock, ...]:
        _ = base_dimension
        out: list[HeadBlock] = []
        start = 0
        for index, (name, dim) in enumerate(zip(self.block_names, self.block_dimensions)):
            stop = start + int(dim)
            out.append(
                HeadBlock(
                    str(name),
                    start,
                    stop,
                    metadata={"symbolic.expression_index": int(index), "symbolic.block_dimension": int(dim)},
                )
            )
            start = stop
        return tuple(out)

    def decode(
        self,
        values: np.ndarray,
        *,
        base_dimension: int,
        base_decode: BaseDecoder,
        context: Mapping[str, Any] | None = None,
    ) -> SymbolicBasisSetModel:
        ctx = dict(context or {})
        arr = self.repair_values(values, base_dimension=base_dimension)
        atoms: list[SymbolicExpressionModel] = []
        for index, block in enumerate(self.blocks(base_dimension)):
            model = base_decode(
                block.values(arr),
                {
                    **ctx,
                    "head.block": block.name,
                    "symbolic.expression_index": int(index),
                    "symbolic.expression_name": block.name,
                },
            )
            if not isinstance(model, SymbolicExpressionModel):
                raise TypeError("SymbolicBasisSetHead expects base_decode to return SymbolicExpressionModel")
            atoms.append(model)
        return SymbolicBasisSetModel(
            atoms=tuple(atoms),
            name=str(self.output_name),
            metadata={"head": self.name, "n_atoms": int(len(atoms))},
        )

    def describe(self, *, base_dimension: int | None = None) -> dict[str, Any]:
        data = super().describe(base_dimension=0 if base_dimension is None else int(base_dimension))
        data["block_names"] = list(self.block_names)
        data["block_dimensions"] = [int(dim) for dim in self.block_dimensions]
        return data


def build_symbolic_basis_head(
    block_names: Sequence[str],
    block_dimensions: Sequence[int],
    *,
    output_name: str = "symbolic_basis_set",
) -> SymbolicBasisSetHead:
    return SymbolicBasisSetHead(
        block_names=tuple(str(name) for name in block_names),
        block_dimensions=tuple(int(dim) for dim in block_dimensions),
        output_name=str(output_name),
    )
