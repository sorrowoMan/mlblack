from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from blackbase.contracts import ComponentContract
from mlblack.core.head import OutputHead
from mlblack.core.representation import ModelRepresentation
from mlblack.core.types import UnknownState
from mlblack.models.symbolic import SymbolicBasisSetModel, SymbolicExpressionModel
from mlblack.representations.codecs import SymbolicExpressionCodec, SymbolicExpressionCodecConfig, SymbolicMultiExpressionCodec, SymbolicMultiExpressionCodecConfig
from mlblack.representations.heads import PointHead, SymbolicBasisSetHead


@dataclass(frozen=True)
class SymbolicExpressionConfig:
    input_dim: int
    expression: Mapping[str, Any]
    name: str = "symbolic_expression"
    feature_names: tuple[str, ...] = tuple()


class SymbolicExpressionRepresentation(ModelRepresentation):
    """Fixed symbolic expression -> point/interval/probability head."""

    name = "symbolic_expression"
    context_requires = ("candidate.unknown_state", "symbolic.expression_spec")
    context_optional = ("symbolic.parameter_values", "data.feature_names")
    context_provides = ("candidate.output", "candidate.model", "model.predict", "symbolic.parameter_specs")
    context_mutates = ("candidate.repaired_state",)
    context_cache = ()
    requires_metrics = ()
    metrics_fallback = "strict"
    context_notes = "Decodes fixed symbolic expression parameters into a model; does not search symbolic structure."
    contract = ComponentContract(
        name=name,
        requires=("candidate.unknown_state", "symbolic.expression_spec"),
        optional=("symbolic.parameter_values", "data.feature_names"),
        provides=("candidate.output", "candidate.model", "model.predict", "symbolic.parameter_specs"),
        mutates=("candidate.repaired_state",),
        supports_gradient=True,
        supports_batch=True,
        supports_resume=True,
        metadata={"family": "symbolic", "structure_search": False},
    )

    def __init__(self, config: SymbolicExpressionConfig, head: OutputHead | None = None) -> None:
        self.config = config
        self.codec = SymbolicExpressionCodec(
            SymbolicExpressionCodecConfig(
                input_dim=int(config.input_dim),
                expression=config.expression,
                name=str(config.name),
                feature_names=tuple(config.feature_names),
                representation_name=self.name,
            )
        )
        self.head = head or PointHead()
        self.base_dimension = int(self.codec.base_dimension)
        self.dimension = int(self.head.parameter_size(self.base_dimension))

    def init(self, context: Mapping[str, Any]) -> UnknownState:
        _ = context
        values = np.zeros(self.dimension, dtype=float)
        base_values = self.codec.init_values()
        values[: base_values.shape[0]] = base_values
        return UnknownState(values=values, metadata={"source": "symbolic_expression_init"})

    def encode(self, model: Any, context: Mapping[str, Any] | None = None) -> UnknownState:
        _ = context
        if self.head.output_kind != "point":
            raise NotImplementedError("encoding is only implemented for point symbolic head")
        if not isinstance(model, SymbolicExpressionModel):
            raise TypeError("SymbolicExpressionRepresentation can only encode SymbolicExpressionModel")
        return UnknownState(values=self.codec.encode(model), metadata={"source": "encoded_symbolic_model"})

    def decode(self, state: UnknownState, context: Mapping[str, Any] | None = None) -> Any:
        ctx = dict(context or {})
        arr = np.asarray(state.values, dtype=float).reshape(-1)
        if arr.shape[0] != self.dimension:
            raise ValueError(f"state dimension {arr.shape[0]} does not match representation dimension {self.dimension}")
        return self.head.decode(
            arr,
            base_dimension=self.base_dimension,
            base_decode=self._decode_base,
            context=ctx,
        )

    def _decode_base(self, values: np.ndarray, context: Mapping[str, Any] | None = None) -> SymbolicExpressionModel:
        return self.codec.decode(values, dict(context or {}))

    def repair(self, state: UnknownState, context: Mapping[str, Any] | None = None) -> UnknownState:
        _ = context
        arr = self.head.repair_values(state.values, base_dimension=self.base_dimension)
        return state.with_values(arr)

    def get_contract(self) -> ComponentContract:
        return self.contract.merged(self.head.get_contract(), name=self.name)

    def describe(self) -> Mapping[str, Any]:
        return {
            "name": self.name,
            "dimension": int(self.dimension),
            "base_dimension": int(self.base_dimension),
            "codec": self.codec.describe(),
            "head": self.head.describe(base_dimension=self.base_dimension),
        }


@dataclass(frozen=True)
class SymbolicBasisSetConfig:
    input_dim: int
    genome: tuple[Mapping[str, Any], ...]
    name: str = "symbolic_basis_set"
    feature_names: tuple[str, ...] = tuple()


class SymbolicBasisSetRepresentation(ModelRepresentation):
    """Fixed multi-expression symbolic candidate -> basis-set output."""

    name = "symbolic_basis_set"
    context_requires = ("candidate.unknown_state", "symbolic.genome")
    context_optional = ("symbolic.parameter_values", "data.feature_names")
    context_provides = ("candidate.symbolic_basis_model", "model.transform", "symbolic.basis_model")
    context_mutates = ("candidate.repaired_state",)
    context_cache = ()
    requires_metrics = ()
    metrics_fallback = "strict"
    context_notes = "Decodes fixed multi-symbol genome into SymbolicBasisSetModel; outer structure search is external."
    contract = ComponentContract(
        name=name,
        requires=("candidate.unknown_state", "symbolic.genome"),
        optional=("symbolic.parameter_values", "data.feature_names"),
        provides=("candidate.symbolic_basis_model", "model.transform", "symbolic.basis_model"),
        mutates=("candidate.repaired_state",),
        supports_gradient=False,
        supports_batch=True,
        supports_resume=True,
        metadata={"family": "symbolic", "head": "basis_set", "structure_search": False},
    )

    def __init__(self, config: SymbolicBasisSetConfig, head: SymbolicBasisSetHead | None = None) -> None:
        self.config = config
        self.codec = SymbolicMultiExpressionCodec(
            SymbolicMultiExpressionCodecConfig(
                input_dim=int(config.input_dim),
                genome=tuple(config.genome),
                name=str(config.name),
                feature_names=tuple(config.feature_names),
                representation_name=self.name,
            )
        )
        block_names = tuple(name for name, _start, _stop in self.codec.blocks())
        self.head = head or SymbolicBasisSetHead(block_names=block_names, block_dimensions=self.codec.block_dimensions)
        self.base_dimension = 0
        self.dimension = int(self.head.parameter_size(self.base_dimension))

    def init(self, context: Mapping[str, Any]) -> UnknownState:
        _ = context
        return UnknownState(values=self.codec.init_values(), metadata={"source": "symbolic_basis_set_init"})

    def decode(self, state: UnknownState, context: Mapping[str, Any] | None = None) -> SymbolicBasisSetModel:
        ctx = dict(context or {})
        arr = np.asarray(state.values, dtype=float).reshape(-1)
        if arr.shape[0] != self.dimension:
            raise ValueError(f"state dimension {arr.shape[0]} does not match representation dimension {self.dimension}")
        return self.head.decode(
            arr,
            base_dimension=self.base_dimension,
            base_decode=self._decode_base,
            context=ctx,
        )

    def _decode_base(self, values: np.ndarray, context: Mapping[str, Any] | None = None) -> SymbolicExpressionModel:
        ctx = dict(context or {})
        index = int(ctx.get("symbolic.expression_index", 0))
        return self.codec.decode_term(index, values, ctx)

    def repair(self, state: UnknownState, context: Mapping[str, Any] | None = None) -> UnknownState:
        _ = context
        return state.with_values(self.codec.repair_values(state.values))

    def get_contract(self) -> ComponentContract:
        return self.contract.merged(self.head.get_contract(), name=self.name)

    def describe(self) -> Mapping[str, Any]:
        return {
            "name": self.name,
            "dimension": int(self.dimension),
            "codec": self.codec.describe(),
            "head": self.head.describe(base_dimension=self.base_dimension),
        }
