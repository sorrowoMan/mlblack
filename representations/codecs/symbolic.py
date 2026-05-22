from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from mlblack.models.symbolic import (
    ParameterSpec,
    SymbolicExpressionModel,
    collect_parameter_specs,
    normalize_expression,
    normalize_genome,
    parameter_values_from_vector,
)


@dataclass(frozen=True)
class SymbolicExpressionCodecConfig:
    input_dim: int
    expression: Mapping[str, Any]
    name: str = "symbolic_expression"
    feature_names: tuple[str, ...] = tuple()
    representation_name: str = "symbolic_expression"


class SymbolicExpressionCodec:
    """Fixed symbolic expression codec.

    The structure is fixed. UnknownState values are only the trainable numeric
    parameter slots collected from the expression.
    """

    def __init__(self, config: SymbolicExpressionCodecConfig) -> None:
        self.config = config
        self.expression = normalize_expression(config.expression, input_dim=int(config.input_dim))
        self.parameter_specs = collect_parameter_specs(self.expression)
        self.trainable_specs = tuple(spec for spec in self.parameter_specs if bool(spec.trainable))
        self.base_dimension = int(len(self.trainable_specs))

    def init_values(self) -> np.ndarray:
        return np.asarray([spec.init for spec in self.trainable_specs], dtype=float).reshape(-1)

    def repair_values(self, values: Sequence[float] | np.ndarray) -> np.ndarray:
        arr = np.asarray(values, dtype=float).reshape(-1)
        if arr.shape[0] != self.base_dimension:
            fixed = np.zeros(self.base_dimension, dtype=float)
            fixed[: min(arr.shape[0], self.base_dimension)] = arr[: min(arr.shape[0], self.base_dimension)]
            arr = fixed
        arr = np.nan_to_num(arr, nan=0.0, posinf=1e6, neginf=-1e6)
        for i, spec in enumerate(self.trainable_specs):
            arr[i] = spec.clamp(float(arr[i]))
        return np.asarray(arr, dtype=float).reshape(-1)

    def encode(self, model: SymbolicExpressionModel) -> np.ndarray:
        return np.asarray(model.parameter_vector(trainable_only=True), dtype=float).reshape(-1)

    def decode(self, values: Sequence[float] | np.ndarray, context: Mapping[str, Any] | None = None) -> SymbolicExpressionModel:
        ctx = dict(context or {})
        arr = self.repair_values(values)
        params = parameter_values_from_vector(self.parameter_specs, arr)
        return SymbolicExpressionModel(
            name=str(ctx.get("symbolic.expression_name", self.config.name)),
            expression=dict(self.expression),
            parameter_specs=tuple(self.parameter_specs),
            param_values=params,
            input_dim=int(self.config.input_dim),
            feature_names=tuple(self.config.feature_names),
            metadata={
                "representation": self.config.representation_name,
                "head_block": ctx.get("head.block"),
                "symbolic.expression_index": ctx.get("symbolic.expression_index"),
            },
        )

    def describe(self) -> dict[str, Any]:
        return {
            "codec": "symbolic_expression",
            "base_dimension": int(self.base_dimension),
            "input_dim": int(self.config.input_dim),
            "name": self.config.name,
            "feature_names": list(self.config.feature_names),
            "parameter_specs": [spec.as_dict() for spec in self.parameter_specs],
            "expression": dict(self.expression),
        }


@dataclass(frozen=True)
class SymbolicMultiExpressionCodecConfig:
    input_dim: int
    genome: tuple[Mapping[str, Any], ...]
    name: str = "symbolic_multi_expression"
    feature_names: tuple[str, ...] = tuple()
    representation_name: str = "symbolic_basis_set"


class SymbolicMultiExpressionCodec:
    """Fixed multi-expression codec used by basis-set heads."""

    def __init__(self, config: SymbolicMultiExpressionCodecConfig) -> None:
        self.config = config
        self.genome = normalize_genome(tuple(config.genome), input_dim=int(config.input_dim))
        self.parameter_specs_by_term: tuple[tuple[ParameterSpec, ...], ...] = tuple(
            collect_parameter_specs(term["expr"]) for term in self.genome
        )
        self.trainable_specs_by_term: tuple[tuple[ParameterSpec, ...], ...] = tuple(
            tuple(spec for spec in specs if bool(spec.trainable)) for specs in self.parameter_specs_by_term
        )
        self.block_dimensions = tuple(len(specs) for specs in self.trainable_specs_by_term)
        self.total_dimension = int(sum(self.block_dimensions))

    def init_values(self) -> np.ndarray:
        chunks: list[np.ndarray] = []
        for specs in self.trainable_specs_by_term:
            chunks.append(np.asarray([spec.init for spec in specs], dtype=float).reshape(-1))
        if not chunks:
            return np.zeros(0, dtype=float)
        return np.concatenate(chunks)

    def blocks(self) -> tuple[tuple[str, int, int], ...]:
        out: list[tuple[str, int, int]] = []
        start = 0
        for i, (term, dim) in enumerate(zip(self.genome, self.block_dimensions)):
            stop = start + int(dim)
            out.append((str(term.get("name", f"term_{i}")), start, stop))
            start = stop
        return tuple(out)

    def repair_values(self, values: Sequence[float] | np.ndarray) -> np.ndarray:
        arr = np.asarray(values, dtype=float).reshape(-1)
        if arr.shape[0] != self.total_dimension:
            fixed = np.zeros(self.total_dimension, dtype=float)
            fixed[: min(arr.shape[0], self.total_dimension)] = arr[: min(arr.shape[0], self.total_dimension)]
            arr = fixed
        arr = np.nan_to_num(arr, nan=0.0, posinf=1e6, neginf=-1e6)
        for term_index, (_name, start, stop) in enumerate(self.blocks()):
            specs = self.trainable_specs_by_term[term_index]
            for local_index, spec in enumerate(specs):
                arr[start + local_index] = spec.clamp(float(arr[start + local_index]))
            if stop - start != len(specs):
                raise ValueError("internal symbolic block dimension mismatch")
        return np.asarray(arr, dtype=float).reshape(-1)

    def decode_term(
        self,
        term_index: int,
        values: Sequence[float] | np.ndarray,
        context: Mapping[str, Any] | None = None,
    ) -> SymbolicExpressionModel:
        ctx = dict(context or {})
        index = int(term_index)
        if index < 0 or index >= len(self.genome):
            raise IndexError(f"term_index {index} out of range")
        term = self.genome[index]
        specs = self.parameter_specs_by_term[index]
        trainable = tuple(spec for spec in specs if bool(spec.trainable))
        arr = np.asarray(values, dtype=float).reshape(-1)
        if arr.shape[0] != len(trainable):
            raise ValueError(f"term {index} vector has length {arr.shape[0]} but expected {len(trainable)}")
        params = parameter_values_from_vector(specs, arr)
        return SymbolicExpressionModel(
            name=str(term.get("name", f"term_{index}")),
            expression=dict(term["expr"]),
            parameter_specs=tuple(specs),
            param_values=params,
            input_dim=int(self.config.input_dim),
            feature_names=tuple(self.config.feature_names),
            metadata={
                "representation": self.config.representation_name,
                "head_block": ctx.get("head.block"),
                "symbolic.expression_index": index,
            },
        )

    def describe(self) -> dict[str, Any]:
        return {
            "codec": "symbolic_multi_expression",
            "input_dim": int(self.config.input_dim),
            "name": self.config.name,
            "feature_names": list(self.config.feature_names),
            "total_dimension": int(self.total_dimension),
            "block_dimensions": list(self.block_dimensions),
            "terms": [
                {
                    "name": str(term.get("name", f"term_{i}")),
                    "expr": dict(term["expr"]),
                    "parameter_specs": [spec.as_dict() for spec in self.parameter_specs_by_term[i]],
                }
                for i, term in enumerate(self.genome)
            ],
        }
