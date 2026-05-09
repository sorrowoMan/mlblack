from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

import numpy as np


SymbolicBenchmarkBuilder = Callable[[int, float, float, int], tuple[np.ndarray, np.ndarray, dict[str, Any]]]


@dataclass(frozen=True)
class SymbolicBenchmarkScenarioDefinition:
    """Generic scenario definition contract for symbolic benchmark problems."""

    key: str
    description: str
    feature_names: tuple[str, ...]
    target_names: tuple[str, ...]
    truth_expression: str
    strict_contract: tuple[str, ...]
    phase_equivalent_contract: tuple[str, ...]
    family_level_contract: tuple[str, ...]
    gate_feature_names: tuple[str, ...]
    periodic_feature_names: tuple[str, ...]
    enable_piecewise_basis: bool
    builder: SymbolicBenchmarkBuilder


@dataclass(frozen=True)
class SymbolicBenchmarkTruthContract:
    """Three-level truth contract used by symbolic benchmark recovery evaluation."""

    expression: str
    strict_contract: tuple[str, ...]
    phase_equivalent_contract: tuple[str, ...]
    family_level_contract: tuple[str, ...]

    def as_formula_payload(self) -> dict[str, Any]:
        return {
            "expression": str(self.expression),
            "basis_contract": tuple(self.strict_contract),
            "strict_contract": tuple(self.strict_contract),
            "phase_equivalent_contract": tuple(self.phase_equivalent_contract),
            "family_level_contract": tuple(self.family_level_contract),
        }


@dataclass(frozen=True)
class SymbolicBenchmarkLaneSpec:
    """Generic lane-level bias hint that can be consumed by outer symbolic orchestration."""

    lane_id: str
    lane_family: str
    description: str
    screening_protocol: str
    challenger_objective_protocol: str
    pool_expansion_bias_protocol: str
    lane_weight: float = 1.0
    repeat_count: int = 1
    locked_repeat_count: int = 1
    trainer_params_overrides: Mapping[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "lane_id": str(self.lane_id),
            "lane_family": str(self.lane_family),
            "description": str(self.description),
            "screening_protocol": str(self.screening_protocol),
            "challenger_objective_protocol": str(self.challenger_objective_protocol),
            "pool_expansion_bias_protocol": str(self.pool_expansion_bias_protocol),
            "lane_weight": float(self.lane_weight),
            "repeat_count": int(self.repeat_count),
            "locked_repeat_count": int(self.locked_repeat_count),
            "trainer_params_overrides": dict(self.trainer_params_overrides or {}),
        }


def truth_contract_for_scenario(
    definition: SymbolicBenchmarkScenarioDefinition,
) -> SymbolicBenchmarkTruthContract:
    return SymbolicBenchmarkTruthContract(
        expression=str(definition.truth_expression),
        strict_contract=tuple(str(value) for value in definition.strict_contract),
        phase_equivalent_contract=tuple(str(value) for value in definition.phase_equivalent_contract),
        family_level_contract=tuple(str(value) for value in definition.family_level_contract),
    )


__all__ = [
    "SymbolicBenchmarkBuilder",
    "SymbolicBenchmarkLaneSpec",
    "SymbolicBenchmarkScenarioDefinition",
    "SymbolicBenchmarkTruthContract",
    "truth_contract_for_scenario",
]
