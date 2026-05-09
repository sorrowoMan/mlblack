from __future__ import annotations

from core.symbolic.benchmark.contracts import (
    SymbolicBenchmarkBuilder,
    SymbolicBenchmarkLaneSpec,
    SymbolicBenchmarkScenarioDefinition,
    SymbolicBenchmarkTruthContract,
    truth_contract_for_scenario,
)


KnownRelationBuilder = SymbolicBenchmarkBuilder
KnownRelationBenchmarkDefinition = SymbolicBenchmarkScenarioDefinition
KnownRelationLaneSpec = SymbolicBenchmarkLaneSpec
KnownRelationTruthContract = SymbolicBenchmarkTruthContract


def truth_contract_for_definition(
    definition: KnownRelationBenchmarkDefinition,
) -> KnownRelationTruthContract:
    return truth_contract_for_scenario(definition)


__all__ = [
    "KnownRelationBenchmarkDefinition",
    "KnownRelationBuilder",
    "KnownRelationLaneSpec",
    "KnownRelationTruthContract",
    "truth_contract_for_definition",
]
