from __future__ import annotations

from typing import Any

from core.symbolic.benchmark.contracts import truth_contract_for_scenario

from my_project.known_relation_symbolic.problem.registry import get_known_relation_benchmark
from my_project.known_relation_symbolic.problem.specs import KnownRelationTruthContract


def truth_contract_for_key(key: str) -> KnownRelationTruthContract:
    return truth_contract_for_scenario(get_known_relation_benchmark(key))


def truth_contract_payload_for_key(key: str) -> dict[str, Any]:
    return truth_contract_for_key(key).as_formula_payload()


__all__ = ["truth_contract_for_key", "truth_contract_payload_for_key"]
