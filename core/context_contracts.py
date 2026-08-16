"""
Forwarding module for context contracts.

This module re-exports from blackbase for seamless migration.
"""

from blackbase.context import (
    ContextContract,
    collect_solver_contracts,
    detect_context_conflicts,
    get_component_contract,
    validate_context_contracts,
)

__all__ = [
    "ContextContract",
    "collect_solver_contracts",
    "detect_context_conflicts",
    "get_component_contract",
    "validate_context_contracts",
]