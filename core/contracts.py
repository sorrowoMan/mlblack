"""Forwarding module for component contracts.

This module re-exports from blackbase for seamless migration.
All implementation now lives in blackbase.contracts.
"""

from __future__ import annotations

from blackbase.contracts import (
    ComponentContract,
    ContractMixin,
    combine_contracts,
)

__all__ = [
    "ComponentContract",
    "ContractMixin",
    "combine_contracts",
]
