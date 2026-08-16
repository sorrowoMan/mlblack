"""
MLBlack core protocol types.

The shared payload objects live in blackbase so nsgablack/mlblack Cases can
exchange candidate state and feedback without depending on each other.
"""

from __future__ import annotations

from blackbase.context import (
    CATEGORY_CACHE,
    CATEGORY_DERIVED,
    CATEGORY_EVENT,
    CATEGORY_INPUT,
    CATEGORY_OUTPUT,
    CATEGORY_RUNTIME,
    ContextField,
    ContextSchema,
)
from blackbase.resources import DataRef, ResourceRequirement
from blackbase.types import Feedback, PopulationSnapshot, TrainerResult, UnknownState


__all__ = [
    "Feedback",
    "UnknownState",
    "PopulationSnapshot",
    "TrainerResult",
    "ContextField",
    "ContextSchema",
    "CATEGORY_CACHE",
    "CATEGORY_DERIVED",
    "CATEGORY_EVENT",
    "CATEGORY_INPUT",
    "CATEGORY_OUTPUT",
    "CATEGORY_RUNTIME",
    "DataRef",
    "ResourceRequirement",
]
