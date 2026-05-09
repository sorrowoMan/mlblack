from __future__ import annotations

from typing import Any


def build_flow_plugins(*_args: Any, **_kwargs: Any) -> list[Any]:
    """Reserved extension point for L3 flow plugins."""
    return []


def build_ops_plugins(*_args: Any, **_kwargs: Any) -> list[Any]:
    """Reserved extension point for L1/L2 ops plugins."""
    return []

