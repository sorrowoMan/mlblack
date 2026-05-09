from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from training import describe_inner_runtime_event_table

from .lifecycle_events import describe_lifecycle_event_table


@dataclass(frozen=True)
class ControlPlaneContract:
    """Unified contract surface for outer lifecycle and inner runtime events."""

    lifecycle_events: tuple[Mapping[str, Any], ...]
    inner_runtime_events: tuple[Mapping[str, Any], ...]

    def to_mapping(self) -> dict[str, Any]:
        return {
            "lifecycle_events": [dict(row) for row in self.lifecycle_events],
            "inner_runtime_events": [dict(row) for row in self.inner_runtime_events],
        }


def describe_control_plane_contract(
    *,
    lifecycle_events: Sequence[Mapping[str, Any]] | None = None,
    inner_runtime_events: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    lifecycle_rows = (
        tuple(dict(row) for row in lifecycle_events)
        if lifecycle_events is not None
        else tuple(dict(row) for row in describe_lifecycle_event_table())
    )
    inner_rows = (
        tuple(dict(row) for row in inner_runtime_events)
        if inner_runtime_events is not None
        else tuple(dict(row) for row in describe_inner_runtime_event_table())
    )
    return ControlPlaneContract(
        lifecycle_events=lifecycle_rows,
        inner_runtime_events=inner_rows,
    ).to_mapping()


__all__ = ["ControlPlaneContract", "describe_control_plane_contract"]
