from __future__ import annotations

from typing import Any, Mapping


def extract_orchestrator_hints(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    return dict(dict(metadata or {}).get("orchestrator_hints", {}) or {})


def extract_trainer_params_overrides(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    hints = extract_orchestrator_hints(metadata)
    return dict(hints.get("trainer_params_overrides", {}) or {})


def extract_lane_specs(metadata: Mapping[str, Any] | None) -> tuple[dict[str, Any], ...]:
    hints = extract_orchestrator_hints(metadata)
    return tuple(dict(row) for row in tuple(hints.get("lane_specs", ()) or ()) if isinstance(row, Mapping))


def extract_core_selection_policy(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    hints = extract_orchestrator_hints(metadata)
    return dict(hints.get("core_selection", {}) or {})


def extract_search_hints(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    return dict(dict(metadata or {}).get("search_hints", {}) or {})


__all__ = [
    "extract_core_selection_policy",
    "extract_lane_specs",
    "extract_orchestrator_hints",
    "extract_search_hints",
    "extract_trainer_params_overrides",
]
