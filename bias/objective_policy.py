from __future__ import annotations

from typing import Any, Mapping, Sequence

from core.symbolic.feature_space.objective_policy import interval_objective_sort_key

from .config import ObjectivePolicyConfig


def build_interval_row_objective_key(
    row: Mapping[str, Any],
    *,
    cfg: ObjectivePolicyConfig,
) -> tuple[float, float, float, float]:
    return interval_objective_sort_key(
        coverage_error_value=float(row.get("obj_coverage_error", float("inf"))),
        pinaw=float(row.get("obj_pinaw", float("inf"))),
        interval_score=float(row.get("obj_interval_score", float("inf"))),
        coverage_error_threshold=float(max(0.0, cfg.coverage_error_threshold)),
    )


def sort_interval_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    cfg: ObjectivePolicyConfig,
) -> list[Mapping[str, Any]]:
    return sorted(rows, key=lambda row: build_interval_row_objective_key(row, cfg=cfg))


__all__ = [
    "ObjectivePolicyConfig",
    "build_interval_row_objective_key",
    "sort_interval_rows",
]
