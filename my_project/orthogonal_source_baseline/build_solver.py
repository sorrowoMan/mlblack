from __future__ import annotations

from dataclasses import asdict
from typing import Any

from my_project.orthogonal_source_baseline.config import OrthogonalSourceBaselineConfig
from my_project.orthogonal_source_baseline.orchestration import run_suite


def build_orthogonal_source_baseline_components(
    cfg: OrthogonalSourceBaselineConfig | None = None,
) -> dict[str, Any]:
    resolved = cfg or OrthogonalSourceBaselineConfig()
    return {
        "config": resolved,
        "config_dict": asdict(resolved),
        "runner": run_suite,
        "representation_component": "core.orthogonal_source.OrthogonalSourceLayer",
        "downstream_models": ("ridge", "random_forest", "gradient_boosting"),
    }


__all__ = ["OrthogonalSourceBaselineConfig", "build_orthogonal_source_baseline_components"]
