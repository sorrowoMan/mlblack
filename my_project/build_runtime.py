from __future__ import annotations

from dataclasses import asdict
from typing import Any

from my_project.config.config import get_project_config
from my_project.features.config import build_feature_bundle
from my_project.model.config import train_model_bundle
from my_project.problem.config import build_problem_context
from my_project.reporting.config import write_report


def build_runtime_components() -> dict[str, Any]:
    cfg = get_project_config()
    return {
        "config": cfg,
        "config_dict": asdict(cfg),
        "problem_builder": build_problem_context,
        "feature_builder": build_feature_bundle,
        "model_trainer": train_model_bundle,
        "report_writer": write_report,
    }
