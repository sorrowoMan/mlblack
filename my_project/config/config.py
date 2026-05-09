from __future__ import annotations

from my_project.config.schema import ProjectConfig, default_project_config


def get_project_config() -> ProjectConfig:
    return default_project_config()
