"""Case-level component registry aggregation."""

from dataclasses import dataclass


@dataclass(frozen=True)
class CaseConfig:
    problem_key: str = "default"
    pipeline_key: str = "default"
    adapter_key: str = "default"
    resource_request: dict | None = None


def get_case_config() -> CaseConfig:
    return CaseConfig()
