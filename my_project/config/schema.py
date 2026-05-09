from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProblemConfig:
    data_csv: str = ""
    target_col: str = "target"


@dataclass(frozen=True)
class FeatureConfig:
    add_bias: bool = True


@dataclass(frozen=True)
class ModelConfig:
    baseline: str = "mean"


@dataclass(frozen=True)
class ReportingConfig:
    output_dir: str = "out"


@dataclass(frozen=True)
class RuntimeConfig:
    random_seed: int = 42


@dataclass(frozen=True)
class ProjectConfig:
    problem: ProblemConfig
    features: FeatureConfig
    model: ModelConfig
    reporting: ReportingConfig
    runtime: RuntimeConfig


def default_project_config() -> ProjectConfig:
    project_root = Path(__file__).resolve().parents[1]
    return ProjectConfig(
        problem=ProblemConfig(),
        features=FeatureConfig(),
        model=ModelConfig(),
        reporting=ReportingConfig(output_dir=str(project_root / "out")),
        runtime=RuntimeConfig(),
    )
