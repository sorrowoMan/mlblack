from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class MlblackRuntimeConfig:
    scenario_name: str = "nowcasting_work_ci"
    output_dir_name: str = "_scenario_runs"
    output_prefix: str = "nowcasting_symbolic_subset_bridge_work_ci_seed"
    graph_cache_namespace: str = "work_ci_nowcasting_subset_bridge"


def build_runs_root(
    project_root: Path,
    *,
    cfg: MlblackRuntimeConfig | None = None,
) -> Path:
    conf = cfg if cfg is not None else MlblackRuntimeConfig()
    override = str(os.environ.get("MLBLACK_SCENARIO_RUNS_ROOT", "")).strip()
    if override:
        return Path(override).expanduser().resolve()
    return Path(project_root) / str(conf.output_dir_name) / str(conf.scenario_name)


def build_output_root(
    project_root: Path,
    *,
    seed: int,
    stamp: str | None = None,
    cfg: MlblackRuntimeConfig | None = None,
) -> Path:
    conf = cfg if cfg is not None else MlblackRuntimeConfig()
    ts = str(stamp) if stamp is not None else datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out_root = build_runs_root(project_root, cfg=conf) / f"{conf.output_prefix}{int(seed)}_{ts}"
    return out_root


__all__ = ["MlblackRuntimeConfig", "build_output_root", "build_runs_root"]
