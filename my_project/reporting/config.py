from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from my_project.config.schema import ReportingConfig
from my_project.reporting.example_reporter import write_json_report


def write_report(payload: Mapping[str, Any], cfg: ReportingConfig, run_id: str) -> Path:
    return write_json_report(payload, out_dir=str(cfg.output_dir), run_id=run_id)
