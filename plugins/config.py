from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .report_writer import write_summary_report


@dataclass(frozen=True)
class ReportingConfig:
    out_root: Path
    interval_alpha: float


def emit_report(
    *,
    report: Mapping[str, Any],
    cfg: ReportingConfig,
    sym_rmse: float,
    xgb_rmse: float,
    sym_interval: Mapping[str, Any],
    xgb_interval: Mapping[str, Any],
) -> Path:
    return write_summary_report(
        report=report,
        out_root=cfg.out_root,
        sym_rmse=float(sym_rmse),
        xgb_rmse=float(xgb_rmse),
        sym_interval=sym_interval,
        xgb_interval=xgb_interval,
        interval_alpha=float(cfg.interval_alpha),
    )


__all__ = ["ReportingConfig", "emit_report"]
