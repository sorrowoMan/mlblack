from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def _jsonable(v: Any) -> Any:
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, Mapping):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, set)):
        return [_jsonable(x) for x in v]
    return str(v)


def write_summary_report(
    *,
    report: Mapping[str, Any],
    out_root: str | Path,
    sym_rmse: float,
    xgb_rmse: float,
    sym_interval: Mapping[str, Any],
    xgb_interval: Mapping[str, Any],
    interval_alpha: float,
) -> Path:
    report_path = Path(out_root) / "summary.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(_jsonable(dict(report)), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("NSGABLACK_SYMBOLIC_SUBSET_BRIDGE_DONE")
    print(f"summary={report_path}")
    print(
        "rmse: "
        f"symbolic_subset={float(sym_rmse):.6f}, "
        f"xgboost={float(xgb_rmse):.6f}, "
        f"delta={float(sym_rmse - xgb_rmse):.6f}"
    )
    print(
        "interval: "
        f"alpha={float(interval_alpha):.3f} | "
        f"symbolic(PICP={float(sym_interval['picp']):.4f}, PINAW={float(sym_interval['pinaw']):.4f}, IS={float(sym_interval['interval_score']):.4f}) | "
        f"xgb(PICP={float(xgb_interval['picp']):.4f}, PINAW={float(xgb_interval['pinaw']):.4f}, IS={float(xgb_interval['interval_score']):.4f})"
    )
    return report_path


__all__ = ["write_summary_report"]
