from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
LEGACY_DATA_DIR = (
    Path.home()
    / "Desktop"
    / "work"
    / "final_pipeline_package_20260402"
    / "04_interval_dataset"
)


def _env_path(env_key: str) -> str | None:
    raw = os.environ.get(env_key, "")
    value = str(raw).strip()
    if not value:
        return None
    return str(Path(value).expanduser())


def _first_existing(candidates: list[Path]) -> Path | None:
    for item in candidates:
        if item.exists():
            return item
    return None


def _resolve_variant(env_key: str, filename: str) -> str:
    env = _env_path(env_key)
    if env is not None:
        return env
    found = _first_existing([DATA_DIR / filename, LEGACY_DATA_DIR / filename])
    if found is not None:
        return str(found)
    return str(DATA_DIR / filename)


def default_work_ci_csv() -> str:
    return _resolve_variant("MLBLACK_WORK_CI_CSV", "ci_interval_opt_table.csv")


def default_work_ci_csv_no_flow_speed_occ() -> str:
    return _resolve_variant(
        "MLBLACK_WORK_CI_CSV_NO_FLOW_SPEED_OCC",
        "ci_interval_opt_table_no_flow_speed_occ.csv",
    )


def default_work_ci_csv_no_flow_speed_occ_lag() -> str:
    return _resolve_variant(
        "MLBLACK_WORK_CI_CSV_NO_FLOW_SPEED_OCC_LAG",
        "ci_interval_opt_table_no_flow_speed_occ_lag.csv",
    )


def apply_env_defaults() -> None:
    os.environ.setdefault("MLBLACK_WORK_CI_CSV", default_work_ci_csv())
    os.environ.setdefault("MLBLACK_WORK_CI_CSV_NO_FLOW_SPEED_OCC", default_work_ci_csv_no_flow_speed_occ())
    os.environ.setdefault("MLBLACK_WORK_CI_CSV_NO_FLOW_SPEED_OCC_LAG", default_work_ci_csv_no_flow_speed_occ_lag())

