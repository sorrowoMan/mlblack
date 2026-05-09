from __future__ import annotations

import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

# Backward-compatibility fallback for historical local setup.
_LEGACY_WORK_CI_CSV = (
    r"C:\Users\hp\Desktop\work\final_pipeline_package_20260402\04_interval_dataset\ci_interval_opt_table.csv"
)
_LEGACY_WORK_CI_CSV_NO_FLOW_SPEED_OCC = (
    r"C:\Users\hp\Desktop\work\final_pipeline_package_20260402\04_interval_dataset\ci_interval_opt_table_no_flow_speed_occ.csv"
)
_LEGACY_WORK_CI_CSV_NO_FLOW_SPEED_OCC_LAG = (
    r"C:\Users\hp\Desktop\work\final_pipeline_package_20260402\04_interval_dataset\ci_interval_opt_table_no_flow_speed_occ_lag.csv"
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


def default_work_ci_csv() -> str:
    """Resolve default work-CI CSV path with env override support.

    Priority:
    1) MLBLACK_WORK_CI_CSV
    2) common repo-relative candidate paths
    3) legacy local absolute path (for backward compatibility only)
    """

    env = _env_path("MLBLACK_WORK_CI_CSV")
    if env is not None:
        return env

    candidate = _first_existing(
        [
            ROOT / "examples" / "data" / "ci_interval_opt_table.csv",
            ROOT / "data" / "ci_interval_opt_table.csv",
            ROOT.parent
            / "work"
            / "final_pipeline_package_20260402"
            / "04_interval_dataset"
            / "ci_interval_opt_table.csv",
        ]
    )
    if candidate is not None:
        return str(candidate)

    return _LEGACY_WORK_CI_CSV


def _default_work_ci_variant_csv(*, env_key: str, filename: str, legacy_fallback: str) -> str:
    env = _env_path(env_key)
    if env is not None:
        return env

    base = Path(default_work_ci_csv())
    candidates = [
        ROOT / "examples" / "data" / filename,
        ROOT / "data" / filename,
        ROOT.parent / "work" / "final_pipeline_package_20260402" / "04_interval_dataset" / filename,
        base.with_name(filename),
    ]
    candidate = _first_existing(candidates)
    if candidate is not None:
        return str(candidate)
    return str(legacy_fallback)


def default_work_ci_csv_no_flow_speed_occ() -> str:
    """Resolve no-flow-speed-occ CSV path with env override support."""

    return _default_work_ci_variant_csv(
        env_key="MLBLACK_WORK_CI_CSV_NO_FLOW_SPEED_OCC",
        filename="ci_interval_opt_table_no_flow_speed_occ.csv",
        legacy_fallback=_LEGACY_WORK_CI_CSV_NO_FLOW_SPEED_OCC,
    )


def default_work_ci_csv_no_flow_speed_occ_lag() -> str:
    """Resolve no-flow-speed-occ-lag CSV path with env override support."""

    return _default_work_ci_variant_csv(
        env_key="MLBLACK_WORK_CI_CSV_NO_FLOW_SPEED_OCC_LAG",
        filename="ci_interval_opt_table_no_flow_speed_occ_lag.csv",
        legacy_fallback=_LEGACY_WORK_CI_CSV_NO_FLOW_SPEED_OCC_LAG,
    )


def default_nsgablack_root() -> str:
    """Resolve nsgablack repo root (contains `nsgablack/__init__.py`)."""

    env = _env_path("NSGABLACK_ROOT")
    if env is not None:
        return env

    candidate = ROOT.parent / "nsgablack"
    if (candidate / "nsgablack" / "__init__.py").exists():
        return str(candidate)

    # Legacy fallback.
    return r"C:\Users\hp\Desktop\nsgablack"


def default_reports_dir() -> str:
    """Directory for generated report files."""

    env = _env_path("MLBLACK_REPORTS_DIR")
    if env is not None:
        return env
    return str(ROOT / "examples" / "out" / "reports")


def default_outputs_dir() -> str:
    """Directory for generated output artifacts."""

    env = _env_path("MLBLACK_OUTPUTS_DIR")
    if env is not None:
        return env
    return str(ROOT / "examples" / "out")


def apply_env_defaults() -> None:
    """Set practical default env vars for local examples if missing."""
    os.environ.setdefault("MLBLACK_WORK_CI_CSV", default_work_ci_csv())
    os.environ.setdefault("MLBLACK_WORK_CI_CSV_NO_FLOW_SPEED_OCC", default_work_ci_csv_no_flow_speed_occ())
    os.environ.setdefault("MLBLACK_WORK_CI_CSV_NO_FLOW_SPEED_OCC_LAG", default_work_ci_csv_no_flow_speed_occ_lag())
    os.environ.setdefault("MLBLACK_REPORTS_DIR", default_reports_dir())
    os.environ.setdefault("MLBLACK_OUTPUTS_DIR", default_outputs_dir())
    os.environ.setdefault("NSGABLACK_ROOT", default_nsgablack_root())
