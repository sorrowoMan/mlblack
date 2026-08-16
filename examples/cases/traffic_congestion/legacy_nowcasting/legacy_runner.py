from __future__ import annotations

import importlib
import json
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LEGACY_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
OUT_DIR = PROJECT_ROOT / "out"


def _prepend(path: Path) -> None:
    text = str(path.resolve())
    if text not in sys.path:
        sys.path.insert(0, text)


def ensure_legacy_importable() -> None:
    _prepend(LEGACY_ROOT)
    for cur in (LEGACY_ROOT, *LEGACY_ROOT.parents):
        if (cur / "mlblack.py").is_file() and (cur / "pyproject.toml").is_file():
            _prepend(cur.parent)
            break
    for candidate in (
        PROJECT_ROOT.parents[3] / "nsgablack" if len(PROJECT_ROOT.parents) > 3 else None,
        Path.home() / "Desktop" / "nsgablack",
    ):
        if candidate is not None and (candidate / "__init__.py").is_file() and (candidate / "core").is_dir():
            _prepend(candidate.parent)
            break


def _resource_context_from_env() -> dict[str, Any]:
    raw = os.environ.get("MLBLACK_RESOURCE_CONTEXT_JSON", "")
    if not raw.strip():
        return {}
    try:
        value = json.loads(raw)
    except Exception:
        return {"raw": raw}
    return dict(value) if isinstance(value, dict) else {"value": value}


def _granted_threads(resource_context: dict[str, Any]) -> int:
    for key in ("threads", "cpu_threads", "workers"):
        value = resource_context.get(key)
        try:
            if value is not None:
                return max(1, int(value))
        except Exception:
            continue
    grant = resource_context.get("grant")
    if isinstance(grant, dict):
        return _granted_threads(grant)
    return 1


def _has_arg(argv: Sequence[str], name: str) -> bool:
    return any(str(item) == str(name) or str(item).startswith(f"{name}=") for item in argv)


def _append_missing(argv: list[str], *pairs: tuple[str, str]) -> list[str]:
    out = list(argv)
    for key, value in pairs:
        if not _has_arg(out, key):
            out.extend([key, str(value)])
    return out


def _default_csv(case_kind: str) -> Path:
    if case_kind == "interval":
        candidate = DATA_DIR / "ci_interval_opt_table_no_flow_speed_occ_lag.csv"
        if candidate.exists():
            return candidate
    candidate = DATA_DIR / "ci_interval_opt_table.csv"
    if candidate.exists():
        return candidate
    candidate = DATA_DIR / "ci_interval_opt_table_no_flow_speed_occ.csv"
    if candidate.exists():
        return candidate
    return DATA_DIR / "ci_interval_opt_table_no_flow_speed_occ_lag.csv"


def _module_name(case_kind: str) -> str:
    if case_kind == "interval":
        return "nowcasting_work_ci.run_nowcasting_symbolic_subset_bridge_work_ci_native_interval"
    if case_kind == "mechanism":
        return "nowcasting_work_ci.run_nowcasting_symbolic_subset_bridge_work_ci"
    raise ValueError(f"unknown legacy case kind: {case_kind}")


@contextmanager
def _patched_argv(argv: Sequence[str]):
    old = list(sys.argv)
    sys.argv = [old[0], *[str(item) for item in argv]]
    try:
        yield
    finally:
        sys.argv = old


@dataclass(frozen=True)
class LegacyNowcastingCheck:
    case: str
    module: str
    csv_path: str
    resource_context: dict[str, Any] = field(default_factory=dict)
    effective_threads: int = 1
    default_args: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "case": self.case,
            "module": self.module,
            "csv_path": self.csv_path,
            "resource_context": dict(self.resource_context),
            "effective_threads": int(self.effective_threads),
            "default_args": list(self.default_args),
        }


def build_default_args(case_kind: str, argv: Sequence[str]) -> list[str]:
    resource_context = _resource_context_from_env()
    threads = _granted_threads(resource_context)
    default_csv = _default_csv(case_kind)
    cache_dir = OUT_DIR / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    args = [str(item) for item in argv]
    args = _append_missing(
        args,
        ("--csv-path", str(default_csv)),
        ("--strict4-branch-parallel-workers", str(threads)),
        ("--graph-cache-db-path", str(cache_dir / f"{case_kind}_expression_graph_cache.sqlite3")),
    )

    if case_kind == "interval":
        args = _append_missing(
            args,
            ("--interval-method", "native_quantile_cqr"),
            ("--drop-same-day-flow-speed-occ", "1"),
        )
    return args


def check_payload(case_kind: str, argv: Sequence[str] = ()) -> LegacyNowcastingCheck:
    ensure_legacy_importable()
    resource_context = _resource_context_from_env()
    effective_args = build_default_args(case_kind, argv)
    module = _module_name(case_kind)
    importlib.import_module(module)
    return LegacyNowcastingCheck(
        case=f"symbolic_{case_kind}_outer",
        module=module,
        csv_path=str(_default_csv(case_kind)),
        resource_context=resource_context,
        effective_threads=_granted_threads(resource_context),
        default_args=tuple(effective_args),
    )


def run_legacy_case(case_kind: str, argv: Sequence[str] = ()) -> int:
    ensure_legacy_importable()
    args = build_default_args(case_kind, argv)
    module = importlib.import_module(_module_name(case_kind))
    main = getattr(module, "main")
    with _patched_argv(args):
        main()
    return 0


__all__ = [
    "DATA_DIR",
    "OUT_DIR",
    "PROJECT_ROOT",
    "LegacyNowcastingCheck",
    "build_default_args",
    "check_payload",
    "ensure_legacy_importable",
    "run_legacy_case",
]

