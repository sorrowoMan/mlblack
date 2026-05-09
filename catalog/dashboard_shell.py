from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

DEFAULT_PROFILE = "framework-core"
DEFAULT_SCOPE = "framework"
DEFAULT_KIND = "all"
DEFAULT_COLUMN_MODE = "standard"
DEFAULT_PAGE_SIZE = 50
DEFAULT_RESULTS_COLLAPSE = "expanded"
QUERY_TRACE_PATH_ENV = "CATALOG_UI_QUERY_TRACE_PATH"
_MEMOIZED_LOADER_REGISTRY: dict[tuple[str, str, int, bool], dict[str, Any]] = {}


def clone_value(value: Any) -> Any:
    return copy.deepcopy(value)


def _query_trace_path() -> Path | None:
    raw = str(os.getenv(QUERY_TRACE_PATH_ENV, "") or "").strip()
    if not raw:
        return None
    return Path(raw)


def _append_query_trace_event(
    *,
    loader_name: str,
    cache_status: str,
    duration_ms: float,
    cache_hits: int,
    cache_misses: int,
    cache_size: int,
) -> None:
    trace_path = _query_trace_path()
    if trace_path is None:
        return
    try:
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        with trace_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "ts": time.time(),
                        "pid": os.getpid(),
                        "loader": str(loader_name),
                        "cache_status": str(cache_status),
                        "duration_ms": round(float(duration_ms), 3),
                        "cache_hits": int(cache_hits),
                        "cache_misses": int(cache_misses),
                        "cache_size": int(cache_size),
                    },
                    ensure_ascii=False,
                )
            )
            handle.write("\n")
    except Exception:
        return


def build_streamlit_command(
    *,
    script_path: str | Path,
    profile: str = DEFAULT_PROFILE,
    scope: str = DEFAULT_SCOPE,
    kind: str = DEFAULT_KIND,
    query: str = "",
    field: str | None = None,
    project_path: str | None = None,
    include_global: bool = False,
    db_path: str | None = None,
    source_mode: str | None = None,
    column_mode: str = DEFAULT_COLUMN_MODE,
    page_size: int = DEFAULT_PAGE_SIZE,
    results_collapse: str = DEFAULT_RESULTS_COLLAPSE,
    host: str | None = None,
    port: int | None = None,
    headless: bool = False,
) -> list[str]:
    command = [sys.executable, "-m", "streamlit", "run", str(Path(script_path).resolve())]
    if host:
        command.extend(["--server.address", str(host)])
    if port is not None:
        command.extend(["--server.port", str(int(port))])
    if headless:
        command.extend(["--server.headless", "true"])
    command.extend(
        [
            "--",
            "--profile",
            str(profile),
            "--scope",
            str(scope),
            "--kind",
            str(kind),
            "--column-mode",
            str(column_mode),
            "--page-size",
            str(int(page_size)),
            "--results-collapse",
            str(results_collapse),
        ]
    )
    if query:
        command.extend(["--query", str(query)])
    if field:
        command.extend(["--field", str(field)])
    if project_path:
        command.extend(["--project-path", str(project_path)])
    if include_global:
        command.append("--include-global")
    if db_path:
        command.extend(["--db-path", str(db_path)])
    if source_mode:
        command.extend(["--source-mode", str(source_mode)])
    return command


def launch_catalog_dashboard(
    *,
    script_path: str | Path,
    profile: str = DEFAULT_PROFILE,
    scope: str = DEFAULT_SCOPE,
    kind: str = DEFAULT_KIND,
    query: str = "",
    field: str | None = None,
    project_path: str | None = None,
    include_global: bool = False,
    db_path: str | None = None,
    source_mode: str | None = None,
    column_mode: str = DEFAULT_COLUMN_MODE,
    page_size: int = DEFAULT_PAGE_SIZE,
    results_collapse: str = DEFAULT_RESULTS_COLLAPSE,
    host: str | None = None,
    port: int | None = None,
    headless: bool = False,
) -> int:
    return int(
        subprocess.call(
            build_streamlit_command(
                script_path=script_path,
                profile=profile,
                scope=scope,
                kind=kind,
                query=query,
                field=field,
                project_path=project_path,
                include_global=include_global,
                db_path=db_path,
                source_mode=source_mode,
                column_mode=column_mode,
                page_size=page_size,
                results_collapse=results_collapse,
                host=host,
                port=port,
                headless=headless,
            )
        )
    )


def _normalize_filter_values(value: object) -> tuple[str, ...]:
    if value in (None, ""):
        return ()
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",")]
        return tuple(part for part in parts if part)
    if isinstance(value, Mapping):
        return tuple(str(key).strip() for key in value.keys() if str(key).strip())
    if isinstance(value, (list, tuple, set, frozenset)):
        out: list[str] = []
        for item in value:
            out.extend(_normalize_filter_values(item))
        return tuple(out)
    text = str(value).strip()
    return (text,) if text else ()


def freeze_filters(field_filters: Mapping[str, object] | Sequence[tuple[str, object]] | None) -> tuple[tuple[str, tuple[str, ...]], ...]:
    if not field_filters:
        return ()
    items = field_filters.items() if isinstance(field_filters, Mapping) else field_filters
    frozen: list[tuple[str, tuple[str, ...]]] = []
    for name, raw_value in items:
        key = str(name or "").strip()
        if not key:
            continue
        values = _normalize_filter_values(raw_value)
        if not values:
            continue
        frozen.append((key, values))
    frozen.sort(key=lambda item: item[0])
    return tuple(frozen)


def thaw_filters(filters_key: tuple[tuple[str, tuple[str, ...]], ...]) -> dict[str, object]:
    thawed: dict[str, object] = {}
    for name, values in filters_key:
        key = str(name or "").strip()
        clean_values = tuple(str(value).strip() for value in values if str(value).strip())
        if not key or not clean_values:
            continue
        thawed[key] = clean_values[0] if len(clean_values) == 1 else clean_values
    return thawed


def memoize_loader(loader: Callable[..., Any], *, maxsize: int = 128, clone_result: bool = True) -> Callable[..., Any]:
    registry_key = (
        str(getattr(loader, "__module__", "")),
        str(getattr(loader, "__qualname__", getattr(loader, "__name__", "loader"))),
        int(maxsize),
        bool(clone_result),
    )
    entry = _MEMOIZED_LOADER_REGISTRY.get(registry_key)
    if entry is None:
        holder: dict[str, Any] = {"loader": loader}

        @lru_cache(maxsize=maxsize)
        def _cached(*args):
            result = holder["loader"](*args)
            return clone_value(result) if clone_result else result

        def _wrapped(*args):
            started = time.perf_counter()
            before = _cached.cache_info()
            result = _cached(*args)
            after = _cached.cache_info()
            duration_ms = (time.perf_counter() - started) * 1000.0
            if after.hits > before.hits:
                cache_status = "hit"
            elif after.misses > before.misses:
                cache_status = "miss"
            else:
                cache_status = "unknown"
            _append_query_trace_event(
                loader_name=str(getattr(holder["loader"], "__name__", "loader")),
                cache_status=cache_status,
                duration_ms=duration_ms,
                cache_hits=after.hits,
                cache_misses=after.misses,
                cache_size=after.currsize,
            )
            return clone_value(result) if clone_result else result

        setattr(_wrapped, "cache_clear", getattr(_cached, "cache_clear"))
        setattr(_wrapped, "cache_info", getattr(_cached, "cache_info"))
        entry = {"holder": holder, "wrapped": _wrapped}
        _MEMOIZED_LOADER_REGISTRY[registry_key] = entry
    else:
        entry["holder"]["loader"] = loader
    return entry["wrapped"]
