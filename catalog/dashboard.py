from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from html import escape
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import parse_qsl, urlencode

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from catalog import (
    catalog_facets,
    catalog_neighbors,
    dashboard_page as _page,
    catalog_schema,
    catalog_source_info,
    catalog_summary,
    dashboard_shell as _shell,
    dashboard_shared as _shared,
    list_entries,
    search_entries,
    show_entry,
)

_KIND_ORDER: tuple[str, ...] = ("family", "preset", "head", "component", "provider", "plugin")
_NAV_ACTION_LOCATE_SELECTED = "locate_selected"
_PENDING_LOCATE_KEY = "catalog_ui_pending_locate_selected"
_PENDING_SCROLL_TARGET_KEY = "catalog_ui_pending_scroll_target"
_KIND_LABELS: dict[str, str] = {
    "family": "家族 Family",
    "preset": "预设 Preset",
    "head": "输出 Head",
    "component": "组件 Component",
    "provider": "Provider",
    "plugin": "Plugin",
}
_FIELD_LABELS: dict[str, str] = {
    "family": "家族",
    "head": "输出头",
    "heads": "输出头集合",
    "runtime_backend": "运行后端",
    "runtime_backends": "运行后端集合",
    "parameter_backends": "参数后端",
    "status": "状态",
    "preset_kind": "预设类型",
    "supports_resume": "支持恢复",
    "supports_warm_start": "支持 warm start",
    "supports_incremental": "支持 incremental",
    "outputs": "输出语义",
    "objective_families": "目标家族",
    "component_surface": "组件表面",
    "component_kind": "组件类型",
    "binding_level": "绑定层级",
    "applicable_families": "适用家族",
    "applicable_presets": "适用预设",
    "provider_surface": "Provider 表面",
    "plane": "平面",
    "supports_batch": "支持批量",
    "supports_individual": "支持单点评估",
    "plugin_surface": "Plugin 表面",
    "lifecycle_plane": "生命周期平面",
    "is_algorithmic": "算法相关",
    "enabled_by_default": "默认启用",
    "artifact_stability_fields": "稳定性字段",
    "artifact_schema_fields": "Artifact 字段",
    "search_mechanism_keys": "搜索机制",
    "search_mechanism_kinds": "搜索机制类型",
    "search_checkpointable_mechanisms": "可 checkpoint 搜索机制",
    "search_replayable_mechanisms": "可 replay 搜索机制",
    "search_family_signature_mechanisms": "影响 family signature 的搜索机制",
    "legacy_trainer_entry": "兼容 trainer 入口",
    "title_zh": "中文标题",
    "summary_zh": "中文摘要",
    "use_when_zh": "适用场景",
}
_DEFAULT_FACET_FIELDS: dict[str, tuple[str, ...]] = {
    "preset": (
        "family",
        "head",
        "runtime_backend",
        "status",
        "preset_kind",
        "supports_resume",
        "supports_warm_start",
        "artifact_stability_fields",
        "search_mechanism_keys",
        "search_family_signature_mechanisms",
    ),
    "family": (
        "family",
        "heads",
        "runtime_backends",
        "parameter_backends",
        "supports_resume",
        "artifact_stability_fields",
        "search_mechanism_keys",
        "search_family_signature_mechanisms",
    ),
    "head": (
        "head",
        "families",
        "objective_families",
        "outputs",
        "artifact_stability_fields",
        "search_mechanism_keys",
        "search_family_signature_mechanisms",
    ),
    "component": ("component_surface", "component_kind", "binding_level", "status"),
    "provider": ("provider_surface", "plane", "supports_batch", "supports_individual", "status"),
    "plugin": ("plugin_surface", "lifecycle_plane", "is_algorithmic", "enabled_by_default", "status"),
}
_NO_SELECTION = _shared.NO_SELECTION
_NAV_STACK_KEY = _shared.NAV_STACK_KEY
_SORT_OPTIONS: tuple[str, ...] = ("default", "title", "key", "kind")
_SORT_LABELS: dict[str, str] = {
    "default": "默认排序",
    "title": "标题",
    "key": "Key",
    "kind": "分类",
}
_DETAIL_TABS: tuple[str, ...] = ("overview", "relations", "source")
_DETAIL_TAB_LABELS: dict[str, str] = {
    "overview": "概览",
    "relations": "关系",
    "source": "来源",
}
_COLUMN_MODE_OPTIONS: tuple[str, ...] = ("compact", "standard", "full")
_COLUMN_MODE_LABELS: dict[str, str] = {
    "compact": "紧凑列",
    "standard": "标准列",
    "full": "完整列",
}
_RESULTS_COLLAPSE_OPTIONS: tuple[str, ...] = ("expanded", "collapsed")
_RESULTS_COLLAPSE_LABELS: dict[str, str] = {
    "expanded": "展开",
    "collapsed": "折叠",
}
_PAGE_SIZE_OPTIONS: tuple[int, ...] = (25, 50, 100, 250)


def dashboard_script_path() -> Path:
    return Path(__file__).resolve()


def build_streamlit_command(
    *,
    profile: str = "framework-core",
    scope: str = "framework",
    kind: str = "preset",
    query: str = "",
    project_path: str | None = None,
    include_global: bool = False,
    db_path: str | None = None,
    source_mode: str | None = None,
    column_mode: str = _shared.DEFAULT_COLUMN_MODE,
    page_size: int = _shared.DEFAULT_PAGE_SIZE,
    results_collapse: str = _shared.DEFAULT_RESULTS_COLLAPSE,
    host: str | None = None,
    port: int | None = None,
    headless: bool = False,
) -> list[str]:
    return _shell.build_streamlit_command(
        script_path=dashboard_script_path(),
        profile=profile,
        scope=scope,
        kind=kind,
        query=query,
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


def launch_catalog_dashboard(
    *,
    profile: str = "framework-core",
    scope: str = "framework",
    kind: str = "preset",
    query: str = "",
    project_path: str | None = None,
    include_global: bool = False,
    db_path: str | None = None,
    source_mode: str | None = None,
    column_mode: str = _shared.DEFAULT_COLUMN_MODE,
    page_size: int = _shared.DEFAULT_PAGE_SIZE,
    results_collapse: str = _shared.DEFAULT_RESULTS_COLLAPSE,
    host: str | None = None,
    port: int | None = None,
    headless: bool = False,
) -> int:
    return _shell.launch_catalog_dashboard(
        script_path=dashboard_script_path(),
        profile=profile,
        scope=scope,
        kind=kind,
        query=query,
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


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="mlblack catalog dashboard")
    parser.add_argument("--profile", type=str, default="framework-core")
    parser.add_argument("--scope", type=str, default="framework")
    parser.add_argument("--kind", type=str, default="preset")
    parser.add_argument("--query", type=str, default="")
    parser.add_argument("--project-path", type=str, default=None)
    parser.add_argument("--include-global", action="store_true")
    parser.add_argument("--db-path", type=str, default=None)
    parser.add_argument("--source-mode", type=str, default=None)
    parser.add_argument("--column-mode", type=str, default=_shared.DEFAULT_COLUMN_MODE, choices=list(_COLUMN_MODE_OPTIONS))
    parser.add_argument("--page-size", type=int, default=_shared.DEFAULT_PAGE_SIZE)
    parser.add_argument("--results-collapse", type=str, default=_shared.DEFAULT_RESULTS_COLLAPSE, choices=list(_RESULTS_COLLAPSE_OPTIONS))
    return parser.parse_known_args(argv)[0]


def _require_streamlit():
    try:
        import streamlit as st
    except Exception as exc:  # pragma: no cover
        raise SystemExit(
            "streamlit is required for catalog dashboard. Install with: python -m pip install streamlit"
        ) from exc
    return st


def _set_page_config(st: Any) -> None:
    try:
        st.set_page_config(page_title="mlblack Catalog", page_icon="MC", layout="wide", initial_sidebar_state="collapsed")
    except Exception:
        pass


def _inject_style(st: Any) -> None:
    st.markdown(
        (
            """
        <style>
        .block-container {padding-top: 1.02rem; padding-bottom: 1.25rem; max-width: 1560px;}
        [data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"] {display: none !important;}
        .catalog-hero {
            background: linear-gradient(135deg, #fdf3df 0%, #f8ddb0 46%, #eeaa57 100%);
            border: 1px solid rgba(122, 79, 30, 0.14);
            border-radius: 24px;
            padding: 1.15rem 1.3rem;
            margin-bottom: 1rem;
            box-shadow: 0 18px 40px rgba(79, 55, 27, 0.08);
        }
        .catalog-hero-head {
            display: flex;
            align-items: center;
            gap: 0.88rem;
            margin-bottom: 0.38rem;
        }
        .catalog-brand {display: flex; align-items: center; gap: 0.88rem;}
        .catalog-icon {
            width: 48px;
            height: 48px;
            border-radius: 14px;
            background: linear-gradient(180deg, rgba(122, 71, 16, 0.96), rgba(177, 101, 19, 0.96));
            color: #fff3df;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.05rem;
            font-weight: 900;
            letter-spacing: 0.06em;
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.08), 0 12px 24px rgba(112, 66, 14, 0.16);
        }
        .catalog-kicker {font-size: 0.78rem; letter-spacing: 0.08em; text-transform: uppercase; color: #7b4f1d; font-weight: 800;}
        .catalog-title {font-size: 2.0rem; line-height: 1.05; color: #2d1e12; font-weight: 800; margin: 0.18rem 0 0.34rem 0;}
        .catalog-sub {font-size: 0.96rem; color: #59442d; max-width: 82ch;}
        .catalog-inline-filters {
            border: 1px solid rgba(90, 62, 28, 0.12);
            border-radius: 18px;
            padding: 0.2rem 0.32rem 0.32rem 0.32rem;
            background: linear-gradient(180deg, rgba(255,255,255,0.96), rgba(249,245,239,0.96));
            margin-bottom: 0.9rem;
        }
        .catalog-stat {
            background: linear-gradient(180deg, rgba(255,255,255,0.96), rgba(248,244,238,0.96));
            border: 1px solid rgba(90, 62, 28, 0.12);
            border-radius: 18px;
            padding: 0.9rem 1rem;
            min-height: 110px;
        }
        .catalog-stat-label {font-size: 0.76rem; text-transform: uppercase; letter-spacing: 0.06em; color: #7b5b38; font-weight: 700;}
        .catalog-stat-value {font-size: 1.45rem; font-weight: 800; color: #2f2115; margin-top: 0.14rem;}
        .catalog-stat-note {font-size: 0.87rem; color: #65513a; margin-top: 0.16rem;}
        .catalog-chip {
            display: inline-block;
            margin: 0.12rem 0.32rem 0.12rem 0;
            padding: 0.18rem 0.48rem;
            border-radius: 999px;
            background: rgba(68, 47, 23, 0.08);
            color: #4e3419;
            font-size: 0.8rem;
            font-weight: 600;
        }
        .catalog-detail {
            background: white;
            border: 1px solid rgba(90, 62, 28, 0.12);
            border-radius: 20px;
            padding: 1rem 1.05rem;
        }
        .catalog-section-title {font-size: 0.92rem; color: #72502d; font-weight: 780; margin-top: 0.88rem; margin-bottom: 0.32rem;}
        .catalog-empty {
            border: 1px dashed rgba(101, 73, 41, 0.26);
            border-radius: 18px;
            padding: 1rem;
            background: rgba(255, 251, 246, 0.9);
            color: #5d4831;
        }
        .catalog-floating {
            position: sticky;
            top: 0.6rem;
            z-index: 30;
            background: linear-gradient(180deg, rgba(255,250,243,0.98), rgba(250,242,230,0.98));
            border: 1px solid rgba(101, 70, 35, 0.18);
            border-radius: 18px;
            padding: 0.82rem 0.9rem;
            margin-bottom: 0.9rem;
            box-shadow: 0 10px 24px rgba(77, 54, 29, 0.08);
        }
        .catalog-floating-label {font-size: 0.76rem; text-transform: uppercase; letter-spacing: 0.06em; color: #8a6439; font-weight: 700;}
        .catalog-floating-title {font-size: 1.02rem; color: #2f2115; font-weight: 800; margin-top: 0.18rem;}
        .catalog-floating-meta {font-size: 0.86rem; color: #6a543c; margin-top: 0.18rem;}
        .catalog-warning {
            border: 1px solid rgba(170, 118, 34, 0.25);
            border-radius: 16px;
            padding: 0.78rem 0.86rem;
            background: rgba(255, 248, 235, 0.96);
            color: #674c22;
            margin-bottom: 0.85rem;
        }
        .catalog-fab-stack {
            position: fixed;
            right: 1.15rem;
            bottom: 1.15rem;
            display: flex;
            flex-direction: column;
            gap: 0.62rem;
            z-index: 9998;
        }
        .catalog-fab {
            width: 48px;
            height: 48px;
            border-radius: 15px;
            position: relative;
            display: flex;
            align-items: center;
            justify-content: center;
            text-decoration: none !important;
            appearance: none;
            background: linear-gradient(180deg, rgba(98, 60, 18, 0.96), rgba(160, 97, 28, 0.96));
            color: #fff6e8 !important;
            box-shadow: 0 14px 30px rgba(83, 49, 12, 0.22);
            border: 1px solid rgba(255,255,255,0.1);
            cursor: pointer;
            padding: 0;
            transition: transform 120ms ease, box-shadow 120ms ease, opacity 120ms ease;
        }
        .catalog-fab:hover {
            transform: translateY(-2px);
            box-shadow: 0 18px 32px rgba(83, 49, 12, 0.26);
        }
        .catalog-fab-disabled {
            opacity: 0.42;
            pointer-events: none;
        }
        .catalog-fab[data-tooltip]::before,
        .catalog-fab[data-tooltip]::after {
            position: absolute;
            opacity: 0;
            pointer-events: none;
            transition: opacity 120ms ease, transform 120ms ease;
        }
        .catalog-fab[data-tooltip]::before {
            content: "";
            right: calc(100% + 6px);
            top: 50%;
            transform: translateY(-50%) translateX(6px);
            border-width: 6px 0 6px 7px;
            border-style: solid;
            border-color: transparent transparent transparent rgba(45, 31, 18, 0.96);
        }
        .catalog-fab[data-tooltip]::after {
            content: attr(data-tooltip);
            right: calc(100% + 14px);
            top: 50%;
            transform: translateY(-50%) translateX(6px);
            white-space: nowrap;
            padding: 0.42rem 0.58rem;
            border-radius: 10px;
            background: rgba(45, 31, 18, 0.96);
            color: #fff6ea;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.01em;
            box-shadow: 0 12px 24px rgba(31, 20, 10, 0.22);
        }
        .catalog-fab[data-tooltip]:hover::before,
        .catalog-fab[data-tooltip]:hover::after,
        .catalog-fab[data-tooltip]:focus-visible::before,
        .catalog-fab[data-tooltip]:focus-visible::after {
            opacity: 1;
            transform: translateY(-50%) translateX(0);
        }
        """
            + _page.PAGE_PROTOCOL_STYLE
            + """
        </style>
        """
        ),
        unsafe_allow_html=True,
    )


def _display_name(name: str) -> str:
    return str(name).replace("_", " ").title()


def _kind_label(kind: str) -> str:
    return _KIND_LABELS.get(str(kind or "").strip().lower(), _display_name(str(kind or "")))


def _field_label(name: str) -> str:
    key = str(name or "").strip()
    return _FIELD_LABELS.get(key, _display_name(key))


def _normalize_kind(kind: str | None) -> str:
    raw = str(kind or "").strip().lower()
    return raw if raw in _KIND_ORDER else "preset"


def _render_scalar(value: Any) -> str:
    if value in (None, "", (), [], {}):
        return ""
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, (list, tuple, set, frozenset)):
        return ", ".join(str(item) for item in value if str(item).strip())
    if isinstance(value, Mapping):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def _chips(values: Sequence[str]) -> str:
    return "".join(f"<span class='catalog-chip'>{escape(str(value))}</span>" for value in values if str(value).strip())


def _entry_title(item: Mapping[str, Any]) -> str:
    fields = dict(item.get("fields", {}) or {})
    title_zh = str(fields.get("title_zh", "") or "").strip()
    return title_zh or str(item.get("name", "") or item.get("key", "")).strip()


def _entry_summary(item: Mapping[str, Any]) -> str:
    fields = dict(item.get("fields", {}) or {})
    summary_zh = str(fields.get("summary_zh", "") or "").strip()
    return summary_zh or str(item.get("summary", "") or "").strip()


def _entry_use_when(item: Mapping[str, Any]) -> tuple[str, ...]:
    fields = dict(item.get("fields", {}) or {})
    raw = fields.get("use_when_zh") or fields.get("use_when") or ()
    return tuple(str(value).strip() for value in raw if str(value).strip())


def _sort_label(sort_name: str) -> str:
    key = str(sort_name or "").strip().lower()
    return _SORT_LABELS.get(key, key)


def _detail_tab_label(tab_name: str) -> str:
    key = str(tab_name or "").strip().lower()
    return _DETAIL_TAB_LABELS.get(key, key)


def _column_mode_label(mode_name: str) -> str:
    key = str(mode_name or "").strip().lower()
    return _COLUMN_MODE_LABELS.get(key, key)


def _results_collapse_label(mode_name: str) -> str:
    key = str(mode_name or "").strip().lower()
    return _RESULTS_COLLAPSE_LABELS.get(key, key)


def _normalize_sort_by(value: object) -> str:
    key = str(value or "").strip().lower()
    return key if key in _SORT_OPTIONS else _shared.DEFAULT_SORT_BY


def _normalize_sort_dir(value: object) -> str:
    key = str(value or "").strip().lower()
    return "desc" if key == "desc" else _shared.DEFAULT_SORT_DIR


def _normalize_detail_tab(value: object) -> str:
    key = str(value or "").strip().lower()
    return key if key in _DETAIL_TABS else _shared.DEFAULT_DETAIL_TAB


def _normalize_column_mode(value: object) -> str:
    key = str(value or "").strip().lower()
    return key if key in _COLUMN_MODE_OPTIONS else _shared.DEFAULT_COLUMN_MODE


def _normalize_page_size(value: object) -> int:
    try:
        page_size = int(str(value or "").strip())
    except Exception:
        return _shared.DEFAULT_PAGE_SIZE
    return page_size if page_size > 0 else _shared.DEFAULT_PAGE_SIZE


def _normalize_results_collapse(value: object) -> str:
    key = str(value or "").strip().lower()
    return key if key in _RESULTS_COLLAPSE_OPTIONS else _shared.DEFAULT_RESULTS_COLLAPSE


def _primary_controls_spec() -> _page.ControlRowSpec:
    return _page.ControlRowSpec(
        row_id=_page.PRIMARY_CONTROLS_ROW_ID,
        section_id="primary",
        slots=(
            _page.ControlSlotSpec("scope", 0.78, "视图"),
            _page.ControlSlotSpec("profile", 0.92, "Profile"),
            _page.ControlSlotSpec("kind", 0.95, "分类"),
            _page.ControlSlotSpec(
                "query",
                1.2,
                "关键词",
                placeholder="按当前分类搜索 key / 标题 / 摘要，也可配合下方字段筛选缩小范围",
            ),
        ),
    )


def _secondary_controls_spec() -> _page.ControlRowSpec:
    return _page.ControlRowSpec(
        row_id=_page.SECONDARY_CONTROLS_ROW_ID,
        section_id="secondary",
        slots=(
            _page.ControlSlotSpec(
                "project_path",
                1.2,
                "Project Path",
                help="项目视图下用于定位 .mlblack-project 或 catalog/entries.toml 所在目录。",
            ),
            _page.ControlSlotSpec("include_global", 0.82, "并入框架条目"),
            _page.ControlSlotSpec(
                "db_path",
                0.9,
                "DB Path / URL",
                help="留空时按 catalog/db.toml 或环境变量自动连接；填写后优先直连这个数据库。",
                placeholder="postgresql://postgres:password@localhost:5432/mlblack_catalog",
            ),
            _page.ControlSlotSpec(
                "source_mode",
                0.9,
                "Source Mode",
                help="自动 / 优先数据库 / 仅数据库 / 仅 registry。",
            ),
        ),
    )


def _view_state_key(scope: str, kind: str, name: str) -> str:
    return _shared.view_state_key(scope, kind, name)


def _facet_fields(kind: str, schema: Mapping[str, Any]) -> tuple[str, ...]:
    configured = tuple(_DEFAULT_FACET_FIELDS.get(kind, ()))
    if configured:
        return configured
    return tuple(str(name) for name in schema.get("fields", ())[:6])


def _freeze_filters(field_filters: Mapping[str, str] | None) -> tuple[tuple[str, str], ...]:
    frozen = _shell.freeze_filters(field_filters)
    return tuple((str(name), str(values[0])) for name, values in frozen if values)


def _thaw_filters(filters_key: tuple[tuple[str, str], ...]) -> dict[str, str]:
    thawed = _shell.thaw_filters(tuple((str(name), (str(value),)) for name, value in filters_key))
    return {str(name): str(value) for name, value in thawed.items() if str(name).strip() and str(value).strip()}


def _item_sort_value(item: Mapping[str, Any], sort_by: str) -> str:
    key = str(sort_by or "").strip().lower()
    if key == "title":
        return _entry_title(item).lower()
    if key == "kind":
        return str(item.get("kind", "") or "").strip().lower()
    if key == "key":
        return str(item.get("key", "") or "").strip().lower()
    return str(item.get("key", "") or "").strip().lower()


def _sorted_items(items: Sequence[Mapping[str, Any]], *, sort_by: str, sort_dir: str) -> list[Mapping[str, Any]]:
    rows = list(items)
    if _normalize_sort_by(sort_by) == _shared.DEFAULT_SORT_BY:
        return rows
    reverse = _normalize_sort_dir(sort_dir) == "desc"
    return sorted(rows, key=lambda item: (_item_sort_value(item, sort_by), str(item.get("key", "") or "").strip().lower()), reverse=reverse)


def _read_query_params(st: Any) -> tuple[dict[str, str], dict[str, tuple[str, ...]]]:
    return _shared.read_query_params(
        st,
        base_keys=(
            "profile",
            "scope",
            "kind",
            "query",
            "selected",
            "project_path",
            "include_global",
            "db_path",
            "source_mode",
            "sort_by",
            "sort_dir",
            "detail_tab",
            "open_relations",
            "column_mode",
            "page_size",
            "results_collapse",
            "nav_action",
        ),
    )


def _query_param_payload(
    *,
    profile: str,
    scope: str,
    kind: str,
    query: str,
    selected: str,
    project_path: str,
    include_global: bool,
    db_path: str,
    source_mode: str,
    sort_by: str,
    sort_dir: str,
    detail_tab: str,
    open_relations: str,
    column_mode: str,
    page_size: int,
    results_collapse: str,
    field_filters: Mapping[str, object] | None = None,
) -> dict[str, str]:
    return _shared.build_query_param_payload(
        base_params={
            "profile": str(profile),
            "scope": str(scope),
            "kind": str(kind),
            "query": str(query),
            "selected": str(selected),
            "project_path": str(project_path),
            "include_global": "1" if include_global else "",
            "db_path": str(db_path),
            "source_mode": str(source_mode),
            "sort_by": str(sort_by),
            "sort_dir": str(sort_dir),
            "detail_tab": str(detail_tab),
            "open_relations": str(open_relations),
            "column_mode": str(column_mode),
            "page_size": str(int(page_size)),
            "results_collapse": str(results_collapse),
        },
        field_filters=field_filters,
        none_sentinel=_NO_SELECTION,
    )


def _build_deep_link_query(
    *,
    profile: str,
    scope: str,
    kind: str,
    query: str,
    selected: str,
    project_path: str,
    include_global: bool,
    db_path: str,
    source_mode: str,
    sort_by: str,
    sort_dir: str,
    detail_tab: str,
    open_relations: str,
    column_mode: str,
    page_size: int,
    results_collapse: str,
    field_filters: Mapping[str, object] | None = None,
) -> str:
    return _shared.build_deep_link_query(
        base_params={
            "profile": str(profile),
            "scope": str(scope),
            "kind": str(kind),
            "query": str(query),
            "selected": str(selected),
            "project_path": str(project_path),
            "include_global": "1" if include_global else "",
            "db_path": str(db_path),
            "source_mode": str(source_mode),
            "sort_by": str(sort_by),
            "sort_dir": str(sort_dir),
            "detail_tab": str(detail_tab),
            "open_relations": str(open_relations),
            "column_mode": str(column_mode),
            "page_size": str(int(page_size)),
            "results_collapse": str(results_collapse),
        },
        field_filters=field_filters,
        none_sentinel=_NO_SELECTION,
    )


def _write_query_params(
    st: Any,
    *,
    profile: str,
    scope: str,
    kind: str,
    query: str,
    selected: str,
    project_path: str,
    include_global: bool,
    db_path: str,
    source_mode: str,
    sort_by: str,
    sort_dir: str,
    detail_tab: str,
    open_relations: str,
    column_mode: str,
    page_size: int,
    results_collapse: str,
    field_filters: Mapping[str, object] | None = None,
) -> None:
    _shared.write_query_params(
        st,
        base_params={
            "profile": str(profile),
            "scope": str(scope),
            "kind": str(kind),
            "query": str(query),
            "selected": str(selected),
            "project_path": str(project_path),
            "include_global": "1" if include_global else "",
            "db_path": str(db_path),
            "source_mode": str(source_mode),
            "sort_by": str(sort_by),
            "sort_dir": str(sort_dir),
            "detail_tab": str(detail_tab),
            "open_relations": str(open_relations),
            "column_mode": str(column_mode),
            "page_size": str(int(page_size)),
            "results_collapse": str(results_collapse),
        },
        field_filters=field_filters,
        none_sentinel=_NO_SELECTION,
    )


def _deep_link_with_nav_action(deep_link_query: str, *, action: str) -> str:
    raw = str(deep_link_query or "").strip()
    query_only, _, fragment = raw.partition("#")
    query_text = query_only[1:] if query_only.startswith("?") else query_only
    params = [(key, value) for key, value in parse_qsl(query_text, keep_blank_values=True) if key != "nav_action"]
    params.append(("nav_action", str(action).strip()))
    rebuilt = "?" + urlencode(params)
    if fragment:
        return f"{rebuilt}#{fragment}"
    return rebuilt


def _has_active_field_filters(field_filters: Mapping[str, object] | None) -> bool:
    if not field_filters:
        return False
    return any(str(value or "").strip() for value in field_filters.values())


def _write_locate_state_and_rerun(
    st: Any,
    *,
    profile: str,
    scope: str,
    kind: str,
    query: str,
    selected: str,
    project_path: str,
    include_global: bool,
    db_path: str,
    source_mode: str,
    sort_by: str,
    sort_dir: str,
    detail_tab: str,
    open_relations: str,
    column_mode: str,
    page_size: int,
    results_collapse: str,
    field_filters: Mapping[str, object] | None,
) -> None:
    _write_query_params(
        st,
        profile=profile,
        scope=scope,
        kind=kind,
        query=query,
        selected=selected,
        project_path=project_path,
        include_global=include_global,
        db_path=db_path,
        source_mode=source_mode,
        sort_by=sort_by,
        sort_dir=sort_dir,
        detail_tab=detail_tab,
        open_relations=open_relations,
        column_mode=column_mode,
        page_size=page_size,
        results_collapse=results_collapse,
        field_filters=field_filters,
    )
    _rerun(st)


def _rerun(st: Any) -> None:
    _shared.rerun(st)


def _normalize_navigation_stack(values: object) -> list[dict[str, str]]:
    return _shared.normalize_navigation_stack(values)


def _navigation_stack(st: Any) -> list[dict[str, str]]:
    return _shared.navigation_stack(st, state_key=_NAV_STACK_KEY)


def _push_navigation_stack(st: Any, *, current_entry: Mapping[str, Any] | None) -> None:
    payload = None if current_entry is None else {**dict(current_entry), "title": _entry_title(current_entry)}
    _shared.push_navigation_stack(st, current_entry=payload, state_key=_NAV_STACK_KEY)


def _pop_navigation_stack(st: Any) -> dict[str, str] | None:
    return _shared.pop_navigation_stack(st, state_key=_NAV_STACK_KEY)


def _restore_navigation_index(st: Any, index: int) -> dict[str, str] | None:
    return _shared.restore_navigation_index(st, index, state_key=_NAV_STACK_KEY)


def _selected_table_row_indices(event: Any) -> tuple[int, ...]:
    if event is None:
        return ()
    selection = getattr(event, "selection", None)
    if selection is not None:
        rows = getattr(selection, "rows", None)
        if rows is not None:
            return tuple(int(value) for value in rows)
    if isinstance(event, Mapping):
        payload = event.get("selection")
        if isinstance(payload, Mapping):
            rows = payload.get("rows")
            if isinstance(rows, Sequence):
                return tuple(int(value) for value in rows)
    return ()


def _selection_state(selected_key: str, items: Sequence[Mapping[str, Any]], *, selected_exists: bool) -> dict[str, Any]:
    return _shared.selection_state(selected_key, items, selected_exists=selected_exists, none_sentinel=_NO_SELECTION)


def _result_rows(items: Sequence[Mapping[str, Any]], facet_fields: Sequence[str], *, column_mode: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    normalized_mode = _normalize_column_mode(column_mode)
    facet_limit = 0
    if normalized_mode == "standard":
        facet_limit = 2
    elif normalized_mode == "full":
        facet_limit = 4
    for index, item in enumerate(items, start=1):
        fields = dict(item.get("fields", {}) or {})
        row: dict[str, str] = {
            "序号": str(index),
            "标题": _entry_title(item),
            "Key": str(item.get("key", "") or ""),
        }
        if normalized_mode in {"standard", "full"}:
            row["摘要"] = _entry_summary(item)
        for field_name in facet_fields[:facet_limit]:
            rendered = _render_scalar(fields.get(field_name))
            if rendered:
                row[_field_label(field_name)] = rendered
        rows.append(row)
    return rows


def _visible_result_items(items: Sequence[Mapping[str, Any]], *, page_size: int) -> list[Mapping[str, Any]]:
    return list(items[: _normalize_page_size(page_size)])


def _clear_scope_kind_filters(st: Any, *, scope: str, kind: str, facet_fields: Sequence[str]) -> None:
    _shared.clear_scope_kind_filters(st, scope=scope, kind=kind, facet_fields=facet_fields)


def _scroll_to_anchor(st: Any, *, anchor_id: str) -> None:
    try:
        from streamlit.components.v1 import html

        html(
            f"""
            <script>
            const target = window.parent.document.getElementById({anchor_id!r});
            if (target) {{
                target.scrollIntoView({{ behavior: "smooth", block: "start" }});
            }}
            </script>
            """,
            height=0,
            width=0,
        )
    except Exception:
        return


def _scroll_action_js(anchor_id: str) -> str:
    safe_anchor = json.dumps(str(anchor_id))
    return (
        "const target = window.parent.document.getElementById("
        f"{safe_anchor}"
        "); if (target) { target.scrollIntoView({ behavior: 'smooth', block: 'start' }); }"
    )


def _floating_nav_markup(
    *,
    locate_target: str | None,
    locate_tooltip: str = "定位当前选中项",
    top_target: str = "catalog-page-top",
) -> str:
    target_svg = (
        "<svg viewBox='0 0 24 24' width='22' height='22' aria-hidden='true' focusable='false'>"
        "<circle cx='12' cy='12' r='7.25' fill='none' stroke='currentColor' stroke-width='1.8'/>"
        "<circle cx='12' cy='12' r='2.5' fill='currentColor'/>"
        "<path d='M12 2.75v3.1M12 18.15v3.1M2.75 12h3.1M18.15 12h3.1' fill='none' stroke='currentColor' stroke-width='1.8' stroke-linecap='round'/>"
        "</svg>"
    )
    top_svg = (
        "<svg viewBox='0 0 24 24' width='22' height='22' aria-hidden='true' focusable='false'>"
        "<path d='M12 5l-6 6m6-6l6 6M12 5v14' fill='none' stroke='currentColor' stroke-width='1.9' stroke-linecap='round' stroke-linejoin='round'/>"
        "</svg>"
    )
    if locate_target:
        locate_button = (
            "<button type='button' class='catalog-fab' "
            f"onclick='{escape(_scroll_action_js(str(locate_target)), quote=True)}' "
            f"title='{escape(str(locate_tooltip), quote=True)}' "
            f"aria-label='{escape(str(locate_tooltip), quote=True)}' "
            f"data-tooltip='{escape(str(locate_tooltip), quote=True)}' "
            f"data-scroll-target='{escape(str(locate_target), quote=True)}'>"
            f"{target_svg}"
            "</button>"
        )
    else:
        locate_button = (
            "<span class='catalog-fab catalog-fab-disabled' "
            f"title='{escape(str(locate_tooltip), quote=True)}' "
            f"aria-label='{escape(str(locate_tooltip), quote=True)}' "
            f"data-tooltip='{escape(str(locate_tooltip), quote=True)}'>"
            f"{target_svg}"
            "</span>"
        )
    top_button = (
        "<button type='button' class='catalog-fab' "
        f"onclick='{escape(_scroll_action_js(str(top_target)), quote=True)}' "
        "title='回到页面顶部' aria-label='回到页面顶部' data-tooltip='回到页面顶部' "
        f"data-scroll-target='{escape(str(top_target), quote=True)}'>"
        f"{top_svg}"
        "</button>"
    )
    return f"<div class='catalog-fab-stack'>{locate_button}{top_button}</div>"


def _render_floating_nav(
    st: Any,
    *,
    locate_target: str | None,
    locate_tooltip: str = "定位当前选中项",
    top_target: str = "catalog-page-top",
) -> None:
    st.markdown(
        _floating_nav_markup(
            locate_target=locate_target,
            locate_tooltip=locate_tooltip,
            top_target=top_target,
        ),
        unsafe_allow_html=True,
    )


def _selection_presence(
    selected_key: str,
    *,
    items: Sequence[Mapping[str, Any]],
    visible_items: Sequence[Mapping[str, Any]],
) -> tuple[bool, bool]:
    key = str(selected_key or "").strip()
    if not key:
        return False, False
    all_keys = {str(item.get("key", "") or "").strip() for item in items}
    visible_keys = {str(item.get("key", "") or "").strip() for item in visible_items}
    return key in all_keys, key in visible_keys


def _copy_current_url(st: Any, *, key: str) -> None:
    if not st.button("复制 Deep-Link", key=key, use_container_width=True):
        return
    try:
        from streamlit.components.v1 import html

        html(
            """
            <script>
            const current = window.parent.location.href;
            navigator.clipboard.writeText(current);
            </script>
            """,
            height=0,
            width=0,
        )
        try:
            st.toast("已复制当前页面链接")
        except Exception:
            st.success("已复制当前页面链接")
    except Exception:
        st.info("浏览器未允许自动复制，请手动复制下方 deep-link。")


def _source_badges(entry: Mapping[str, Any], source_info: Mapping[str, Any]) -> tuple[str, ...]:
    return (
        f"entry:{str(entry.get('source', 'unknown'))}",
        f"view:{str(source_info.get('effective_source', 'registry'))}",
        f"scope:{str(source_info.get('scope', 'framework'))}",
    )


def _resolve_source_file(entry: Mapping[str, Any], *, project_root: str | None = None) -> Path | None:
    raw_path = str(entry.get("path", "") or "").strip()
    if raw_path:
        path = Path(raw_path)
        if not path.is_absolute():
            base = Path(project_root).resolve() if str(entry.get("source", "")).strip() == "project" and project_root else ROOT
            path = (base / path).resolve()
        if path.exists():
            return path

    if str(entry.get("source", "")).strip() in {"registry", "derived_registry"}:
        candidate = ROOT / "catalog" / "registry.py"
        if candidate.exists():
            return candidate.resolve()

    fields = dict(entry.get("fields", {}) or {})
    module_name = str(fields.get("module", "") or "").strip()
    if not module_name:
        return None
    try:
        spec = importlib.util.find_spec(module_name)
    except Exception:
        spec = None
    if spec is None or not spec.origin:
        return None
    origin = Path(str(spec.origin))
    return origin.resolve() if origin.exists() else None


def _open_source_file(path: Path) -> bool:
    target = Path(path).resolve()
    if not target.exists():
        return False
    try:
        if sys.platform.startswith("win"):
            os.startfile(str(target))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(target)])
        else:
            subprocess.Popen(["xdg-open", str(target)])
        return True
    except Exception:
        return False


def _reveal_source_file(path: Path) -> bool:
    target = Path(path).resolve()
    if not target.exists():
        return False
    try:
        if sys.platform.startswith("win"):
            subprocess.Popen(["explorer", "/select,", str(target)])
        elif sys.platform == "darwin":
            subprocess.Popen(["open", "-R", str(target)])
        else:
            subprocess.Popen(["xdg-open", str(target.parent)])
        return True
    except Exception:
        return False


def _field_rows(mapping: Mapping[str, Any], *, exclude: Sequence[str] | None = None) -> list[dict[str, str]]:
    excluded = {str(value).strip() for value in tuple(exclude or ()) if str(value).strip()}
    rows: list[dict[str, str]] = []
    for key, value in mapping.items():
        if str(key) in excluded:
            continue
        rendered = _render_scalar(value)
        if not rendered:
            continue
        rows.append({"字段": _field_label(str(key)), "值": rendered})
    return rows


def _filter_jump_values(value: Any) -> tuple[str, ...]:
    if value in (None, "", (), [], {}):
        return tuple()
    if isinstance(value, Mapping):
        return tuple(str(key).strip() for key in value.keys() if str(key).strip())
    if isinstance(value, (list, tuple, set, frozenset)):
        return tuple(str(item).strip() for item in value if str(item).strip())
    text = str(value).strip()
    return tuple() if not text else (text,)


def _set_detail_filter_jump(
    st: Any,
    *,
    scope: str,
    kind: str,
    field_name: str,
    field_value: str,
) -> None:
    state_key = _shared.facet_state_key(scope, kind, str(field_name))
    st.session_state[state_key] = str(field_value)


def _catalog_kind_from_target(*, target_key: str, target_kind: str = "") -> str:
    raw_kind = str(target_kind or "").strip().lower()
    if not raw_kind and ":" in str(target_key):
        raw_kind = str(target_key).split(":", 1)[0].strip().lower()
    return raw_kind if raw_kind in _KIND_ORDER else ""


def _select_catalog_entry(
    st: Any,
    *,
    target_key: str,
    target_kind: str = "",
) -> None:
    st.session_state["catalog_ui_selected"] = str(target_key or "")
    resolved_kind = _catalog_kind_from_target(target_key=target_key, target_kind=target_kind)
    if resolved_kind:
        st.session_state["catalog_ui_kind"] = resolved_kind


def _callback_select_entry(st: Any, *, target_key: str, target_kind: str = "") -> None:
    _select_catalog_entry(st, target_key=target_key, target_kind=target_kind)


def _callback_reveal_selected(st: Any, *, scope: str, kind: str, facet_fields: Sequence[str]) -> None:
    st.session_state["catalog_ui_query"] = ""
    _clear_scope_kind_filters(st, scope=scope, kind=kind, facet_fields=facet_fields)


def _callback_locate_selected(st: Any) -> None:
    st.session_state[_PENDING_SCROLL_TARGET_KEY] = "catalog-results-anchor"


def _callback_pop_navigation(st: Any) -> None:
    target = _pop_navigation_stack(st)
    if target is not None:
        _select_catalog_entry(
            st,
            target_key=str(target.get("key", "") or ""),
            target_kind=str(target.get("kind", "") or ""),
        )


def _callback_restore_navigation(st: Any, *, index: int) -> None:
    target = _restore_navigation_index(st, index)
    if target is not None:
        _select_catalog_entry(
            st,
            target_key=str(target.get("key", "") or ""),
            target_kind=str(target.get("kind", "") or ""),
        )


def _callback_clear_selected(st: Any) -> None:
    st.session_state["catalog_ui_selected"] = _NO_SELECTION


def _callback_jump_relation(
    st: Any,
    *,
    current_entry: Mapping[str, Any] | None,
    target_key: str,
    target_kind: str = "",
) -> None:
    _push_navigation_stack(st, current_entry=current_entry)
    _select_catalog_entry(st, target_key=target_key, target_kind=target_kind)


def _callback_clear_filters(st: Any, *, scope: str, kind: str, facet_fields: Sequence[str]) -> None:
    _clear_scope_kind_filters(st, scope=scope, kind=kind, facet_fields=facet_fields)


def _callback_field_jump(
    st: Any,
    *,
    scope: str,
    kind: str,
    field_name: str,
    field_value: str,
) -> None:
    _set_detail_filter_jump(
        st,
        scope=scope,
        kind=kind,
        field_name=field_name,
        field_value=field_value,
    )


def _render_filter_jump_section(
    st: Any,
    *,
    scope: str,
    kind: str,
    fields: Mapping[str, Any],
    facet_fields: Sequence[str],
) -> None:
    allowed = {str(name).strip() for name in tuple(facet_fields) if str(name).strip()}
    groups: list[tuple[str, tuple[str, ...]]] = []
    for field_name in (
        "search_mechanism_keys",
        "search_family_signature_mechanisms",
        "search_checkpointable_mechanisms",
        "search_replayable_mechanisms",
        "artifact_stability_fields",
    ):
        if field_name not in allowed:
            continue
        values = _filter_jump_values(fields.get(field_name))
        if values:
            groups.append((field_name, values))

    if not groups:
        return

    st.markdown("<div class='catalog-section-title'>字段跳转</div>", unsafe_allow_html=True)
    st.caption("点击下面的字段值，会直接把它写入当前字段筛选。")
    for field_name, values in groups:
        st.markdown(f"**{_field_label(field_name)}**")
        visible_values = tuple(values[:12])
        cols = st.columns(min(4, max(1, len(visible_values))))
        for index, value in enumerate(visible_values):
            with cols[index % len(cols)]:
                st.button(
                    str(value),
                    key=f"catalog_ui::field_jump::{scope}::{kind}::{field_name}::{index}::{value}",
                    use_container_width=True,
                    on_click=_callback_field_jump,
                    kwargs={
                        "st": st,
                        "scope": scope,
                        "kind": kind,
                        "field_name": field_name,
                        "field_value": str(value),
                    },
                )
        if len(values) > len(visible_values):
            st.caption(f"其余 {len(values) - len(visible_values)} 个值可继续从字段筛选里缩小范围。")


def _render_navigation_stack(st: Any, *, current_key: str) -> None:
    stack = _navigation_stack(st)
    if not stack:
        return
    st.markdown("<div class='catalog-section-title'>跳转栈</div>", unsafe_allow_html=True)
    for index, item in enumerate(reversed(stack[-6:])):
        real_index = len(stack) - 1 - index
        label = str(item.get("title", "") or item.get("key", ""))
        key = str(item.get("key", "") or "")
        if not key:
            continue
        if key == current_key:
            st.caption(f"当前：{label} | {key}")
            continue
        st.button(
            f"返回 {label} | {key}",
            key=f"catalog_ui::stack::{real_index}::{key}",
            use_container_width=True,
            on_click=_callback_restore_navigation,
            kwargs={"st": st, "index": real_index},
        )


def _raw_relation_neighbor_groups(entry: Mapping[str, Any] | None) -> dict[str, list[dict[str, Any]]]:
    relations = dict((entry or {}).get("relations", {}) or {})
    groups: dict[str, list[dict[str, Any]]] = {}
    for relation_name, relation_value in relations.items():
        rows: list[dict[str, Any]] = []
        for candidate_key in _filter_jump_values(relation_value):
            key = str(candidate_key).strip()
            if not key:
                continue
            guessed_kind = _catalog_kind_from_target(target_key=key)
            rows.append(
                {
                    "key": key,
                    "kind": guessed_kind,
                    "name": key.split(":", 1)[-1] if ":" in key else key,
                    "summary": "",
                    "fields": {},
                    "fallback": True,
                }
            )
        if rows:
            groups[str(relation_name)] = rows
    return groups


def _relation_neighbor_groups(
    *,
    entry: Mapping[str, Any] | None,
    neighbors: Mapping[str, Any] | None,
) -> dict[str, list[dict[str, Any]]]:
    merged = {
        str(name): list(rows)
        for name, rows in dict((neighbors or {}).get("neighbors", {}) or {}).items()
        if rows
    }
    for relation_name, rows in _raw_relation_neighbor_groups(entry).items():
        merged.setdefault(str(relation_name), rows)
    return merged


def _load_source_info(
    profile: str,
    scope: str,
    project_path: str,
    include_global: bool,
    db_path: str,
    source_mode: str,
) -> dict[str, Any]:
    return catalog_source_info(
        profile=profile,
        scope=scope,
        project_path=project_path or None,
        include_global=include_global,
        db_path=db_path or None,
        source_mode=source_mode or None,
    )


_cached_source_info = _shell.memoize_loader(_load_source_info, maxsize=64)


def _load_summary(
    profile: str,
    scope: str,
    project_path: str,
    include_global: bool,
    db_path: str,
    source_mode: str,
) -> dict[str, Any]:
    return catalog_summary(
        profile=profile,
        scope=scope,
        project_path=project_path or None,
        include_global=include_global,
        db_path=db_path or None,
        source_mode=source_mode or None,
    )


_cached_summary = _shell.memoize_loader(_load_summary, maxsize=64)


def _load_schema(
    profile: str,
    scope: str,
    project_path: str,
    include_global: bool,
    kind: str,
    db_path: str,
    source_mode: str,
) -> dict[str, Any]:
    return catalog_schema(
        profile=profile,
        scope=scope,
        project_path=project_path or None,
        include_global=include_global,
        kind=kind,
        db_path=db_path or None,
        source_mode=source_mode or None,
    )


_cached_schema = _shell.memoize_loader(_load_schema, maxsize=128)


def _load_facets(
    profile: str,
    scope: str,
    project_path: str,
    include_global: bool,
    kind: str,
    query: str,
    filters_key: tuple[tuple[str, str], ...],
    db_path: str,
    source_mode: str,
) -> dict[str, Any]:
    return catalog_facets(
        profile=profile,
        scope=scope,
        project_path=project_path or None,
        include_global=include_global,
        kind=kind,
        query=query,
        field_filters=_thaw_filters(filters_key),
        fields=_DEFAULT_FACET_FIELDS.get(kind),
        limit_per_field=24,
        db_path=db_path or None,
        source_mode=source_mode or None,
    )


_cached_facets = _shell.memoize_loader(_load_facets, maxsize=256)


def _load_items(
    profile: str,
    scope: str,
    project_path: str,
    include_global: bool,
    kind: str,
    query: str,
    filters_key: tuple[tuple[str, str], ...],
    db_path: str,
    source_mode: str,
) -> tuple[dict[str, Any], ...]:
    filters = _thaw_filters(filters_key)
    entries = (
        search_entries(
            query,
            profile=profile,
            scope=scope,
            project_path=project_path or None,
            include_global=include_global,
            kind=kind,
            limit=250,
            field_filters=filters,
            db_path=db_path or None,
            source_mode=source_mode or None,
        )
        if str(query).strip()
        else list_entries(
            profile=profile,
            scope=scope,
            project_path=project_path or None,
            include_global=include_global,
            kind=kind,
            limit=250,
            field_filters=filters,
            db_path=db_path or None,
            source_mode=source_mode or None,
        )
    )
    return tuple(entry.to_dict() for entry in entries)


_cached_items = _shell.memoize_loader(_load_items, maxsize=256)


def _load_selected(
    profile: str,
    scope: str,
    project_path: str,
    include_global: bool,
    selected_key: str,
    db_path: str,
    source_mode: str,
) -> dict[str, Any] | None:
    key = str(selected_key or "").strip()
    if not key:
        return None
    entry = show_entry(
        key,
        profile=profile,
        scope=scope,
        project_path=project_path or None,
        include_global=include_global,
        db_path=db_path or None,
        source_mode=source_mode or None,
    )
    return None if entry is None else entry.to_dict()


_cached_selected = _shell.memoize_loader(_load_selected, maxsize=256)


def _load_neighbors(
    profile: str,
    scope: str,
    project_path: str,
    include_global: bool,
    selected_key: str,
    db_path: str,
    source_mode: str,
) -> dict[str, Any] | None:
    key = str(selected_key or "").strip()
    if not key:
        return None
    return catalog_neighbors(
        key,
        profile=profile,
        scope=scope,
        project_path=project_path or None,
        include_global=include_global,
        db_path=db_path or None,
        source_mode=source_mode or None,
    )


_cached_neighbors = _shell.memoize_loader(_load_neighbors, maxsize=256)


def _render_selection_float(
    st: Any,
    *,
    selection: Mapping[str, Any],
    selected_entry: Mapping[str, Any] | None,
    visible_items: Sequence[Mapping[str, Any]],
    scope: str,
    kind: str,
    facet_fields: Sequence[str],
) -> None:
    selected_key = str(selection.get("selected_key", "") or "").strip()
    if not selected_key:
        return

    title = _entry_title(selected_entry or {"key": selected_key, "name": selected_key})
    row_index = selection.get("row_index")
    if selection.get("visible", False) and isinstance(row_index, int):
        meta = f"当前选中项位于结果表格第 {int(row_index) + 1} 行。"
    elif selection.get("hidden", False):
        meta = "当前选中项被筛选条件暂时隐藏。"
    else:
        meta = "当前选中项不在结果表格里。"

    st.markdown(
        (
            "<div class='catalog-floating'>"
            "<div class='catalog-floating-label'>Current Selection</div>"
            f"<div class='catalog-floating-title'>{escape(title)}</div>"
            f"<div class='catalog-floating-meta'><code>{escape(selected_key)}</code><br/>{escape(meta)}</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )

    visible_rows = list(visible_items or ())
    if selection.get("visible", False) and isinstance(row_index, int) and len(visible_rows) > 1:
        nav_cols = st.columns((1.0, 1.0))
        prev_disabled = int(row_index) <= 0
        next_disabled = int(row_index) >= len(visible_rows) - 1
        prev_target = visible_rows[max(0, int(row_index) - 1)]
        next_target = visible_rows[min(len(visible_rows) - 1, int(row_index) + 1)]
        nav_cols[0].button(
            "上一项",
            key=f"catalog_ui::prev::{scope}::{kind}",
            use_container_width=True,
            disabled=prev_disabled,
            on_click=_callback_select_entry,
            kwargs={
                "st": st,
                "target_key": str(prev_target.get("key", "") or ""),
                "target_kind": str(prev_target.get("kind", "") or ""),
            },
        )
        nav_cols[1].button(
            "下一项",
            key=f"catalog_ui::next::{scope}::{kind}",
            use_container_width=True,
            disabled=next_disabled,
            on_click=_callback_select_entry,
            kwargs={
                "st": st,
                "target_key": str(next_target.get("key", "") or ""),
                "target_kind": str(next_target.get("kind", "") or ""),
            },
        )
    stack = _navigation_stack(st)
    action_cols = st.columns((1.0, 1.0, 1.0))
    if selection.get("hidden", False):
        action_cols[0].button(
            "显示它",
            key=f"catalog_ui::reveal::{scope}::{kind}",
            use_container_width=True,
            on_click=_callback_reveal_selected,
            kwargs={"st": st, "scope": scope, "kind": kind, "facet_fields": facet_fields},
        )
    else:
        action_cols[0].button(
            "定位到结果区",
            key=f"catalog_ui::locate_selection::{scope}::{kind}",
            use_container_width=True,
            on_click=_callback_locate_selected,
            kwargs={"st": st},
        )
    if stack:
        action_cols[1].button(
            "返回上一个",
            key=f"catalog_ui::back::{scope}::{kind}",
            use_container_width=True,
            on_click=_callback_pop_navigation,
            kwargs={"st": st},
        )
    else:
        action_cols[1].caption("跳转栈为空")
    action_cols[2].button(
        "清除选中",
        key=f"catalog_ui::clear_selected::{scope}::{kind}",
        use_container_width=True,
        on_click=_callback_clear_selected,
        kwargs={"st": st},
    )


def _render_results_table(
    st: Any,
    *,
    items: Sequence[Mapping[str, Any]],
    scope: str,
    kind: str,
    facet_fields: Sequence[str],
    column_mode: str,
) -> str:
    raw_selected_key = str(st.session_state.get("catalog_ui_selected", "") or "")
    cleared_selection = raw_selected_key == _NO_SELECTION
    selected_key = "" if cleared_selection else raw_selected_key
    if not items:
        st.markdown("<div class='catalog-empty'>当前筛选条件下没有结果。可以清空筛选或修改关键词。</div>", unsafe_allow_html=True)
        return ""

    try:
        table_event = st.dataframe(
            pd.DataFrame(_result_rows(items, facet_fields, column_mode=column_mode)),
            use_container_width=True,
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
            key=f"catalog_ui::results::{scope}::{kind}",
        )
        selected_rows = _selected_table_row_indices(table_event)
        if selected_rows:
            index = int(selected_rows[0])
            if 0 <= index < len(items):
                selected_key = str(items[index].get("key", ""))
    except Exception:
        st.table(_result_rows(items, facet_fields, column_mode=column_mode))

    if not selected_key and not cleared_selection:
        selected_key = str(items[0].get("key", ""))
    return selected_key or (_NO_SELECTION if cleared_selection else "")


def _render_detail(
    st: Any,
    *,
    entry: Mapping[str, Any] | None,
    neighbors: Mapping[str, Any] | None,
    source_info: Mapping[str, Any],
    deep_link_query: str,
    scope: str,
    kind: str,
    detail_tab: str,
    expanded_relation_groups: Sequence[str],
) -> None:
    if not entry:
        st.info("当前没有选中条目。")
        return

    entry_key = str(entry.get("key", "") or "")
    source_path = _resolve_source_file(entry, project_root=str(source_info.get("project_root", "") or "") or None)
    st.markdown("<div class='catalog-detail'>", unsafe_allow_html=True)
    st.markdown(f"## {_entry_title(entry)}")
    st.code(entry_key, language=None)
    badge_values = _source_badges(entry, source_info) + tuple(str(tag) for tag in entry.get("tags", ()) if str(tag).strip())
    if badge_values:
        st.markdown(_chips(badge_values), unsafe_allow_html=True)
    summary = _entry_summary(entry)
    if summary:
        st.markdown(summary)

    _render_navigation_stack(st, current_key=entry_key)
    current_tab = _normalize_detail_tab(detail_tab)
    expanded_groups = {str(value).strip() for value in expanded_relation_groups if str(value).strip()}

    if current_tab == "overview":
        st.markdown("<div class='catalog-section-title'>适用场景</div>", unsafe_allow_html=True)
        use_when = _entry_use_when(entry)
        if use_when:
            for row in use_when:
                st.markdown(f"- {row}")
        else:
            st.caption("无")

        fields = dict(entry.get("fields", {}) or {})
        field_rows = _field_rows(fields, exclude=("title_zh", "summary_zh", "use_when_zh"))
        if field_rows:
            st.markdown("<div class='catalog-section-title'>字段</div>", unsafe_allow_html=True)
            st.dataframe(pd.DataFrame(field_rows), use_container_width=True, hide_index=True)
        _render_filter_jump_section(
            st,
            scope=scope,
            kind=kind,
            fields=fields,
            facet_fields=_facet_fields(kind, {"fields": tuple(fields.keys())}),
        )

    elif current_tab == "relations":
        relations = dict(entry.get("relations", {}) or {})
        relation_rows = _field_rows(relations)
        if relation_rows:
            st.markdown("<div class='catalog-section-title'>关系</div>", unsafe_allow_html=True)
            st.dataframe(pd.DataFrame(relation_rows), use_container_width=True, hide_index=True)

        neighbor_groups = _relation_neighbor_groups(entry=entry, neighbors=neighbors)
        st.markdown("<div class='catalog-section-title'>关系跳转</div>", unsafe_allow_html=True)
        if not neighbor_groups:
            st.info("当前条目还没有可跳转的关系。")
        else:
            for relation_name, rows in neighbor_groups.items():
                if not rows:
                    continue
                with st.expander(f"{_field_label(relation_name)} ({len(rows)})", expanded=relation_name in expanded_groups):
                    for index, row in enumerate(rows):
                        target_key = str(row.get("key", "") or "")
                        target_title = str(dict(row.get("fields", {}) or {}).get("title_zh", "") or row.get("name", "") or target_key)
                        cols = st.columns((0.72, 0.28))
                        with cols[0]:
                            st.markdown(f"**{target_title or target_key}**")
                            st.caption(f"{_kind_label(str(row.get('kind', '') or ''))} | {target_key}")
                            st.caption(str(dict(row.get("fields", {}) or {}).get("summary_zh", "") or row.get("summary", "") or ""))
                        with cols[1]:
                            if target_key and not bool(row.get("missing")):
                                st.button(
                                    "跳转",
                                    key=f"catalog_ui::jump::{relation_name}::{index}::{target_key}",
                                    use_container_width=True,
                                    on_click=_callback_jump_relation,
                                    kwargs={
                                        "st": st,
                                        "current_entry": entry,
                                        "target_key": target_key,
                                        "target_kind": str(row.get("kind", "") or ""),
                                    },
                                )
                            else:
                                st.caption("缺失")

    else:
        st.markdown("<div class='catalog-section-title'>来源与链接</div>", unsafe_allow_html=True)
        action_cols = st.columns((1.1, 0.95, 0.95))
        with action_cols[0]:
            _copy_current_url(st, key=f"catalog_ui::copy::{entry_key}")
        with action_cols[1]:
            if source_path and st.button("打开 Source File", key=f"catalog_ui::open::{entry_key}", use_container_width=True):
                if _open_source_file(source_path):
                    try:
                        st.toast("已打开 source file")
                    except Exception:
                        st.success("已打开 source file")
                else:
                    st.warning("打开 source file 失败。")
        with action_cols[2]:
            if source_path and st.button("定位到 Source File", key=f"catalog_ui::reveal::{entry_key}", use_container_width=True):
                if _reveal_source_file(source_path):
                    try:
                        st.toast("已定位到 source file")
                    except Exception:
                        st.success("已定位到 source file")
                else:
                    st.warning("定位 source file 失败。")

        st.text_input("Deep-Link", value=deep_link_query, key=f"catalog_ui::deeplink::{entry_key}")
        st.markdown("<div class='catalog-section-title'>Source File</div>", unsafe_allow_html=True)
        if source_path is not None:
            st.code(str(source_path), language=None)
        else:
            st.caption("当前条目暂时无法解析到本地 source file。")

    st.caption(f"kind={str(entry.get('kind', '') or 'unknown')} | scope={scope} | active-view={kind}")
    st.markdown("</div>", unsafe_allow_html=True)
def run_dashboard(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    st = _require_streamlit()
    _set_page_config(st)
    _inject_style(st)

    query_params, query_filters = _read_query_params(st)
    st.session_state.setdefault("catalog_ui_profile", str(query_params.get("profile", args.profile or "framework-core")))
    st.session_state.setdefault("catalog_ui_scope", str(query_params.get("scope", args.scope or "framework")))
    st.session_state.setdefault("catalog_ui_kind", str(query_params.get("kind", args.kind or "preset")))
    st.session_state.setdefault("catalog_ui_query", str(query_params.get("query", args.query or "")))
    st.session_state.setdefault("catalog_ui_selected", str(query_params.get("selected", "")))
    st.session_state.setdefault("catalog_ui_project_path", str(query_params.get("project_path", args.project_path or "")))
    st.session_state.setdefault(
        "catalog_ui_include_global",
        str(query_params.get("include_global", "1" if args.include_global else "")).strip().lower() in {"1", "true", "yes", "on"},
    )
    st.session_state.setdefault("catalog_ui_db_path", str(query_params.get("db_path", args.db_path or "")))
    st.session_state.setdefault("catalog_ui_source_mode", str(query_params.get("source_mode", args.source_mode or "")))
    st.session_state.setdefault(_NAV_STACK_KEY, [])
    if str(query_params.get("nav_action", "") or "").strip().lower() == _NAV_ACTION_LOCATE_SELECTED:
        st.session_state[_PENDING_LOCATE_KEY] = True

    current_profile = str(st.session_state["catalog_ui_profile"] or "framework-core")
    current_scope = str(st.session_state["catalog_ui_scope"] or "framework")
    current_project_path = str(st.session_state["catalog_ui_project_path"] or "")
    current_include_global = bool(st.session_state["catalog_ui_include_global"])
    current_db_path = str(st.session_state["catalog_ui_db_path"] or "")
    current_source_mode = str(st.session_state["catalog_ui_source_mode"] or "")

    source_info = _cached_source_info(
        current_profile,
        current_scope,
        current_project_path,
        current_include_global,
        current_db_path,
        current_source_mode,
    )
    summary = _cached_summary(
        current_profile,
        current_scope,
        current_project_path,
        current_include_global,
        current_db_path,
        current_source_mode,
    )
    current_kind = _normalize_kind(str(st.session_state["catalog_ui_kind"] or "preset"))
    schema = _cached_schema(
        current_profile,
        current_scope,
        current_project_path,
        current_include_global,
        current_kind,
        current_db_path,
        current_source_mode,
    )
    kinds = tuple(str(kind) for kind in schema.get("kinds", ()) if str(kind).strip()) or _KIND_ORDER
    if current_kind not in kinds:
        current_kind = str(kinds[0])
    st.session_state["catalog_ui_kind"] = current_kind
    facet_fields = _facet_fields(current_kind, schema)
    _shared.sync_query_filters_to_session(
        st,
        scope=current_scope,
        kind=current_kind,
        facet_fields=facet_fields,
        query_filters=query_filters,
        multi_value=False,
    )
    sort_by_key = _view_state_key(current_scope, current_kind, "sort_by")
    sort_dir_key = _view_state_key(current_scope, current_kind, "sort_dir")
    detail_tab_key = _view_state_key(current_scope, current_kind, "detail_tab")
    open_relations_key = _view_state_key(current_scope, current_kind, "open_relations")
    column_mode_key = _view_state_key(current_scope, current_kind, "column_mode")
    page_size_key = _view_state_key(current_scope, current_kind, "page_size")
    results_collapse_key = _view_state_key(current_scope, current_kind, "results_collapse")
    st.session_state.setdefault(column_mode_key, _normalize_column_mode(args.column_mode))
    st.session_state.setdefault(page_size_key, _normalize_page_size(args.page_size))
    st.session_state.setdefault(results_collapse_key, _normalize_results_collapse(args.results_collapse))
    if bool(st.session_state.get(_PENDING_LOCATE_KEY)):
        st.session_state[results_collapse_key] = "expanded"
    sort_by_value = _normalize_sort_by(query_params.get("sort_by", st.session_state.get(sort_by_key, _shared.DEFAULT_SORT_BY)))
    sort_dir_value = _normalize_sort_dir(query_params.get("sort_dir", st.session_state.get(sort_dir_key, _shared.DEFAULT_SORT_DIR)))
    detail_tab_value = _normalize_detail_tab(query_params.get("detail_tab", st.session_state.get(detail_tab_key, _shared.DEFAULT_DETAIL_TAB)))
    open_relation_values = (
        _shared.normalize_csv_values(query_params.get("open_relations", st.session_state.get(open_relations_key, ())))
        if "open_relations" in query_params or open_relations_key in st.session_state
        else ()
    )
    column_mode_value = _normalize_column_mode(query_params.get("column_mode", st.session_state.get(column_mode_key, _shared.DEFAULT_COLUMN_MODE)))
    page_size_value = _normalize_page_size(query_params.get("page_size", st.session_state.get(page_size_key, _shared.DEFAULT_PAGE_SIZE)))
    results_collapse_value = _normalize_results_collapse(
        query_params.get("results_collapse", st.session_state.get(results_collapse_key, _shared.DEFAULT_RESULTS_COLLAPSE))
    )
    if bool(st.session_state.get(_PENDING_LOCATE_KEY)):
        results_collapse_value = "expanded"
    if st.session_state.get(sort_by_key) != sort_by_value:
        st.session_state[sort_by_key] = sort_by_value
    if st.session_state.get(sort_dir_key) != sort_dir_value:
        st.session_state[sort_dir_key] = sort_dir_value
    if st.session_state.get(detail_tab_key) != detail_tab_value:
        st.session_state[detail_tab_key] = detail_tab_value
    if tuple(st.session_state.get(open_relations_key, ())) != tuple(open_relation_values):
        st.session_state[open_relations_key] = tuple(open_relation_values)
    if st.session_state.get(column_mode_key) != column_mode_value:
        st.session_state[column_mode_key] = column_mode_value
    if _normalize_page_size(st.session_state.get(page_size_key, _shared.DEFAULT_PAGE_SIZE)) != page_size_value:
        st.session_state[page_size_key] = page_size_value
    if st.session_state.get(results_collapse_key) != results_collapse_value:
        st.session_state[results_collapse_key] = results_collapse_value

    _page.render_top_anchor(st)
    _page.render_hero(
        st,
        _page.HeroSpec(
            icon_text="MC",
            kicker="mlblack catalog",
            title="字段型 Catalog 查询页",
            subtitle=(
                "顶部集中控制查询与字段筛选，中间是可点击结果表格，右侧展示详情、来源、跳转栈与 source file 动作。"
                "页面同时支持 framework / project 双视图，以及 registry / DB 驱动的目录读取。"
            ),
        ),
    )
    root_note = str(source_info.get("project_root", "") or "")
    _page.render_stat_cards(
        st,
        (
            _page.StatCardSpec(
                label="Scope",
                value=current_scope,
                note=str(source_info.get("effective_source", "registry")),
            ),
            _page.StatCardSpec(
                label="Entries",
                value=str(int(summary.get("total", 0))),
                note="当前视图下的条目总数",
            ),
            _page.StatCardSpec(
                label="Kinds",
                value=str(len(summary.get("by_kind", {}))),
                note=" / ".join(sorted(summary.get("by_kind", {}).keys())[:5]),
            ),
            _page.StatCardSpec(
                label="Project Root",
                value="已连接" if bool(root_note) else "未连接",
                note=root_note or "当前不是 mlblack scaffold 目录",
            ),
        ),
    )

    _page.render_section_header(
        st,
        _page.SectionHeaderSpec(
            section_id=_page.FILTER_SECTION_ID,
            label="Filter",
            title="查询与筛选",
            subtitle="先确定视图、profile、分类与数据源，再用字段筛选把结果收窄到可点选范围。",
        ),
    )

    primary_controls = _page.render_control_row(st, _primary_controls_spec())
    scope = primary_controls["scope"].column.radio(
        primary_controls["scope"].label,
        options=("framework", "project"),
        index=0 if current_scope == "framework" else 1,
        horizontal=True,
    )
    profile = primary_controls["profile"].column.selectbox(
        primary_controls["profile"].label,
        options=("framework-core", "default"),
        index=0 if current_profile == "framework-core" else 1,
    )
    kind = primary_controls["kind"].column.selectbox(
        primary_controls["kind"].label,
        options=list(kinds),
        index=max(0, list(kinds).index(current_kind)) if current_kind in kinds else 0,
        format_func=_kind_label,
    )
    query = primary_controls["query"].column.text_input(
        primary_controls["query"].label,
        value=str(st.session_state["catalog_ui_query"] or ""),
        placeholder=primary_controls["query"].placeholder,
    )

    secondary_controls = _page.render_control_row(st, _secondary_controls_spec())
    project_path = secondary_controls["project_path"].column.text_input(
        secondary_controls["project_path"].label,
        value=current_project_path,
        disabled=scope != "project",
        help=secondary_controls["project_path"].help,
    )
    include_global = secondary_controls["include_global"].column.checkbox(
        secondary_controls["include_global"].label,
        value=current_include_global,
        disabled=scope != "project",
    )
    db_controls_disabled = scope == "project" and not include_global
    db_path = secondary_controls["db_path"].column.text_input(
        secondary_controls["db_path"].label,
        value=current_db_path,
        disabled=db_controls_disabled,
        placeholder=secondary_controls["db_path"].placeholder,
        help=secondary_controls["db_path"].help,
    )
    source_mode = secondary_controls["source_mode"].column.selectbox(
        secondary_controls["source_mode"].label,
        options=("", "prefer", "only", "off"),
        index=("", "prefer", "only", "off").index(current_source_mode) if current_source_mode in ("", "prefer", "only", "off") else 0,
        disabled=db_controls_disabled,
        help=secondary_controls["source_mode"].help,
    )
    if db_controls_disabled:
        secondary_controls["db_path"].caption("当前项目视图只看本地 catalog，未并入框架条目时不读取数据库。")
    else:
        secondary_controls["project_path"].caption("项目视图可以接本地 scaffold；framework 视图也可以直接切 DB-backed catalog。")
        secondary_controls["db_path"].caption("留空时优先读取本地 catalog DB 配置；填写后会直接用这个 DB URL。")

    state_changed = False
    updates = {
        "catalog_ui_scope": scope,
        "catalog_ui_profile": profile,
        "catalog_ui_kind": kind,
        "catalog_ui_query": query,
        "catalog_ui_project_path": project_path,
        "catalog_ui_include_global": include_global,
        "catalog_ui_db_path": db_path,
        "catalog_ui_source_mode": source_mode,
    }
    for state_key, value in updates.items():
        if st.session_state.get(state_key) != value:
            st.session_state[state_key] = value
            state_changed = True
    if state_changed:
        _rerun(st)

    if current_scope == "project" and not bool(source_info.get("project_found", False)):
        st.warning("当前没有发现 .mlblack-project 或 catalog/entries.toml。请提供 Project Path。")

    current_query = str(st.session_state["catalog_ui_query"] or "")
    current_filters = {
        str(name): str(value)
        for name, value in _shared.collect_session_filters(
            st,
            scope=current_scope,
            kind=current_kind,
            facet_fields=facet_fields,
            multi_value=False,
        ).items()
    }

    facet_payload = _cached_facets(
        current_profile,
        current_scope,
        current_project_path,
        current_include_global,
        current_kind,
        current_query,
        _freeze_filters(current_filters),
        current_db_path,
        current_source_mode,
    )

    filter_fields_with_rows: list[tuple[str, list[dict[str, Any]]]] = []
    for field_name in facet_fields:
        rows = list(facet_payload.get("facets", {}).get(field_name, []))
        if rows:
            filter_fields_with_rows.append((field_name, rows))
    with st.container():
        st.markdown("<div class='catalog-inline-filters'>", unsafe_allow_html=True)
        expander_title = f"字段筛选 · {_kind_label(current_kind)}"
        with st.expander(expander_title, expanded=False):
            if not filter_fields_with_rows:
                st.caption("当前分类下没有可用的字段筛选项。")
            else:
                filter_cols = st.columns(min(4, len(filter_fields_with_rows)))
                for index, (field_name, rows) in enumerate(filter_fields_with_rows):
                    options = [str(row.get("value", "")) for row in rows if str(row.get("value", "")).strip()]
                    if not options:
                        continue
                    state_key = f"catalog_ui::facet::{current_scope}::{current_kind}::{field_name}"
                    current_value = str(st.session_state.get(state_key, "") or "")
                    values = [""] + options
                    labels_by_value = {"": "不限"}
                    for row in rows:
                        value = str(row.get("value", "") or "").strip()
                        if not value:
                            continue
                        labels_by_value[value] = f"{value} ({int(row.get('count', 0))})"
                    if current_value and current_value not in values:
                        values.append(current_value)
                        labels_by_value[current_value] = f"{current_value}（当前）"
                    with filter_cols[index % len(filter_cols)]:
                        st.selectbox(
                            _field_label(field_name),
                            options=values,
                            index=values.index(current_value) if current_value in values else 0,
                            format_func=lambda value, labels_by_value=labels_by_value: labels_by_value.get(str(value), str(value)),
                            key=state_key,
                        )
                st.button(
                    "清空字段筛选",
                    use_container_width=False,
                    on_click=_callback_clear_filters,
                    kwargs={"st": st, "scope": current_scope, "kind": current_kind, "facet_fields": facet_fields},
                )
        st.markdown("</div>", unsafe_allow_html=True)

    field_filters = {
        str(name): str(value)
        for name, value in _shared.collect_session_filters(
            st,
            scope=current_scope,
            kind=current_kind,
            facet_fields=facet_fields,
            multi_value=False,
        ).items()
    }

    sort_cols = st.columns((0.9, 0.78, 1.52))
    current_sort_by = _normalize_sort_by(st.session_state.get(sort_by_key, _shared.DEFAULT_SORT_BY))
    current_sort_dir = _normalize_sort_dir(st.session_state.get(sort_dir_key, _shared.DEFAULT_SORT_DIR))
    sort_by = sort_cols[0].selectbox(
        "结果排序",
        options=list(_SORT_OPTIONS),
        index=list(_SORT_OPTIONS).index(current_sort_by),
        format_func=_sort_label,
    )
    sort_dir = sort_cols[1].radio(
        "方向",
        options=("asc", "desc"),
        index=0 if current_sort_dir == "asc" else 1,
        horizontal=True,
        format_func=lambda value: "升序" if value == "asc" else "降序",
    )
    sort_cols[2].caption("Deep-link 会记住当前排序方式。")
    st.session_state[sort_by_key] = sort_by
    st.session_state[sort_dir_key] = sort_dir

    display_cols = st.columns((0.9, 0.8, 0.95, 1.45))
    current_column_mode = _normalize_column_mode(st.session_state.get(column_mode_key, _shared.DEFAULT_COLUMN_MODE))
    current_page_size = _normalize_page_size(st.session_state.get(page_size_key, _shared.DEFAULT_PAGE_SIZE))
    current_results_collapse = _normalize_results_collapse(st.session_state.get(results_collapse_key, _shared.DEFAULT_RESULTS_COLLAPSE))
    column_mode = display_cols[0].selectbox(
        "列显示方案",
        options=list(_COLUMN_MODE_OPTIONS),
        index=list(_COLUMN_MODE_OPTIONS).index(current_column_mode),
        format_func=_column_mode_label,
    )
    page_size_options = sorted({* _PAGE_SIZE_OPTIONS, current_page_size})
    page_size = int(
        display_cols[1].selectbox(
            "Page Size",
            options=page_size_options,
            index=page_size_options.index(current_page_size),
            format_func=lambda value: f"{int(value)} 条",
        )
    )
    results_collapse = display_cols[2].radio(
        "结果折叠",
        options=list(_RESULTS_COLLAPSE_OPTIONS),
        index=list(_RESULTS_COLLAPSE_OPTIONS).index(current_results_collapse),
        horizontal=True,
        format_func=_results_collapse_label,
    )
    display_cols[3].caption("Deep-link 会记住当前列方案、结果分页窗口和折叠状态。")
    st.session_state[column_mode_key] = column_mode
    st.session_state[page_size_key] = page_size
    st.session_state[results_collapse_key] = results_collapse

    item_payloads = list(
        _cached_items(
            current_profile,
            current_scope,
            current_project_path,
            current_include_global,
            current_kind,
            current_query,
            _freeze_filters(field_filters),
            current_db_path,
            current_source_mode,
        )
    )
    item_payloads = _sorted_items(item_payloads, sort_by=sort_by, sort_dir=sort_dir)
    visible_item_payloads = _visible_result_items(item_payloads, page_size=page_size)

    selected_key = str(st.session_state.get("catalog_ui_selected", "") or "").strip()
    selected_entry = (
        _cached_selected(
            current_profile,
            current_scope,
            current_project_path,
            current_include_global,
            selected_key,
            current_db_path,
            current_source_mode,
        )
        if selected_key
        else None
    )
    selection = _selection_state(selected_key, visible_item_payloads, selected_exists=selected_entry is not None)
    if not selection["selected_key"] and visible_item_payloads and selected_key != _NO_SELECTION:
        selected_key = str(visible_item_payloads[0].get("key", "") or "")
        selected_entry = _cached_selected(
            current_profile,
            current_scope,
            current_project_path,
            current_include_global,
            selected_key,
            current_db_path,
            current_source_mode,
        )
        selection = _selection_state(selected_key, visible_item_payloads, selected_exists=selected_entry is not None)

    if bool(st.session_state.get(_PENDING_LOCATE_KEY)):
        selected_in_items, selected_in_visible = _selection_presence(
            selected_key,
            items=item_payloads,
            visible_items=visible_item_payloads,
        )
        if not selected_key or selected_key == _NO_SELECTION or selected_entry is None:
            st.session_state[_PENDING_LOCATE_KEY] = False
        else:
            selected_kind = _normalize_kind(str(selected_entry.get("kind", "") or ""))
            if current_kind != selected_kind and selected_kind in kinds:
                st.session_state["catalog_ui_kind"] = selected_kind
                _write_locate_state_and_rerun(
                    st,
                    profile=current_profile,
                    scope=current_scope,
                    kind=selected_kind,
                    query=str(st.session_state.get("catalog_ui_query", "") or ""),
                    selected=selected_key,
                    project_path=current_project_path,
                    include_global=current_include_global,
                    db_path=current_db_path,
                    source_mode=current_source_mode,
                    sort_by=sort_by,
                    sort_dir=sort_dir,
                    detail_tab=_normalize_detail_tab(st.session_state.get(detail_tab_key, _shared.DEFAULT_DETAIL_TAB)),
                    open_relations=_shared.csv_param_value(st.session_state.get(open_relations_key, ())),
                    column_mode=column_mode,
                    page_size=page_size,
                    results_collapse="expanded",
                    field_filters=field_filters,
                )
            if not selected_in_items:
                next_query = str(st.session_state.get("catalog_ui_query", "") or "")
                next_filters: dict[str, object] = dict(field_filters)
                changed = False
                if next_query.strip():
                    st.session_state["catalog_ui_query"] = ""
                    next_query = ""
                    changed = True
                if _has_active_field_filters(field_filters):
                    _clear_scope_kind_filters(st, scope=current_scope, kind=current_kind, facet_fields=facet_fields)
                    next_filters = {}
                    changed = True
                if changed:
                    _write_locate_state_and_rerun(
                        st,
                        profile=current_profile,
                        scope=current_scope,
                        kind=current_kind,
                        query=next_query,
                        selected=selected_key,
                        project_path=current_project_path,
                        include_global=current_include_global,
                        db_path=current_db_path,
                        source_mode=current_source_mode,
                        sort_by=sort_by,
                        sort_dir=sort_dir,
                        detail_tab=_normalize_detail_tab(st.session_state.get(detail_tab_key, _shared.DEFAULT_DETAIL_TAB)),
                        open_relations=_shared.csv_param_value(st.session_state.get(open_relations_key, ())),
                        column_mode=column_mode,
                        page_size=page_size,
                        results_collapse="expanded",
                        field_filters=next_filters,
                    )
            if selected_in_items and not selected_in_visible:
                desired_page_size = max(_normalize_page_size(st.session_state.get(page_size_key, page_size)), len(item_payloads))
                if _normalize_page_size(st.session_state.get(page_size_key, page_size)) < desired_page_size:
                    st.session_state[page_size_key] = desired_page_size
                    _write_locate_state_and_rerun(
                        st,
                        profile=current_profile,
                        scope=current_scope,
                        kind=current_kind,
                        query=str(st.session_state.get("catalog_ui_query", "") or ""),
                        selected=selected_key,
                        project_path=current_project_path,
                        include_global=current_include_global,
                        db_path=current_db_path,
                        source_mode=current_source_mode,
                        sort_by=sort_by,
                        sort_dir=sort_dir,
                        detail_tab=_normalize_detail_tab(st.session_state.get(detail_tab_key, _shared.DEFAULT_DETAIL_TAB)),
                        open_relations=_shared.csv_param_value(st.session_state.get(open_relations_key, ())),
                        column_mode=column_mode,
                        page_size=desired_page_size,
                        results_collapse="expanded",
                        field_filters=field_filters,
                    )
            if selected_in_items:
                st.session_state[_PENDING_LOCATE_KEY] = False
                st.session_state[_PENDING_SCROLL_TARGET_KEY] = "catalog-results-anchor"

    left, right = st.columns((1.28, 0.92), gap="large")
    with left:
        _render_selection_float(
            st,
            selection=selection,
            selected_entry=selected_entry,
            visible_items=visible_item_payloads,
            scope=current_scope,
            kind=current_kind,
            facet_fields=facet_fields,
        )
        if selection.get("hidden", False):
            st.markdown(
                "<div class='catalog-warning'>当前选中项仍保留在右侧详情里，但它已经不在中间结果表格中。你可以点“显示它”清空搜索与字段筛选，让它重新出现。</div>",
                unsafe_allow_html=True,
            )
        _page.render_section_header(
            st,
            _page.SectionHeaderSpec(
                section_id=_page.RESULT_SECTION_ID,
                label="Results",
                title=f"结果表格 · {_kind_label(current_kind)}",
                subtitle="当前结果支持单击切换选中项，deep-link 会记住排序、分页窗口与折叠状态。",
                note=f"{len(visible_item_payloads)} / {len(item_payloads)}",
            ),
        )
        st.markdown("<div id='catalog-results-anchor'></div>", unsafe_allow_html=True)
        results_title = f"结果表格 · {_kind_label(current_kind)}"
        results_label = f"{results_title}（显示 {len(visible_item_payloads)} / {len(item_payloads)}）"
        with st.expander(results_label, expanded=results_collapse == "expanded"):
            if len(item_payloads) > len(visible_item_payloads):
                st.caption(f"当前命中 {len(item_payloads)} 条，当前展示前 {len(visible_item_payloads)} 条。")
            else:
                st.caption(f"当前命中 {len(item_payloads)} 条。")
            selected_key = _render_results_table(
                st,
                items=visible_item_payloads,
                scope=current_scope,
                kind=current_kind,
                facet_fields=facet_fields,
                column_mode=column_mode,
            )
        if selected_key and selected_key != _NO_SELECTION:
            st.caption(f"当前选中：{selected_key}")
        pending_scroll_target = str(st.session_state.get(_PENDING_SCROLL_TARGET_KEY, "") or "").strip()
        if pending_scroll_target:
            _scroll_to_anchor(st, anchor_id=pending_scroll_target)
            st.session_state[_PENDING_SCROLL_TARGET_KEY] = ""
    st.session_state["catalog_ui_selected"] = selected_key if selected_key else (_NO_SELECTION if selected_key == _NO_SELECTION else "")
    if selected_key and selected_key != _NO_SELECTION:
        if selected_entry is None or str(selected_entry.get("key", "") or "") != selected_key:
            selected_entry = _cached_selected(
                current_profile,
                current_scope,
                current_project_path,
                current_include_global,
                selected_key,
                current_db_path,
                current_source_mode,
            )
    else:
        selected_entry = None
    selection = _selection_state(selected_key, visible_item_payloads, selected_exists=selected_entry is not None)
    floating_locate_target = "catalog-results-anchor" if selection.get("visible", False) else None
    floating_locate_tooltip = (
        "定位当前选中项"
        if selection.get("visible", False)
        else ("当前选中项已被筛选隐藏，请先显示它" if selection.get("hidden", False) else "当前没有选中项")
    )
    _render_floating_nav(
        st,
        locate_target=floating_locate_target,
        locate_tooltip=floating_locate_tooltip,
    )

    with right:
        _page.render_section_header(
            st,
            _page.SectionHeaderSpec(
                section_id=_page.DETAIL_SECTION_ID,
                label="Detail",
                title="详情与跳转",
                subtitle="右侧统一承接条目详情、关系跳转、来源信息与 deep-link 回位。",
            ),
        )
        neighbors = _cached_neighbors(
            current_profile,
            current_scope,
            current_project_path,
            current_include_global,
            selected_key,
            current_db_path,
            current_source_mode,
        ) if selected_key else None
        neighbor_groups = _relation_neighbor_groups(entry=selected_entry, neighbors=neighbors)
        relation_options = tuple(
            str(name)
            for name, rows in neighbor_groups.items()
            if rows
        )
        current_detail_tab = _normalize_detail_tab(st.session_state.get(detail_tab_key, _shared.DEFAULT_DETAIL_TAB))
        detail_cols = st.columns((1.0, 1.3))
        detail_tab = detail_cols[0].radio(
            "详情页签",
            options=list(_DETAIL_TABS),
            index=list(_DETAIL_TABS).index(current_detail_tab),
            horizontal=True,
            format_func=_detail_tab_label,
        )
        st.session_state[detail_tab_key] = detail_tab
        expanded_relation_groups = tuple(
            value
            for value in _shared.normalize_csv_values(st.session_state.get(open_relations_key, ()))
            if value in relation_options
        )
        if detail_tab == "relations" and relation_options:
            chosen_groups = detail_cols[1].multiselect(
                "展开关系组",
                options=list(relation_options),
                default=list(expanded_relation_groups),
                format_func=_field_label,
            )
            expanded_relation_groups = tuple(str(value).strip() for value in chosen_groups if str(value).strip())
        elif detail_tab == "relations":
            detail_cols[1].caption("当前条目没有可展开的关系分组。")
            expanded_relation_groups = ()
        else:
            detail_cols[1].caption("Deep-link 会记住当前详情页签与展开状态。")
        st.session_state[open_relations_key] = expanded_relation_groups
        deep_link_query = _build_deep_link_query(
            profile=current_profile,
            scope=current_scope,
            kind=current_kind,
            query=current_query,
            selected="" if selected_key == _NO_SELECTION else selected_key,
            project_path=current_project_path,
            include_global=current_include_global,
            db_path=current_db_path,
            source_mode=current_source_mode,
            sort_by=sort_by,
            sort_dir=sort_dir,
            detail_tab=detail_tab,
            open_relations=_shared.csv_param_value(expanded_relation_groups),
            column_mode=column_mode,
            page_size=page_size,
            results_collapse=results_collapse,
            field_filters=field_filters,
        )
        _write_query_params(
            st,
            profile=current_profile,
            scope=current_scope,
            kind=current_kind,
            query=current_query,
            selected="" if selected_key == _NO_SELECTION else selected_key,
            project_path=current_project_path,
            include_global=current_include_global,
            db_path=current_db_path,
            source_mode=current_source_mode,
            sort_by=sort_by,
            sort_dir=sort_dir,
            detail_tab=detail_tab,
            open_relations=_shared.csv_param_value(expanded_relation_groups),
            column_mode=column_mode,
            page_size=page_size,
            results_collapse=results_collapse,
            field_filters=field_filters,
        )
        _render_detail(
            st,
            entry=selected_entry,
            neighbors=neighbors,
            source_info=source_info,
            deep_link_query=deep_link_query,
            scope=current_scope,
            kind=current_kind,
            detail_tab=detail_tab,
            expanded_relation_groups=expanded_relation_groups,
        )


def main(argv: Sequence[str] | None = None) -> None:
    run_dashboard(argv)


if __name__ == "__main__":  # pragma: no cover
    main(sys.argv[1:])




