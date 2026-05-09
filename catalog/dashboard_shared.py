from __future__ import annotations

from typing import Any, Mapping, Sequence
from urllib.parse import urlencode

NO_SELECTION = "__none__"
NAV_STACK_KEY = "catalog_ui_navigation_stack"
FILTER_QUERY_PREFIX = "f_"
DEFAULT_DETAIL_TAB = "overview"
DEFAULT_SORT_BY = "default"
DEFAULT_SORT_DIR = "asc"
DEFAULT_COLUMN_MODE = "standard"
DEFAULT_PAGE_SIZE = 50
DEFAULT_RESULTS_COLLAPSE = "expanded"


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


def normalize_csv_values(value: object) -> tuple[str, ...]:
    return _normalize_filter_values(value)


def csv_param_value(value: object) -> str:
    return ",".join(normalize_csv_values(value))


def view_state_key(scope: str, kind: str, name: str) -> str:
    return f"catalog_ui::view::{scope}::{kind}::{name}"


def read_query_params(
    st: Any,
    *,
    base_keys: Sequence[str],
    filter_prefix: str = FILTER_QUERY_PREFIX,
) -> tuple[dict[str, str], dict[str, tuple[str, ...]]]:
    def _coerce_params(raw: Mapping[str, Any]) -> tuple[dict[str, str], dict[str, tuple[str, ...]]]:
        base: dict[str, str] = {}
        filters: dict[str, tuple[str, ...]] = {}
        for key, raw_value in raw.items():
            value = raw_value[-1] if isinstance(raw_value, list) else raw_value
            text = str(value or "").strip()
            if not text:
                continue
            if key in base_keys:
                base[str(key)] = text
                continue
            if str(key).startswith(filter_prefix):
                field_name = str(key)[len(filter_prefix) :].strip()
                if not field_name:
                    continue
                values = _normalize_filter_values(text)
                if values:
                    filters[field_name] = values
        return base, filters

    try:
        params = st.query_params
        return _coerce_params({str(key): params.get(key) for key in list(params.keys())})
    except Exception:
        try:
            return _coerce_params(st.experimental_get_query_params())
        except Exception:
            return {}, {}


def build_query_param_payload(
    *,
    base_params: Mapping[str, object],
    field_filters: Mapping[str, object] | None = None,
    none_sentinel: str = NO_SELECTION,
    filter_prefix: str = FILTER_QUERY_PREFIX,
) -> dict[str, str]:
    payload: dict[str, str] = {}
    for key, raw_value in base_params.items():
        if str(key) == "selected" and str(raw_value or "").strip() == none_sentinel:
            text = ""
        else:
            text = str(raw_value or "").strip()
        if text:
            payload[str(key)] = text
    for field_name, raw_value in dict(field_filters or {}).items():
        values = _normalize_filter_values(raw_value)
        if not values:
            continue
        payload[f"{filter_prefix}{str(field_name).strip()}"] = ",".join(values)
    return payload


def build_deep_link_query(
    *,
    base_params: Mapping[str, object],
    field_filters: Mapping[str, object] | None = None,
    none_sentinel: str = NO_SELECTION,
    filter_prefix: str = FILTER_QUERY_PREFIX,
) -> str:
    payload = build_query_param_payload(
        base_params=base_params,
        field_filters=field_filters,
        none_sentinel=none_sentinel,
        filter_prefix=filter_prefix,
    )
    return "?" + urlencode(payload)


def write_query_params(
    st: Any,
    *,
    base_params: Mapping[str, object],
    field_filters: Mapping[str, object] | None = None,
    none_sentinel: str = NO_SELECTION,
    filter_prefix: str = FILTER_QUERY_PREFIX,
) -> None:
    payload = build_query_param_payload(
        base_params=base_params,
        field_filters=field_filters,
        none_sentinel=none_sentinel,
        filter_prefix=filter_prefix,
    )
    try:
        params = st.query_params
        params.clear()
        for key, value in payload.items():
            params[str(key)] = str(value)
        return
    except Exception:
        pass
    try:
        st.experimental_set_query_params(**payload)
    except Exception:
        pass


def rerun(st: Any) -> None:
    try:
        st.rerun()
        return
    except Exception:
        pass
    try:
        st.experimental_rerun()
    except Exception:
        return


def normalize_navigation_stack(values: object) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        return out
    for raw in values:
        if not isinstance(raw, Mapping):
            continue
        key = str(raw.get("key", "") or "").strip()
        if not key:
            continue
        out.append(
            {
                "key": key,
                "title": str(raw.get("title", "") or key),
                "kind": str(raw.get("kind", "") or ""),
            }
        )
    return out


def navigation_stack(st: Any, *, state_key: str = NAV_STACK_KEY) -> list[dict[str, str]]:
    stack = normalize_navigation_stack(st.session_state.get(state_key, ()))
    st.session_state[state_key] = stack
    return stack


def push_navigation_stack(
    st: Any,
    *,
    current_entry: Mapping[str, Any] | None,
    state_key: str = NAV_STACK_KEY,
) -> None:
    if not current_entry:
        return
    key = str(current_entry.get("key", "") or "").strip()
    if not key:
        return
    stack = navigation_stack(st, state_key=state_key)
    if stack and stack[-1].get("key") == key:
        return
    stack.append(
        {
            "key": key,
            "title": str(current_entry.get("title", "") or key),
            "kind": str(current_entry.get("kind", "") or ""),
        }
    )
    st.session_state[state_key] = stack[-12:]


def pop_navigation_stack(st: Any, *, state_key: str = NAV_STACK_KEY) -> dict[str, str] | None:
    stack = navigation_stack(st, state_key=state_key)
    if not stack:
        return None
    target = dict(stack[-1])
    st.session_state[state_key] = stack[:-1]
    return target


def restore_navigation_index(
    st: Any,
    index: int,
    *,
    state_key: str = NAV_STACK_KEY,
) -> dict[str, str] | None:
    stack = navigation_stack(st, state_key=state_key)
    if index < 0 or index >= len(stack):
        return None
    target = dict(stack[index])
    st.session_state[state_key] = stack[:index]
    return target


def selection_state(
    selected_key: str,
    items: Sequence[Mapping[str, Any]],
    *,
    selected_exists: bool,
    none_sentinel: str = NO_SELECTION,
) -> dict[str, Any]:
    key = str(selected_key or "").strip()
    visible_keys = [str(item.get("key", "")) for item in items]
    if not key or key == none_sentinel:
        return {"selected_key": "", "visible": False, "hidden": False, "row_index": None}
    if key in visible_keys:
        return {
            "selected_key": key,
            "visible": True,
            "hidden": False,
            "row_index": visible_keys.index(key),
        }
    return {
        "selected_key": key,
        "visible": False,
        "hidden": bool(selected_exists),
        "row_index": None,
    }


def facet_state_key(scope: str, kind: str, field_name: str) -> str:
    return f"catalog_ui::facet::{scope}::{kind}::{field_name}"


def clear_scope_kind_filters(
    st: Any,
    *,
    scope: str,
    kind: str,
    facet_fields: Sequence[str],
) -> None:
    for field_name in facet_fields:
        st.session_state.pop(facet_state_key(scope, kind, str(field_name)), None)


def sync_query_filters_to_session(
    st: Any,
    *,
    scope: str,
    kind: str,
    facet_fields: Sequence[str],
    query_filters: Mapping[str, Sequence[str]],
    multi_value: bool,
) -> None:
    for field_name in facet_fields:
        state_key = facet_state_key(scope, kind, str(field_name))
        values = tuple(str(value).strip() for value in query_filters.get(str(field_name), ()) if str(value).strip())
        if multi_value:
            if str(field_name) in query_filters:
                st.session_state[state_key] = values
            else:
                st.session_state.setdefault(state_key, values)
        else:
            value = values[0] if values else ""
            if str(field_name) in query_filters:
                st.session_state[state_key] = value
            else:
                st.session_state.setdefault(state_key, value)


def collect_session_filters(
    st: Any,
    *,
    scope: str,
    kind: str,
    facet_fields: Sequence[str],
    multi_value: bool,
) -> dict[str, object]:
    out: dict[str, object] = {}
    for field_name in facet_fields:
        value = st.session_state.get(facet_state_key(scope, kind, str(field_name)))
        if multi_value:
            values = tuple(str(item).strip() for item in value or () if str(item).strip())
            if values:
                out[str(field_name)] = values
        else:
            text = str(value or "").strip()
            if text:
                out[str(field_name)] = text
    return out
