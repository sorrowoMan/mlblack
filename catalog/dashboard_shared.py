from __future__ import annotations

from typing import Any, Mapping, Sequence
from urllib.parse import urlencode

NO_SELECTION = "__none__"
FILTER_QUERY_PREFIX = "f_"
DEFAULT_SOURCE = "db"
DEFAULT_PROFILE = "default"
DEFAULT_KIND = "all"
DEFAULT_PAGE_SIZE = 50


def normalize_csv_values(value: object) -> tuple[str, ...]:
    if value in (None, ""):
        return tuple()
    if isinstance(value, str):
        return tuple(part.strip() for part in value.split(",") if part.strip())
    if isinstance(value, Mapping):
        return tuple(str(key).strip() for key in value.keys() if str(key).strip())
    if isinstance(value, (list, tuple, set, frozenset)):
        out: list[str] = []
        for item in value:
            out.extend(normalize_csv_values(item))
        return tuple(out)
    text = str(value).strip()
    return (text,) if text else tuple()


def read_query_params(
    st: Any,
    *,
    base_keys: Sequence[str],
    filter_prefix: str = FILTER_QUERY_PREFIX,
) -> tuple[dict[str, str], dict[str, tuple[str, ...]]]:
    def coerce(raw: Mapping[str, Any]) -> tuple[dict[str, str], dict[str, tuple[str, ...]]]:
        base: dict[str, str] = {}
        filters: dict[str, tuple[str, ...]] = {}
        for key, raw_value in raw.items():
            value = raw_value[-1] if isinstance(raw_value, list) else raw_value
            text = str(value or "").strip()
            if not text:
                continue
            if key in base_keys:
                base[str(key)] = text
            elif str(key).startswith(filter_prefix):
                field_name = str(key)[len(filter_prefix):].strip()
                values = normalize_csv_values(text)
                if field_name and values:
                    filters[field_name] = values
        return base, filters

    try:
        params = st.query_params
        return coerce({str(key): params.get(key) for key in list(params.keys())})
    except Exception:
        try:
            return coerce(st.experimental_get_query_params())
        except Exception:
            return {}, {}


def build_query_param_payload(
    *,
    base_params: Mapping[str, object],
    field_filters: Mapping[str, object] | None = None,
    filter_prefix: str = FILTER_QUERY_PREFIX,
) -> dict[str, str]:
    payload: dict[str, str] = {}
    for key, raw_value in base_params.items():
        text = str(raw_value or "").strip()
        if text:
            payload[str(key)] = text
    for field_name, raw_value in dict(field_filters or {}).items():
        values = normalize_csv_values(raw_value)
        if values:
            payload[f"{filter_prefix}{str(field_name).strip()}"] = ",".join(values)
    return payload


def build_deep_link_query(
    *,
    base_params: Mapping[str, object],
    field_filters: Mapping[str, object] | None = None,
    filter_prefix: str = FILTER_QUERY_PREFIX,
) -> str:
    return "?" + urlencode(build_query_param_payload(base_params=base_params, field_filters=field_filters, filter_prefix=filter_prefix))


def write_query_params(
    st: Any,
    *,
    base_params: Mapping[str, object],
    field_filters: Mapping[str, object] | None = None,
    filter_prefix: str = FILTER_QUERY_PREFIX,
) -> None:
    payload = build_query_param_payload(base_params=base_params, field_filters=field_filters, filter_prefix=filter_prefix)
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
        return

