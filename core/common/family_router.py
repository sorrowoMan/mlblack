from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


def _normalize_name(value: str | None, default: str = "") -> str:
    text = str(value or "").strip().lower()
    return text or str(default)


def _normalize_match_values(value: Any) -> tuple[str, ...]:
    if value is None:
        return tuple()
    if isinstance(value, (tuple, list, set, frozenset)):
        return tuple(
            normalized
            for item in value
            for normalized in (_normalize_name(item),)
            if normalized
        )
    normalized = _normalize_name(value)
    return (normalized,) if normalized else tuple()


def _coerce_payload(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    describe = getattr(value, "description_dict", None)
    if callable(describe):
        described = describe()
        if isinstance(described, Mapping):
            return dict(described)
    as_dict = getattr(value, "as_dict", None)
    if callable(as_dict):
        described = as_dict()
        if isinstance(described, Mapping):
            return dict(described)
    if hasattr(value, "__dict__"):
        return dict(getattr(value, "__dict__"))
    return {}


def _payload_path_values(payload: Any, path: str) -> tuple[str, ...]:
    current: Any = _coerce_payload(payload)
    for part in tuple(str(path).split(".")):
        if isinstance(current, Mapping):
            current = current.get(part)
        else:
            current = getattr(current, part, None)
        if current is None:
            return tuple()
    return _normalize_match_values(current)


@dataclass(frozen=True)
class FamilyRouteSpec:
    family_key: str
    route_key: str
    match_fields: Mapping[str, Any] = field(default_factory=dict)
    status: str = "stable"
    summary: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "family_key", _normalize_name(self.family_key))
        object.__setattr__(self, "route_key", _normalize_name(self.route_key))
        object.__setattr__(
            self,
            "match_fields",
            {
                str(key).strip(): _normalize_match_values(value)
                for key, value in dict(self.match_fields).items()
                if str(key).strip()
            },
        )
        object.__setattr__(self, "status", _normalize_name(self.status, "stable"))
        object.__setattr__(self, "summary", str(self.summary or "").strip())
        object.__setattr__(self, "metadata", dict(self.metadata))

    def as_dict(self) -> dict[str, Any]:
        return {
            "family_key": self.family_key,
            "route_key": self.route_key,
            "match_fields": {
                str(key): tuple(values)
                for key, values in dict(self.match_fields).items()
            },
            "status": self.status,
            "summary": self.summary,
            "metadata": dict(self.metadata),
        }


def serialize_family_route_registry(
    routes: tuple[FamilyRouteSpec, ...] | list[FamilyRouteSpec] | None,
) -> list[dict[str, Any]]:
    route_specs = tuple(routes or ())
    return [route.as_dict() for route in route_specs]


def family_route_matches(route: FamilyRouteSpec, payload: Any) -> bool:
    for field_path, allowed_values in dict(route.match_fields).items():
        actual_values = _payload_path_values(payload, str(field_path))
        if not actual_values:
            return False
        if allowed_values and not any(value in tuple(allowed_values) for value in actual_values):
            return False
    return True


def match_family_routes(
    routes: tuple[FamilyRouteSpec, ...] | list[FamilyRouteSpec],
    payload: Any,
) -> tuple[FamilyRouteSpec, ...]:
    route_specs = tuple(routes or ())
    return tuple(route for route in route_specs if family_route_matches(route, payload))


def _route_context_label(
    routes: tuple[FamilyRouteSpec, ...],
    payload: Any,
) -> str:
    field_paths = tuple(
        sorted(
            {
                str(field_path)
                for route in routes
                for field_path in dict(route.match_fields).keys()
                if str(field_path).strip()
            }
        )
    )
    labels: list[str] = []
    for field_path in field_paths:
        values = _payload_path_values(payload, field_path)
        rendered = "|".join(values) if values else "<missing>"
        labels.append(f"{field_path}='{rendered}'")
    return ", ".join(labels)


def _route_label(route: FamilyRouteSpec) -> str:
    match_label = ", ".join(
        f"{field_path}={'|'.join(values) if values else '*'}"
        for field_path, values in dict(route.match_fields).items()
    )
    return f"{route.route_key}({match_label})"


def resolve_family_route_spec(
    routes: tuple[FamilyRouteSpec, ...] | list[FamilyRouteSpec],
    payload: Any,
    *,
    family_key: str | None = None,
) -> FamilyRouteSpec:
    route_specs = tuple(routes or ())
    matches = match_family_routes(route_specs, payload)
    if len(matches) == 1:
        return matches[0]

    family_label = _normalize_name(family_key or (route_specs[0].family_key if route_specs else "family"), "family")
    registry_label = "; ".join(_route_label(route) for route in route_specs)
    context_label = _route_context_label(route_specs, payload)

    if len(matches) > 1:
        matched_label = "; ".join(_route_label(route) for route in matches)
        raise ValueError(
            f"{family_label} route conflict: {context_label} matched multiple registered routes "
            f"[{matched_label}]. Registered routes: [{registry_label}]"
        )

    raise ValueError(
        f"unsupported {family_label} route: {context_label} did not match any registered route. "
        f"Registered routes: [{registry_label}]"
    )


__all__ = [
    "FamilyRouteSpec",
    "family_route_matches",
    "match_family_routes",
    "resolve_family_route_spec",
    "serialize_family_route_registry",
]
