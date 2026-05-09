from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


MECHANISM_BINDING_LEVELS: tuple[str, ...] = ("optional", "bound", "defining")
MECHANISM_KINDS: tuple[str, ...] = ("sampling", "sample_weighting", "state_signal_view", "aggregation")


def _normalize_name(value: str | None, default: str) -> str:
    text = str(value or "").strip().lower()
    return text or str(default)


def _normalize_binding_level(value: str | None) -> str:
    level = _normalize_name(value, "optional")
    if level not in MECHANISM_BINDING_LEVELS:
        raise ValueError(
            f"mechanism binding_level must be one of {MECHANISM_BINDING_LEVELS}, got '{value}'"
        )
    return level


def _normalize_kind(value: str | None, default: str) -> str:
    kind = _normalize_name(value, default)
    if kind not in MECHANISM_KINDS:
        raise ValueError(f"mechanism kind must be one of {MECHANISM_KINDS}, got '{value}'")
    return kind


@dataclass(frozen=True)
class MechanismProtocolBase:
    mechanism_key: str
    mechanism_kind: str
    binding_level: str = "optional"
    required_fields: tuple[str, ...] = tuple()
    optional_fields: tuple[str, ...] = tuple()
    provides_fields: tuple[str, ...] = tuple()
    notes: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "mechanism_key", _normalize_name(self.mechanism_key, "mechanism"))
        object.__setattr__(self, "mechanism_kind", _normalize_kind(self.mechanism_kind, "sampling"))
        object.__setattr__(self, "binding_level", _normalize_binding_level(self.binding_level))
        object.__setattr__(self, "required_fields", tuple(str(v) for v in tuple(self.required_fields)))
        object.__setattr__(self, "optional_fields", tuple(str(v) for v in tuple(self.optional_fields)))
        object.__setattr__(self, "provides_fields", tuple(str(v) for v in tuple(self.provides_fields)))
        object.__setattr__(self, "notes", None if self.notes is None else str(self.notes))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def as_dict(self) -> dict[str, Any]:
        return {
            "mechanism_key": str(self.mechanism_key),
            "mechanism_kind": str(self.mechanism_kind),
            "binding_level": str(self.binding_level),
            "required_fields": [str(v) for v in self.required_fields],
            "optional_fields": [str(v) for v in self.optional_fields],
            "provides_fields": [str(v) for v in self.provides_fields],
            "notes": None if self.notes is None else str(self.notes),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class SamplingProtocol(MechanismProtocolBase):
    selection_axes: tuple[str, ...] = ("row",)
    output_view_kind: str = "sampled_view_ref"

    def __init__(
        self,
        *,
        mechanism_key: str = "sampling",
        binding_level: str = "optional",
        required_fields: Sequence[str] = tuple(),
        optional_fields: Sequence[str] = tuple(),
        provides_fields: Sequence[str] = ("sampled_view_ref",),
        selection_axes: Sequence[str] = ("row",),
        output_view_kind: str = "sampled_view_ref",
        notes: str | None = None,
        metadata: Mapping[str, Any] = {},
    ) -> None:
        super().__init__(
            mechanism_key=mechanism_key,
            mechanism_kind="sampling",
            binding_level=binding_level,
            required_fields=tuple(required_fields),
            optional_fields=tuple(optional_fields),
            provides_fields=tuple(provides_fields),
            notes=notes,
            metadata=dict(metadata),
        )
        object.__setattr__(self, "selection_axes", tuple(str(v) for v in tuple(selection_axes)))
        object.__setattr__(self, "output_view_kind", str(output_view_kind or "sampled_view_ref"))

    def as_dict(self) -> dict[str, Any]:
        payload = super().as_dict()
        payload.update(
            {
                "selection_axes": [str(v) for v in self.selection_axes],
                "output_view_kind": str(self.output_view_kind),
            }
        )
        return payload


@dataclass(frozen=True)
class SampleWeightingProtocol(MechanismProtocolBase):
    output_weight_kind: str = "sample_weight_ref"

    def __init__(
        self,
        *,
        mechanism_key: str = "sample_weighting",
        binding_level: str = "optional",
        required_fields: Sequence[str] = tuple(),
        optional_fields: Sequence[str] = tuple(),
        provides_fields: Sequence[str] = ("sample_weight_ref",),
        output_weight_kind: str = "sample_weight_ref",
        notes: str | None = None,
        metadata: Mapping[str, Any] = {},
    ) -> None:
        super().__init__(
            mechanism_key=mechanism_key,
            mechanism_kind="sample_weighting",
            binding_level=binding_level,
            required_fields=tuple(required_fields),
            optional_fields=tuple(optional_fields),
            provides_fields=tuple(provides_fields),
            notes=notes,
            metadata=dict(metadata),
        )
        object.__setattr__(self, "output_weight_kind", str(output_weight_kind or "sample_weight_ref"))

    def as_dict(self) -> dict[str, Any]:
        payload = super().as_dict()
        payload["output_weight_kind"] = str(self.output_weight_kind)
        return payload


@dataclass(frozen=True)
class StateSignalViewProtocol(MechanismProtocolBase):
    signal_names: tuple[str, ...] = tuple()

    def __init__(
        self,
        *,
        mechanism_key: str = "state_signal_view",
        binding_level: str = "optional",
        required_fields: Sequence[str] = tuple(),
        optional_fields: Sequence[str] = tuple(),
        provides_fields: Sequence[str] = tuple(),
        signal_names: Sequence[str] = tuple(),
        notes: str | None = None,
        metadata: Mapping[str, Any] = {},
    ) -> None:
        super().__init__(
            mechanism_key=mechanism_key,
            mechanism_kind="state_signal_view",
            binding_level=binding_level,
            required_fields=tuple(required_fields),
            optional_fields=tuple(optional_fields),
            provides_fields=tuple(provides_fields),
            notes=notes,
            metadata=dict(metadata),
        )
        object.__setattr__(self, "signal_names", tuple(str(v) for v in tuple(signal_names)))

    def as_dict(self) -> dict[str, Any]:
        payload = super().as_dict()
        payload["signal_names"] = [str(v) for v in self.signal_names]
        return payload


@dataclass(frozen=True)
class AggregationProtocol(MechanismProtocolBase):
    aggregation_mode: str = "mean"

    def __init__(
        self,
        *,
        mechanism_key: str = "aggregation",
        binding_level: str = "optional",
        required_fields: Sequence[str] = tuple(),
        optional_fields: Sequence[str] = tuple(),
        provides_fields: Sequence[str] = ("aggregated_output_ref",),
        aggregation_mode: str = "mean",
        notes: str | None = None,
        metadata: Mapping[str, Any] = {},
    ) -> None:
        super().__init__(
            mechanism_key=mechanism_key,
            mechanism_kind="aggregation",
            binding_level=binding_level,
            required_fields=tuple(required_fields),
            optional_fields=tuple(optional_fields),
            provides_fields=tuple(provides_fields),
            notes=notes,
            metadata=dict(metadata),
        )
        object.__setattr__(self, "aggregation_mode", str(aggregation_mode or "mean").strip().lower() or "mean")

    def as_dict(self) -> dict[str, Any]:
        payload = super().as_dict()
        payload["aggregation_mode"] = str(self.aggregation_mode)
        return payload


def coerce_mechanism_protocol(
    value: MechanismProtocolBase | Mapping[str, Any],
) -> MechanismProtocolBase:
    if isinstance(value, MechanismProtocolBase):
        return value
    raw = dict(value)
    kind = _normalize_kind(raw.get("mechanism_kind"), "sampling")
    kwargs = {
        "mechanism_key": raw.get("mechanism_key", kind),
        "binding_level": raw.get("binding_level", "optional"),
        "required_fields": tuple(raw.get("required_fields", tuple())),
        "optional_fields": tuple(raw.get("optional_fields", tuple())),
        "provides_fields": tuple(raw.get("provides_fields", tuple())),
        "notes": raw.get("notes"),
        "metadata": dict(raw.get("metadata", {})),
    }
    if kind == "sampling":
        return SamplingProtocol(
            **kwargs,
            selection_axes=tuple(raw.get("selection_axes", ("row",))),
            output_view_kind=str(raw.get("output_view_kind", "sampled_view_ref")),
        )
    if kind == "sample_weighting":
        return SampleWeightingProtocol(
            **kwargs,
            output_weight_kind=str(raw.get("output_weight_kind", "sample_weight_ref")),
        )
    if kind == "state_signal_view":
        return StateSignalViewProtocol(
            **kwargs,
            signal_names=tuple(raw.get("signal_names", tuple())),
        )
    return AggregationProtocol(
        **kwargs,
        aggregation_mode=str(raw.get("aggregation_mode", "mean")),
    )


def serialize_mechanism_protocols(
    values: Sequence[MechanismProtocolBase | Mapping[str, Any]] | None,
) -> list[dict[str, Any]]:
    if values is None:
        return []
    return [coerce_mechanism_protocol(value).as_dict() for value in tuple(values)]


__all__ = [
    "AggregationProtocol",
    "MECHANISM_BINDING_LEVELS",
    "MECHANISM_KINDS",
    "MechanismProtocolBase",
    "SampleWeightingProtocol",
    "SamplingProtocol",
    "StateSignalViewProtocol",
    "coerce_mechanism_protocol",
    "serialize_mechanism_protocols",
]
