from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .context_contracts import ContextContract
from .context_keys import normalize_context_keys


@dataclass(frozen=True)
class ComponentContract:
    """Explicit composition contract for a framework component.

    Canonical component declarations are nsgablack-style class attributes:
    context_requires/context_provides/context_mutates/context_cache,
    requires_metrics, metrics_fallback and context_notes. This dataclass remains
    the serializable bridge used by existing reports, catalog and doctor.
    """

    name: str = ""
    requires: tuple[str, ...] = tuple()
    optional: tuple[str, ...] = tuple()
    provides: tuple[str, ...] = tuple()
    mutates: tuple[str, ...] = tuple()
    cache: tuple[str, ...] = tuple()
    supports_gradient: bool | None = None
    supports_batch: bool | None = None
    supports_resume: bool | None = None
    requires_metrics: tuple[str, ...] = tuple()
    metrics_fallback: str = "strict"
    context_notes: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_context_contract(
        cls,
        contract: ContextContract,
        *,
        supports_gradient: bool | None = None,
        supports_batch: bool | None = None,
        supports_resume: bool | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "ComponentContract":
        return cls(
            name=contract.name,
            requires=contract.context_requires,
            optional=contract.context_optional,
            provides=contract.context_provides,
            mutates=contract.context_mutates,
            cache=contract.context_cache,
            supports_gradient=supports_gradient,
            supports_batch=supports_batch,
            supports_resume=supports_resume,
            requires_metrics=contract.requires_metrics,
            metrics_fallback=contract.metrics_fallback,
            context_notes=contract.context_notes,
            metadata={**dict(contract.metadata), **dict(metadata or {})},
        )

    def to_context_contract(self) -> ContextContract:
        return ContextContract(
            name=self.name,
            context_requires=self.requires,
            context_optional=self.optional,
            context_provides=self.provides,
            context_mutates=self.mutates,
            context_cache=self.cache,
            requires_metrics=self.requires_metrics,
            metrics_fallback=self.metrics_fallback,
            context_notes=self.context_notes,
            metadata=dict(self.metadata),
        )

    def with_name(self, name: str) -> "ComponentContract":
        return ComponentContract(
            name=str(name),
            requires=self.requires,
            optional=self.optional,
            provides=self.provides,
            mutates=self.mutates,
            cache=self.cache,
            supports_gradient=self.supports_gradient,
            supports_batch=self.supports_batch,
            supports_resume=self.supports_resume,
            requires_metrics=_dedupe((*self.requires_metrics, *other.requires_metrics)),
            metrics_fallback=self.metrics_fallback,
            context_notes=self.context_notes,
            metadata=dict(self.metadata),
        )

    def merged(self, other: "ComponentContract", *, name: str | None = None) -> "ComponentContract":
        return ComponentContract(
            name=str(name if name is not None else (self.name or other.name)),
            requires=_dedupe((*self.requires, *other.requires)),
            optional=_dedupe((*self.optional, *other.optional)),
            provides=_dedupe((*self.provides, *other.provides)),
            mutates=_dedupe((*self.mutates, *other.mutates)),
            cache=_dedupe((*self.cache, *other.cache)),
            supports_gradient=_merge_bool(self.supports_gradient, other.supports_gradient),
            supports_batch=_merge_bool(self.supports_batch, other.supports_batch),
            supports_resume=_merge_bool(self.supports_resume, other.supports_resume),
            requires_metrics=self.requires_metrics,
            metrics_fallback=other.metrics_fallback if other.metrics_fallback != "strict" else self.metrics_fallback,
            context_notes="; ".join(part for part in (self.context_notes, other.context_notes) if part),
            metadata={**dict(self.metadata), **dict(other.metadata)},
        )

    def describe(self) -> dict[str, Any]:
        context = self.to_context_contract().as_dict()
        return {
            "name": self.name,
            "requires": list(self.requires),
            "optional": list(self.optional),
            "provides": list(self.provides),
            "mutates": list(self.mutates),
            "cache": list(self.cache),
            "context_requires": list(self.requires),
            "context_optional": list(self.optional),
            "context_provides": list(self.provides),
            "context_mutates": list(self.mutates),
            "context_cache": list(self.cache),
            "requires_metrics": list(self.requires_metrics),
            "metrics_fallback": self.metrics_fallback,
            "context_notes": self.context_notes,
            "supports_gradient": self.supports_gradient,
            "supports_batch": self.supports_batch,
            "supports_resume": self.supports_resume,
            "metadata": dict(self.metadata),
            "context_contract": context,
        }


class ContractMixin:
    """Mixin for objects that expose a stable component contract."""

    context_requires: tuple[str, ...] = tuple()
    context_optional: tuple[str, ...] = tuple()
    context_provides: tuple[str, ...] = tuple()
    context_mutates: tuple[str, ...] = tuple()
    context_cache: tuple[str, ...] = tuple()
    requires_metrics: tuple[str, ...] = tuple()
    metrics_fallback: str = "strict"
    context_notes: str = ""
    contract = ComponentContract()

    def get_context_contract(self) -> ContextContract:
        raw = getattr(self, "contract", ComponentContract())
        fallback = _coerce_contract(raw)
        return ContextContract.from_component(self, fallback_contract=fallback)

    def get_contract(self) -> ComponentContract:
        raw = getattr(self, "contract", ComponentContract())
        fallback = _coerce_contract(raw)
        context_contract = ContextContract.from_component(self, fallback_contract=fallback)
        return ComponentContract.from_context_contract(
            context_contract,
            supports_gradient=fallback.supports_gradient,
            supports_batch=fallback.supports_batch,
            supports_resume=fallback.supports_resume,
            metadata=fallback.metadata,
        )


def combine_contracts(name: str, *contracts: ComponentContract) -> ComponentContract:
    merged = ComponentContract(name=name)
    for contract in contracts:
        merged = merged.merged(contract, name=name)
    return merged


def _coerce_contract(raw: Any) -> ComponentContract:
    if isinstance(raw, ComponentContract):
        return raw
    if isinstance(raw, ContextContract):
        return ComponentContract.from_context_contract(raw)
    if isinstance(raw, Mapping):
        payload = dict(raw)
        for source, target in (
            ("context_requires", "requires"),
            ("context_optional", "optional"),
            ("context_provides", "provides"),
            ("context_mutates", "mutates"),
            ("context_cache", "cache"),
        ):
            if source in payload and target not in payload:
                payload[target] = payload[source]
        for key in ("requires", "optional", "provides", "mutates", "cache"):
            if key in payload:
                payload[key] = normalize_context_keys(payload[key])
        return ComponentContract(**payload)
    raise TypeError("contract must be ComponentContract, ContextContract or mapping")


def _dedupe(values: tuple[str, ...]) -> tuple[str, ...]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        key = str(value)
        if key not in seen:
            seen.add(key)
            out.append(key)
    return tuple(out)


def _merge_bool(left: bool | None, right: bool | None) -> bool | None:
    if left is None:
        return right
    if right is None:
        return left
    return bool(left and right)
