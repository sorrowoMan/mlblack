from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from .context_keys import METRIC_FALLBACKS, METRIC_KEYS, normalize_context_keys, unknown_context_keys


@dataclass(frozen=True)
class ContextContract:
    """nsgablack-style context declaration for a component class.

    Components declare plain class attributes:
    context_requires/context_provides/context_mutates/context_cache,
    requires_metrics, metrics_fallback and context_notes.
    """

    name: str = ""
    context_requires: tuple[str, ...] = tuple()
    context_optional: tuple[str, ...] = tuple()
    context_provides: tuple[str, ...] = tuple()
    context_mutates: tuple[str, ...] = tuple()
    context_cache: tuple[str, ...] = tuple()
    requires_metrics: tuple[str, ...] = tuple()
    metrics_fallback: str = "strict"
    context_notes: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_component(cls, component: Any, *, fallback_contract: Any | None = None) -> "ContextContract":
        source = component if isinstance(component, type) else type(component)
        fallback = fallback_contract
        name = str(getattr(component, "name", getattr(source, "name", getattr(fallback, "name", source.__name__))))
        return cls(
            name=name,
            context_requires=_read_keys(component, source, "context_requires", getattr(fallback, "requires", ())),
            context_optional=_read_keys(component, source, "context_optional", getattr(fallback, "optional", ())),
            context_provides=_read_keys(component, source, "context_provides", getattr(fallback, "provides", ())),
            context_mutates=_read_keys(component, source, "context_mutates", getattr(fallback, "mutates", ())),
            context_cache=_read_keys(component, source, "context_cache", getattr(fallback, "cache", ())),
            requires_metrics=tuple(str(v) for v in getattr(component, "requires_metrics", getattr(source, "requires_metrics", ()))),
            metrics_fallback=str(getattr(component, "metrics_fallback", getattr(source, "metrics_fallback", "strict"))),
            context_notes=str(getattr(component, "context_notes", getattr(source, "context_notes", ""))),
            metadata=dict(getattr(fallback, "metadata", {}) or {}),
        )

    def all_context_keys(self) -> tuple[str, ...]:
        return normalize_context_keys((
            *self.context_requires,
            *self.context_optional,
            *self.context_provides,
            *self.context_mutates,
            *self.context_cache,
        ))

    def unknown_keys(self) -> tuple[str, ...]:
        return unknown_context_keys(self.all_context_keys())

    def unknown_metric_keys(self) -> tuple[str, ...]:
        known = {str(key) for key in METRIC_KEYS}
        return tuple(metric for metric in self.requires_metrics if str(metric) not in known)

    def validate(self, *, strict: bool = False) -> tuple[str, ...]:
        unknown = self.unknown_keys()
        if strict and unknown:
            raise ValueError(f"{self.name} declares unknown context keys: {unknown}")
        unknown_metrics = self.unknown_metric_keys()
        if strict and unknown_metrics:
            raise ValueError(f"{self.name} declares unknown metric keys: {unknown_metrics}")
        if self.metrics_fallback not in METRIC_FALLBACKS:
            if strict:
                raise ValueError(f"{self.name} declares invalid metrics_fallback: {self.metrics_fallback}")
        return unknown

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "context_requires": list(self.context_requires),
            "context_optional": list(self.context_optional),
            "context_provides": list(self.context_provides),
            "context_mutates": list(self.context_mutates),
            "context_cache": list(self.context_cache),
            "requires_metrics": list(self.requires_metrics),
            "metrics_fallback": self.metrics_fallback,
            "context_notes": self.context_notes,
            "metadata": dict(self.metadata),
        }


def _read_keys(component: Any, source: Any, attr: str, fallback: Iterable[str]) -> tuple[str, ...]:
    if hasattr(component, attr):
        return normalize_context_keys(getattr(component, attr))
    if hasattr(source, attr):
        return normalize_context_keys(getattr(source, attr))
    return normalize_context_keys(fallback)
