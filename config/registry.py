from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping, Tuple

Factory = Callable[..., Any]


def _normalize_key(key: str) -> str:
    return str(key).strip().lower()


@dataclass
class ComponentRegistry:
    """Generic registry for factory-based component construction."""

    kind: str
    _factories: Dict[str, Factory] = field(default_factory=dict)
    _metadata: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def register(
        self,
        key: str,
        factory: Factory,
        *,
        replace: bool = False,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        k = _normalize_key(key)
        if not k:
            raise ValueError(f"{self.kind} key must not be empty")
        if not callable(factory):
            raise TypeError(f"{self.kind} factory for '{k}' must be callable")
        if k in self._factories and not bool(replace):
            raise KeyError(f"{self.kind} '{k}' already registered")
        self._factories[k] = factory
        self._metadata[k] = dict(metadata or {})

    def create(self, key: str, **kwargs: Any) -> Any:
        k = _normalize_key(key)
        factory = self._factories.get(k)
        if factory is None:
            available = ", ".join(sorted(self._factories.keys()))
            raise KeyError(f"Unknown {self.kind}: '{key}'. Available: [{available}]")
        return factory(**kwargs)

    def get(self, key: str) -> Factory | None:
        return self._factories.get(_normalize_key(key))

    def keys(self) -> Tuple[str, ...]:
        return tuple(sorted(self._factories.keys()))

    def metadata(self, key: str) -> Dict[str, Any]:
        return dict(self._metadata.get(_normalize_key(key), {}))

    def describe(self) -> Tuple[Dict[str, Any], ...]:
        return tuple(
            {
                "key": key,
                "metadata": self.metadata(key),
            }
            for key in self.keys()
        )


@dataclass
class MLBlackConfig:
    """Holds registries for semantic-typed assembly."""

    pipelines: ComponentRegistry = field(default_factory=lambda: ComponentRegistry(kind="pipeline"))
    biases: ComponentRegistry = field(default_factory=lambda: ComponentRegistry(kind="bias"))
    numericizers: ComponentRegistry = field(default_factory=lambda: ComponentRegistry(kind="numericizer"))
    trainers: ComponentRegistry = field(default_factory=lambda: ComponentRegistry(kind="trainer"))
    capabilities: ComponentRegistry = field(default_factory=lambda: ComponentRegistry(kind="capability"))
