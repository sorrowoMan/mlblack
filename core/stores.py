from __future__ import annotations

from collections.abc import Iterator, MutableMapping
from dataclasses import dataclass, field
from typing import Any, Mapping
from uuid import uuid4


@dataclass(frozen=True)
class SnapshotRecord:
    """Metadata envelope for snapshot payloads when a backend wants one."""

    payload: Any
    kind: str = "snapshot"
    metadata: Mapping[str, Any] = field(default_factory=dict)


class InMemoryContextStore(MutableMapping[str, Any]):
    """Small runtime state store.

    Context should hold lightweight values and references to snapshots, not
    large arrays, traces or model payloads.
    """

    def __init__(self, initial: Mapping[str, Any] | None = None) -> None:
        self._data: dict[str, Any] = dict(initial or {})

    def __getitem__(self, key: str) -> Any:
        return self._data[str(key)]

    def __setitem__(self, key: str, value: Any) -> None:
        self._data[str(key)] = value

    def __delitem__(self, key: str) -> None:
        del self._data[str(key)]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def write(self, key: str, value: Any) -> None:
        self[str(key)] = value

    def read(self, key: str, default: Any = None) -> Any:
        return self._data.get(str(key), default)

    def require(self, key: str) -> Any:
        key = str(key)
        if key not in self._data:
            raise KeyError(f"context key not found: {key}")
        return self._data[key]

    def project(self, prefix: str) -> dict[str, Any]:
        prefix = str(prefix)
        return {key: value for key, value in self._data.items() if key.startswith(prefix)}

    def to_dict(self) -> dict[str, Any]:
        return dict(self._data)


class InMemorySnapshotStore(MutableMapping[str, Any]):
    """In-memory snapshot store for heavy runtime payloads."""

    def __init__(self, initial: Mapping[str, Any] | None = None) -> None:
        self._data: dict[str, Any] = dict(initial or {})

    def __getitem__(self, key: str) -> Any:
        return self._data[str(key)]

    def __setitem__(self, key: str, value: Any) -> None:
        self._data[str(key)] = value

    def __delitem__(self, key: str) -> None:
        del self._data[str(key)]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def write(
        self,
        payload: Any,
        *,
        key: str | None = None,
        kind: str = "snapshot",
        metadata: Mapping[str, Any] | None = None,
        envelope: bool = False,
    ) -> str:
        snapshot_key = str(key or f"{kind}:{uuid4().hex}")
        self._data[snapshot_key] = (
            SnapshotRecord(payload=payload, kind=str(kind), metadata=dict(metadata or {}))
            if envelope
            else payload
        )
        return snapshot_key

    def read(self, key: str) -> Any:
        return self._data[str(key)]

    def to_dict(self) -> dict[str, Any]:
        return dict(self._data)
