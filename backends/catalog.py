from __future__ import annotations

from typing import Any

from .registry import get_backend, list_backends


def list_backend_catalog_entries() -> tuple[dict[str, Any], ...]:
    entries: list[dict[str, Any]] = []
    for name in list_backends():
        backend = get_backend(name)
        contract = backend.contract()
        entries.append(
            {
                "kind": "backend",
                "name": contract.name,
                "provides": tuple(contract.provides),
                "methods": dict(contract.methods),
                "metadata": dict(contract.metadata),
            }
        )
        for capability in contract.capabilities:
            entries.append(
                {
                    "kind": "backend_capability",
                    "name": f"{capability.backend}.{capability.capability}",
                    "backend": capability.backend,
                    "capability": capability.capability,
                    "provides": tuple(capability.provides),
                    "methods": dict(capability.methods),
                    "routes": tuple(capability.routes),
                    "heads": tuple(capability.heads),
                    "metadata": capability.as_dict(),
                }
            )
    return tuple(entries)


__all__ = ["list_backend_catalog_entries"]
