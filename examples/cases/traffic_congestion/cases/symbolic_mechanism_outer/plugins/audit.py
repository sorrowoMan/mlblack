from __future__ import annotations

from typing import Any, Mapping


def build_case_audit(*, resource_context: Mapping[str, Any] | None = None) -> dict[str, Any]:
    return {
        "case": "symbolic_mechanism_outer",
        "bridge": "legacy_nowcasting_split_modules",
        "resource_context": dict(resource_context or {}),
    }


__all__ = ["build_case_audit"]
