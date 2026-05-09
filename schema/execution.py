from __future__ import annotations

from typing import Any, Dict

from config import describe_execution_spec_schema


EXECUTION_SPEC_SCHEMA: Dict[str, Any] = describe_execution_spec_schema()


def get_execution_spec_schema() -> Dict[str, Any]:
    return describe_execution_spec_schema()


__all__ = ["EXECUTION_SPEC_SCHEMA", "get_execution_spec_schema"]
