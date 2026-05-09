from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping


def write_scaffold_report(payload: Mapping[str, Any], output_path: str | Path) -> str:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(dict(payload)), encoding="utf-8")
    return str(path)


__all__ = ["write_scaffold_report"]
