from __future__ import annotations

from pathlib import Path
import sys


def ensure_case_importable(start: Path | None = None) -> None:
    origin = (start or Path(__file__)).resolve()
    for cur in (origin, *origin.parents):
        if (cur / "legacy_nowcasting").is_dir():
            _prepend(cur / "legacy_nowcasting")
            _prepend(cur)
            break
    for cur in (origin, *origin.parents):
        if (cur / "mlblack.py").is_file() and (cur / "pyproject.toml").is_file():
            _prepend(cur.parent)
            break


def _prepend(path: Path) -> None:
    text = str(path.resolve())
    if text not in sys.path:
        sys.path.insert(0, text)

