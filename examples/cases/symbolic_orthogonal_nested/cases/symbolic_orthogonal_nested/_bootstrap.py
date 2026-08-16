from __future__ import annotations

import sys
from pathlib import Path


def ensure_case_importable(start: Path | None = None) -> None:
    """Make local mlblack and sibling nsgablack importable for direct script runs."""

    origin = (start or Path(__file__)).resolve()
    for cur in (origin, *origin.parents):
        if (cur / "mlblack").is_dir():
            _prepend(cur)
            break

    # The user's nsgablack repo is a sibling project with a flat package root.
    # Prefer discovery over a hard dependency on installation.
    candidates: list[Path] = []
    for cur in (origin, *origin.parents):
        candidates.append(cur / "nsgablack")
        candidates.append(cur.parent / "nsgablack")
    desktop = Path.home() / "Desktop"
    candidates.append(desktop / "nsgablack")
    for repo in candidates:
        if (repo / "__init__.py").is_file() and (repo / "core").is_dir():
            _prepend(repo.parent)
            return


def _prepend(path: Path) -> None:
    text = str(path.resolve())
    if text not in sys.path:
        sys.path.insert(0, text)


__all__ = ["ensure_case_importable"]
