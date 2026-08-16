from __future__ import annotations

from pathlib import Path
import sys


def _prepend(path: Path) -> None:
    text = str(path.resolve())
    if text not in sys.path:
        sys.path.insert(0, text)


_ROOT = Path(__file__).resolve().parent
_prepend(_ROOT)

# Make the current mlblack and sibling nsgablack repos importable for subprocess runs.
for _cur in (_ROOT, *_ROOT.parents):
    if (_cur / "mlblack.py").is_file() and (_cur / "pyproject.toml").is_file():
        _prepend(_cur.parent)
        break

for _candidate in (
    _ROOT.parents[4] / "nsgablack" if len(_ROOT.parents) > 4 else None,
    Path.home() / "Desktop" / "nsgablack",
):
    if _candidate is not None and (_candidate / "__init__.py").is_file() and (_candidate / "core").is_dir():
        _prepend(_candidate.parent)
        break

