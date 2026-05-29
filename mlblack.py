"""Development import shim for the promoted package-root layout.

This repository root is also the ``mlblack`` package directory. When Python is
started from this directory, ``import mlblack`` normally looks for a nested
``mlblack`` directory and fails. This shim marks the current module as a package
and executes the real package ``__init__.py`` so local one-liners work from the
repo root.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent

# Unified architecture: ensure nsgablack (sibling package) is importable.
_NSGABLACK_ROOT = _ROOT.parent / "nsgablack"
_NSGABLACK_PARENT = str(_NSGABLACK_ROOT.parent)
if _NSGABLACK_ROOT.is_dir() and _NSGABLACK_PARENT not in sys.path:
    sys.path.insert(0, _NSGABLACK_PARENT)

if __name__ == "__main__":
    parent = str(_ROOT.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    from mlblack.cli import main

    raise SystemExit(main())

__path__ = [str(_ROOT)]  # type: ignore[var-annotated]
if __spec__ is not None:
    __spec__.submodule_search_locations = __path__
__package__ = __name__

_INIT = _ROOT / "__init__.py"
exec(compile(_INIT.read_text(encoding="utf-8-sig"), str(_INIT), "exec"), globals(), globals())
