"""Local development path helper for the promoted mlblack-root layout.

The repository root is the mlblack package directory itself. When commands are
run from inside this directory, Python's default sys.path points at the package
contents, not at its parent. Adding the parent keeps `import mlblack` working for
local tests and examples without reintroducing a nested mlblack/mlblack layout.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PARENT = str(Path(__file__).resolve().parent.parent)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)
