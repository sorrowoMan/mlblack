# Alias entry for unified Solver = Trainer scaffold.
from __future__ import annotations

from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from build_solver import build_solver as build_trainer  # noqa: E402
else:
    from .build_solver import build_solver as build_trainer

__all__ = ["build_trainer"]
