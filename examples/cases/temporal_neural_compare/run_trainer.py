# Alias CLI entry for unified Solver = Trainer scaffold.
from __future__ import annotations

from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from run_solver import main  # noqa: E402
else:
    from .run_solver import main

if __name__ == "__main__":
    main()
