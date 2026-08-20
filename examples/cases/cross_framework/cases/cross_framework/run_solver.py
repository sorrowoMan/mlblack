from __future__ import annotations

# CLI contract: --check builds the real nested Case assembly without executing it.

from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from build_solver import main  # noqa: E402
else:
    from .build_solver import main


if __name__ == "__main__":
    main()
