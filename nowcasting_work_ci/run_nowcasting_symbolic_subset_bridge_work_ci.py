from __future__ import annotations

"""Deprecated compatibility shim.

Preferred location:
`nowcasting_work_ci.compat.run_nowcasting_symbolic_subset_bridge_work_ci`.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nowcasting_work_ci.compat.run_nowcasting_symbolic_subset_bridge_work_ci import main

__all__ = ["main"]


if __name__ == "__main__":
    main()
