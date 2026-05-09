from __future__ import annotations

"""Deprecated compatibility wrapper for the reporting utility.

Preferred implementation location:
`nowcasting_work_ci.tools.aggregate_and_plot_results`.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nowcasting_work_ci.tools.aggregate_and_plot_results import main

__all__ = ["main"]


if __name__ == "__main__":
    main()
