# -*- coding: utf-8 -*-
from __future__ import annotations

"""Deprecated compatibility wrapper for deterministic smoke regression tooling.

Preferred implementation location:
`nowcasting_work_ci.tools.run_deterministic_smoke_regression`.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nowcasting_work_ci.tools.run_deterministic_smoke_regression import (
    build_summary_snapshot,
    diff_summary_snapshots,
    main,
    run_smoke_once,
)

__all__ = [
    "build_summary_snapshot",
    "diff_summary_snapshots",
    "main",
    "run_smoke_once",
]


if __name__ == "__main__":
    raise SystemExit(main())
