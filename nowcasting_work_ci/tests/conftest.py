from __future__ import annotations

import sys
from pathlib import Path


DESKTOP = Path(__file__).resolve().parents[3]
MLBLACK_ROOT = DESKTOP / "mlblack"
if str(MLBLACK_ROOT) not in sys.path:
    sys.path.insert(0, str(MLBLACK_ROOT))

from nowcasting_work_ci._internal.repo_paths import ensure_repo_import_order

ensure_repo_import_order(
    mlblack_root=MLBLACK_ROOT,
    nsgablack_root=DESKTOP / "nsgablack",
)
