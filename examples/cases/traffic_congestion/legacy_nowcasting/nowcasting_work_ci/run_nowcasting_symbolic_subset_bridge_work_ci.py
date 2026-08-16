from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LEGACY_ROOT = PROJECT_ROOT / "legacy_nowcasting"
for path in (LEGACY_ROOT, PROJECT_ROOT):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

from cases.symbolic_mechanism_outer.runtime.runner import main


if __name__ == "__main__":
    main()
