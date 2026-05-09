from __future__ import annotations

"""Deprecated compatibility shim for the formal experiment dashboard surface."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment.dashboard import main


if __name__ == "__main__":
    main()
