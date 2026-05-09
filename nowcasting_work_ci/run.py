# -*- coding: utf-8 -*-
"""Preferred public CLI entrypoint.

Intentional architecture boundary:
- nsgablack_side: outer orchestration / solver assembly
- mlblack_side: model-side evaluation and training runtime
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nowcasting_work_ci.mlblack_side.runtime import main as run_main


def main(argv: list[str] | None = None) -> None:
    run_main(argv if argv is not None else None)


if __name__ == "__main__":
    main()
