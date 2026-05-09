# -*- coding: utf-8 -*-
"""Deprecated compatibility CLI entrypoint.

Preferred entrypoint is `nowcasting_work_ci/run.py`.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nowcasting_work_ci.run import main as run_main


def main(argv: list[str] | None = None) -> None:
    forwarded = list(argv if argv is not None else sys.argv[1:])
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]
    if "--check" in forwarded:
        print("[check] scaffold entry ok | target=run.py -> mlblack_side.runtime.main")
        return
    run_main(forwarded)


if __name__ == "__main__":
    main()
