from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nowcasting_work_ci.mlblack_side.orthogonal_basis import main as run_main


def main(argv: list[str] | None = None) -> None:
    run_main(argv if argv is not None else None)


if __name__ == "__main__":
    main()
