from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nowcasting_work_ci.run_solver import main as run_solver_main


def main() -> None:
    argv = list(sys.argv[1:])
    if "--interval-method" not in argv:
        argv = ["--interval-method", "symmetric_residual", *argv]
    run_solver_main(argv=argv)


if __name__ == "__main__":
    main()
