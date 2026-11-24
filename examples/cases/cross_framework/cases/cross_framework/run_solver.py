from __future__ import annotations

# CLI contract: --check builds the real nested Case assembly without executing it.

import argparse
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from build_solver import build_solver  # noqa: E402
else:
    from .build_solver import build_solver

from mlblack.project.scaffold import print_case_check


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run the formal nested cross-framework Case")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    solver = build_solver()
    if args.check:
        print_case_check(solver)
        return 0
    print(solver.run())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
