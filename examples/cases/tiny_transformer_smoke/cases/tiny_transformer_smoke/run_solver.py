from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from build_solver import build_solver
else:
    from .build_solver import build_solver

from mlblack.project.scaffold import print_case_check


def main(argv=None):
    parser = argparse.ArgumentParser(description="tiny transformer smoke case")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--steps", type=int, default=1)
    args = parser.parse_args(argv)
    runner = build_solver()
    runner.steps = max(1, int(args.steps))
    if args.check:
        print_case_check(runner)
        return 0
    runner.run()
    return 0


if __name__ == "__main__":
    main()
