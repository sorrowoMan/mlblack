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


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Cross-framework inner Trainer Case")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--steps", type=int, default=2)
    args = parser.parse_args(argv)
    trainer = build_solver(component_overrides={"max_steps": int(args.steps)})
    if args.check:
        print_case_check(trainer)
        return 0
    print(trainer.fit(max_steps=max(1, int(args.steps))).report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
