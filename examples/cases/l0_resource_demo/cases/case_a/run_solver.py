from __future__ import annotations

import argparse

from mlblack.project.scaffold import print_case_check

try:
    from .build_solver import build_solver
except ImportError:  # direct script execution
    from build_solver import build_solver


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run the mlblack L0 resource demo.")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--steps", type=int, default=1)
    args = parser.parse_args(argv)

    trainer = build_solver()
    if args.check:
        print_case_check(trainer)
        return 0
    trainer.run(max_steps=max(1, int(args.steps)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
