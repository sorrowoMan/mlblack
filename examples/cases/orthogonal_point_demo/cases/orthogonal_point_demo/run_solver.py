# -*- coding: utf-8 -*-
"""Canonical CLI for the orthogonal point demo Case."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from build_solver import build_solver  # noqa: E402
else:
    from .build_solver import build_solver

from mlblack.project.scaffold import print_case_check


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run the orthogonal point demo")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--max-steps", type=int, default=300)
    args = parser.parse_args(argv)
    trainer = build_solver()
    if args.check:
        print_case_check(trainer)
        return 0
    result = trainer.fit(max_steps=args.max_steps)
    print(trainer.build_report())
    print(result.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
