"""Canonical CLI for the ARIMAX factor-attribution diagnostic Case."""

from __future__ import annotations

import argparse

from blackbase.project import load_resource_context_from_env

try:
    from .build_solver import build_solver
except ImportError:  # direct script execution
    from build_solver import build_solver

from mlblack.project.scaffold import print_case_check


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run ARIMAX factor attribution.")
    parser.add_argument("--ar-order", type=int, default=2, help="AR order")
    parser.add_argument("--ma-order", type=int, default=1, help="MA order")
    parser.add_argument("--diff", type=int, default=0, help="Differencing order")
    parser.add_argument("--check", action="store_true", help="Build and validate only; do not run analysis")
    args = parser.parse_args(argv)
    trainer = build_solver(
        config={"ar_order": args.ar_order, "ma_order": args.ma_order, "diff": args.diff},
        resource_context=load_resource_context_from_env("mlblack"),
    )
    if args.check:
        print_case_check(trainer)
        return 0
    trainer.fit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
