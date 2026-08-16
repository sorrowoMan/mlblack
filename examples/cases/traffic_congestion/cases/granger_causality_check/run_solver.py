"""Canonical CLI for the Granger-causality diagnostic Case."""

from __future__ import annotations

import argparse

from blackbase.project import load_resource_context_from_env

try:
    from .build_solver import build_solver
except ImportError:  # direct script execution
    from build_solver import build_solver

from mlblack.project.scaffold import print_case_check


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run the Granger-causality diagnostic.")
    parser.add_argument("--maxlag", type=int, default=7, help="Maximum lag for Granger test")
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance threshold")
    parser.add_argument("--check", action="store_true", help="Build and validate only; do not run analysis")
    args = parser.parse_args(argv)
    trainer = build_solver(
        config={"maxlag": args.maxlag, "alpha": args.alpha},
        resource_context=load_resource_context_from_env("mlblack"),
    )
    if args.check:
        print_case_check(trainer)
        return 0
    trainer.fit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
