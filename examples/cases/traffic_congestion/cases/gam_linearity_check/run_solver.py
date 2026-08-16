"""Canonical CLI for the GAM linearity diagnostic Case."""

from __future__ import annotations

import argparse

from blackbase.project import load_resource_context_from_env

try:
    from .build_solver import build_solver
except ImportError:  # direct script execution
    from build_solver import build_solver

from mlblack.project.scaffold import print_case_check


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run the GAM linearity diagnostic.")
    parser.add_argument("--n-knots", type=int, default=6, help="B-spline knots")
    parser.add_argument("--degree", type=int, default=3, help="B-spline degree")
    parser.add_argument("--top-k", type=int, default=8, help="Top features to show")
    parser.add_argument("--check", action="store_true", help="Build and validate only; do not run analysis")
    args = parser.parse_args(argv)
    trainer = build_solver(
        config={"n_knots": args.n_knots, "degree": args.degree, "top_k": args.top_k},
        resource_context=load_resource_context_from_env("mlblack"),
    )
    if args.check:
        print_case_check(trainer)
        return 0
    trainer.fit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
