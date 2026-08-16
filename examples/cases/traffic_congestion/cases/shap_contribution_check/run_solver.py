"""Canonical CLI for the feature-contribution consistency diagnostic Case."""

from __future__ import annotations

import argparse

from blackbase.project import load_resource_context_from_env

try:
    from .build_solver import build_solver
except ImportError:  # direct script execution
    from build_solver import build_solver

from mlblack.project.scaffold import print_case_check


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run feature-contribution consistency checks.")
    parser.add_argument("--n-estimators", type=int, default=200, help="XGBoost trees")
    parser.add_argument("--top-k", type=int, default=10, help="Top features to compare")
    parser.add_argument("--check", action="store_true", help="Build and validate only; do not run analysis")
    args = parser.parse_args(argv)
    trainer = build_solver(
        config={"n_estimators": args.n_estimators, "top_k": args.top_k},
        resource_context=load_resource_context_from_env("mlblack"),
    )
    if args.check:
        print_case_check(trainer)
        return 0
    trainer.fit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
