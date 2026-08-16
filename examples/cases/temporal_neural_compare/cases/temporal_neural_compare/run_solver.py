"""Compatibility CLI for the historical temporal-neural comparison Case."""

from __future__ import annotations

import argparse

from blackbase.project import load_resource_context_from_env

try:
    from .build_solver import build_solver
except ImportError:  # direct script execution
    from build_solver import build_solver

from mlblack.project.scaffold import print_case_check


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the historical comparison entry as one LSTM compatibility Trainer."
    )
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--check", action="store_true", help="Build and validate only; do not fit")
    args = parser.parse_args(argv)
    trainer = build_solver(resource_context=load_resource_context_from_env("mlblack"))
    if args.check:
        print_case_check(trainer)
        return 0
    trainer.fit(max_steps=max(1, int(args.steps)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
