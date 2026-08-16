"""Canonical CLI for the TFT temporal forecast Trainer."""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from blackbase.project import load_resource_context_from_env

try:
    from .build_solver import build_solver
except ImportError:  # direct script execution
    from build_solver import build_solver

from mlblack.project.scaffold import print_case_check


def _extract_rmse(trainer) -> float:
    feedback = getattr(trainer, "best_feedback", None)
    if feedback is None:
        return float("nan")
    metrics = dict(getattr(feedback, "metrics", {}) or {})
    for key in ("valid.rmse", "train.rmse"):
        if key in metrics:
            return float(metrics[key])
    objectives = np.asarray(feedback.objectives, dtype=float).ravel()
    return float(np.sqrt(objectives[0])) if len(objectives) else float("nan")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run the TFT temporal forecast Case.")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--check", action="store_true", help="Build and validate only; do not fit")
    args = parser.parse_args(argv)
    trainer = build_solver(resource_context=load_resource_context_from_env("mlblack"))
    if args.check:
        print_case_check(trainer)
        return 0

    started = time.perf_counter()
    trainer.fit(max_steps=max(1, int(args.steps)))
    resource = trainer.get_resource_context().as_dict()
    print(
        "[case-result] "
        + json.dumps(
            {
                "model": "TFT",
                "rmse": _extract_rmse(trainer),
                "elapsed_seconds": time.perf_counter() - started,
                "namespace": resource.get("namespace", ""),
                "threads": resource.get("threads", 1),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

