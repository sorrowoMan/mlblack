# -*- coding: utf-8 -*-
"""Standard assembly entry for the orthogonal point demo case."""

from __future__ import annotations

import argparse
from pathlib import Path

from mlblack.presets import build_orthogonal_linear_point_trainer

from pipeline.main import build_pipeline

_HERE = Path(__file__).resolve().parent


def build_solver(
    *,
    config=None,
    seed: int = 7,
    n_samples: int = 240,
    valid_ratio: float = 0.2,
    learning_rate: float = 0.05,
    l2: float = 1e-4,
    complexity_weight: float = 0.0,
    max_components: int = 5,
    energy_threshold: float = 0.999,
    resource_context: dict | None = None,
    component_overrides: dict | None = None,
):
    """Assemble and return the inner trainer using the canonical case entry."""

    del config
    overrides = dict(component_overrides or {})
    pipeline_builder = overrides.pop("pipeline", build_pipeline)
    trainer_builder = overrides.pop(
        "trainer",
        build_orthogonal_linear_point_trainer,
    )
    if overrides:
        raise ValueError(
            "unsupported orthogonal_point_demo component overrides: "
            f"{sorted(overrides)}"
        )
    data = (
        pipeline_builder(
            seed=int(seed),
            n_samples=int(n_samples),
            valid_ratio=float(valid_ratio),
            feature_names=("x1", "x2"),
        )
        if callable(pipeline_builder)
        else pipeline_builder
    )
    trainer = trainer_builder(
        data,
        learning_rate=float(learning_rate),
        l2=float(l2),
        complexity_weight=float(complexity_weight),
        max_components=max_components,
        energy_threshold=energy_threshold,
        run_name="demo_orthogonal_linear_point",
    )
    if resource_context:
        trainer.set_resource_context(dict(resource_context))
    return trainer


def _build_parser():
    parser = argparse.ArgumentParser(
        description="Build and run the orthogonal point demo scaffold.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Build and validate assembly only; do not run fit().",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=300,
        help="Training steps used when running the demo.",
    )
    return parser


def main(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)
    trainer = build_solver()
    if bool(args.check):
        problem = getattr(trainer, "problem", None)
        rep = getattr(trainer, "representation", None)
        adapter = getattr(trainer, "adapter", None)
        print(
            "[check] assembly ok | "
            f"problem={type(problem).__name__ if problem else 'None'} | "
            f"representation={type(rep).__name__ if rep else 'None'} | "
            f"adapter={type(adapter).__name__ if adapter else 'None'}"
        )
        return
    result = trainer.fit(max_steps=int(args.max_steps))
    print(trainer.build_report())
    print(result.report)


if __name__ == "__main__":
    main()
