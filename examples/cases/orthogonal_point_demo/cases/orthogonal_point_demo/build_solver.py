# -*- coding: utf-8 -*-
"""Standard assembly entry for the orthogonal point demo case."""

from __future__ import annotations

from mlblack.presets import build_orthogonal_linear_point_trainer

from pipeline.main import build_pipeline

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
    if resource_context is not None:
        trainer.set_resource_context(dict(resource_context))
    return trainer
