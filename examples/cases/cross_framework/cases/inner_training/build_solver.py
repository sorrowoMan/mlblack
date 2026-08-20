from __future__ import annotations

from typing import Any, Mapping

from mlblack.presets import build_orthogonal_linear_point_trainer

from pipeline.main import build_data


def build_solver(
    config=None,
    *,
    resource_context: Mapping[str, Any] | None = None,
    component_overrides: Mapping[str, Any] | None = None,
):
    del config
    overrides = dict(component_overrides or {})
    data_builder = overrides.pop("pipeline", build_data)
    trainer_builder = overrides.pop("trainer", build_orthogonal_linear_point_trainer)
    learning_rate = float(overrides.pop("learning_rate", 0.05))
    max_steps = int(overrides.pop("max_steps", 2))
    if overrides:
        raise ValueError(f"unsupported inner Trainer overrides: {sorted(overrides)}")
    data = data_builder() if callable(data_builder) else data_builder
    trainer = trainer_builder(
        data,
        learning_rate=learning_rate,
        l2=1e-4,
        run_name="cross_framework_inner_training",
    )
    trainer.max_steps = max(1, max_steps)
    if resource_context is not None:
        trainer.set_resource_context(resource_context)
    return trainer


__all__ = ["build_solver"]
