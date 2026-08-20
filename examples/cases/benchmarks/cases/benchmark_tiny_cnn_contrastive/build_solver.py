from __future__ import annotations

from typing import Any, Mapping

from mlblack.presets import build_tiny_cnn_image_contrastive_trainer
from nsgablack.core import BudgetController

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
    trainer_builder = overrides.pop("trainer", build_tiny_cnn_image_contrastive_trainer)
    max_steps = overrides.pop("max_steps", None)
    if overrides:
        raise ValueError("unsupported benchmark_tiny_cnn_contrastive overrides: " + str(sorted(overrides)))
    data = data_builder() if callable(data_builder) else data_builder
    trainer = trainer_builder(
        data,
        conv_channels=(4,), embedding_dim=4, learning_rate=1e-2, random_seed=55,
        run_name="benchmark_tiny_cnn_contrastive",
        resource_context=resource_context,
    )
    if max_steps is not None:
        trainer.register_controller(
            BudgetController(
                max_generations=max(1, int(max_steps)),
                name="benchmark_tiny_cnn_contrastive.step_limit",
            )
        )
    return trainer


__all__ = ["build_solver"]
