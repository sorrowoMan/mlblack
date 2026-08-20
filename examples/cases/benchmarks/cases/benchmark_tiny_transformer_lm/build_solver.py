from __future__ import annotations

from typing import Any, Mapping

from mlblack.presets import build_tiny_transformer_lm_trainer
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
    trainer_builder = overrides.pop("trainer", build_tiny_transformer_lm_trainer)
    max_steps = overrides.pop("max_steps", None)
    if overrides:
        raise ValueError("unsupported benchmark_tiny_transformer_lm overrides: " + str(sorted(overrides)))
    data = data_builder() if callable(data_builder) else data_builder
    trainer = trainer_builder(
        data,
        vocab_size=12, max_length=5, hidden_dim=8, num_layers=1, num_heads=2, learning_rate=1e-2, random_seed=57,
        run_name="benchmark_tiny_transformer_lm",
        resource_context=resource_context,
    )
    if max_steps is not None:
        trainer.register_controller(
            BudgetController(
                max_generations=max(1, int(max_steps)),
                name="benchmark_tiny_transformer_lm.step_limit",
            )
        )
    return trainer


__all__ = ["build_solver"]
