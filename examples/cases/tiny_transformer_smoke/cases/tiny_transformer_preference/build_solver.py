from __future__ import annotations

from typing import Any, Mapping

from mlblack.presets import build_tiny_transformer_dpo_preference_trainer

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
    trainer_builder = overrides.pop(
        "trainer",
        build_tiny_transformer_dpo_preference_trainer,
    )
    if overrides:
        raise ValueError(
            "unsupported tiny Transformer preference overrides: "
            f"{sorted(overrides)}"
        )
    data = data_builder() if callable(data_builder) else data_builder
    return trainer_builder(
        data,
        vocab_size=10,
        max_length=4,
        hidden_dim=8,
        num_layers=1,
        num_heads=2,
        qlora={"rank": 2, "bits": 4, "targets": ("attention.q", "attention.v")},
        learning_rate=1e-2,
        random_seed=21,
        run_name="tiny_transformer_dpo_case",
        resource_context=resource_context,
    )


__all__ = ["build_solver"]
