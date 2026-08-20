from __future__ import annotations

from typing import Any, Mapping

from mlblack.presets import build_tiny_transformer_lm_trainer

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
    if overrides:
        raise ValueError(
            "unsupported tiny Transformer language-model overrides: "
            f"{sorted(overrides)}"
        )
    data = data_builder() if callable(data_builder) else data_builder
    return trainer_builder(
        data,
        vocab_size=12,
        max_length=5,
        hidden_dim=8,
        num_layers=1,
        num_heads=2,
        ffn_kind="swiglu",
        norm="rms_norm",
        position_encoding="rope",
        learning_rate=1e-2,
        random_seed=9,
        run_name="tiny_transformer_lm_case",
        resource_context=resource_context,
    )


__all__ = ["build_solver"]
