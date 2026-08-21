"""Canonical assembly for one PatchTST temporal forecast Trainer."""

from __future__ import annotations

from collections.abc import Mapping

from blackbase.resources import coerce_resource_context
from mlblack.presets.neural import build_temporal_patchtst_forecast_trainer

try:
    from .pipeline import build_pipeline
except ImportError:  # direct script execution
    from pipeline import build_pipeline


def build_solver(config=None, *, resource_context=None, component_overrides=None):
    overrides = dict(component_overrides or {})
    payload = dict(config or {}) if isinstance(config, Mapping) else {}
    payload.update(dict(overrides.pop("config", {}) or {}))
    max_steps = overrides.pop("max_steps", None)
    grant = coerce_resource_context(resource_context)
    data = overrides.pop("data", None)
    if data is None:
        data = build_pipeline(config=payload, resource_context=grant, component_overrides={})
    builder = overrides.pop("trainer_builder", None) or build_temporal_patchtst_forecast_trainer
    trainer_kwargs = dict(overrides.pop("trainer_kwargs", {}) or {})
    if overrides:
        raise ValueError("unsupported temporal_patchtst overrides: " + str(sorted(overrides)))
    trainer_kwargs.setdefault("input_dim", 1)
    trainer_kwargs.setdefault("sequence_length", int(payload.get("sequence_length", 12)))
    trainer_kwargs.setdefault("output_dim", 1)
    trainer_kwargs.setdefault("device", str(grant.device or "cpu"))
    trainer_kwargs.setdefault("random_seed", int(payload.get("random_seed", 42)))
    trainer_kwargs.setdefault("run_name", "temporal_patchtst_compare")
    trainer_kwargs.setdefault("resource_context", grant)
    trainer = builder(data, **trainer_kwargs)
    if max_steps is not None:
        trainer.set_max_steps(max(1, int(max_steps)))
    return trainer
