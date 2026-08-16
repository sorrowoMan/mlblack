"""Canonical synthetic-data pipeline for one temporal forecast Trainer."""

from __future__ import annotations

from collections.abc import Mapping

from mlblack.pipeline.synthetic_temporal import build_sine_forecast_data_view


def build_pipeline(*, config=None, resource_context=None, component_overrides=None):
    del resource_context
    payload = dict(config or {}) if isinstance(config, Mapping) else {}
    overrides = dict(component_overrides or {})
    if overrides.get("data") is not None:
        return overrides["data"]
    return build_sine_forecast_data_view(
        n_train=int(payload.get("n_train", 200)),
        n_valid=int(payload.get("n_valid", 50)),
        sequence_length=int(payload.get("sequence_length", 12)),
        noise_std=float(payload.get("noise_std", 0.15)),
        random_seed=int(payload.get("random_seed", 42)),
    )


def run_pipeline_slot(*args, **kwargs):
    del args
    return build_pipeline(**kwargs)


__all__ = ["build_pipeline", "run_pipeline_slot"]

