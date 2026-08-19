"""Compatibility assembly: the historical comparison Case now builds one LSTM Trainer."""

from __future__ import annotations

from collections.abc import Mapping

from blackbase.resources import coerce_resource_context
from mlblack.presets.neural import build_temporal_lstm_forecast_trainer

try:
    from .pipeline import build_pipeline
except ImportError:  # direct script execution
    from pipeline import build_pipeline


def build_solver(config=None, *, resource_context=None, component_overrides=None):
    payload = dict(config or {}) if isinstance(config, Mapping) else {}
    overrides = dict(component_overrides or {})
    grant = coerce_resource_context(resource_context)
    sequence_length = int(payload.get("sequence_length", 12))
    data = overrides.get("data")
    if data is None:
        data = build_pipeline(
            n_train=int(payload.get("n_train", 200)),
            n_val=int(payload.get("n_valid", 50)),
            seq_len=sequence_length,
            noise_std=float(payload.get("noise_std", 0.15)),
            random_seed=int(payload.get("random_seed", 42)),
        )
    builder = overrides.get("trainer_builder") or build_temporal_lstm_forecast_trainer
    trainer_kwargs = dict(overrides.get("trainer_kwargs", {}) or {})
    trainer_kwargs.setdefault("input_dim", 1)
    trainer_kwargs.setdefault("sequence_length", sequence_length)
    trainer_kwargs.setdefault("output_dim", 1)
    trainer_kwargs.setdefault("device", str(grant.device or "cpu"))
    trainer_kwargs.setdefault("random_seed", int(payload.get("random_seed", 42)))
    trainer_kwargs.setdefault("run_name", "temporal_neural_compare_compat_lstm")
    trainer_kwargs.setdefault("resource_context", grant)
    return builder(data, **trainer_kwargs)
