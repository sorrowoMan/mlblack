# -*- coding: utf-8 -*-
"""Temporal neural forecast comparison: 7 models on synthetic sine wave data.

Pure framework preset composition -- zero custom ML components.
"""

from __future__ import annotations

import time
import numpy as np

from mlblack.presets.neural import (
    build_temporal_lstm_forecast_trainer,
    build_temporal_tcn_forecast_trainer,
    build_temporal_transformer_forecast_trainer,
    build_temporal_nbeats_forecast_trainer,
    build_temporal_deepar_forecast_trainer,
    build_temporal_patchtst_forecast_trainer,
    build_temporal_tft_forecast_trainer,
)

from pipeline.data_generator import build_data_view

PRESET_BUILDERS = {
    "LSTM": build_temporal_lstm_forecast_trainer,
    "TCN": build_temporal_tcn_forecast_trainer,
    "Transformer": build_temporal_transformer_forecast_trainer,
    "N-BEATS": build_temporal_nbeats_forecast_trainer,
    "DeepAR": build_temporal_deepar_forecast_trainer,
    "PatchTST": build_temporal_patchtst_forecast_trainer,
    "TFT": build_temporal_tft_forecast_trainer,
}

TRAIN_STEPS = 50
SEQUENCE_LENGTH = 12


def build_solver():
    """Canonical unified scaffold entry; returns the configured trainer set."""

    data = build_data_view(
        n_train=200,
        n_val=50,
        seq_len=SEQUENCE_LENGTH,
        noise_std=0.15,
        random_seed=42,
    )
    return {
        name: builder(
            data,
            input_dim=1,
            sequence_length=SEQUENCE_LENGTH,
            output_dim=1,
            device="cpu",
            random_seed=42,
        )
        for name, builder in PRESET_BUILDERS.items()
    }


def _extract_rmse(trainer):
    if hasattr(trainer, "feedback") and trainer.feedback:
        fb = trainer.feedback[0]
        metrics = dict(getattr(fb, "metrics", {}))
        for key in ("valid.rmse", "train.rmse"):
            if key in metrics:
                return metrics[key]
    if hasattr(trainer, "best_feedback") and trainer.best_feedback is not None:
        obj = np.asarray(trainer.best_feedback.objectives, dtype=float).ravel()
        if len(obj):
            return float(np.sqrt(obj[0]))
    return float("nan")


def main():
    data = build_data_view(
        n_train=200,
        n_val=50,
        seq_len=SEQUENCE_LENGTH,
        noise_std=0.15,
        random_seed=42,
    )

    results = []
    for name, builder in PRESET_BUILDERS.items():
        print(f"\n{'='*50}")
        print(f"  {name}")
        print(f"{'='*50}")
        t0 = time.perf_counter()
        try:
            trainer = builder(
                data,
                input_dim=1,
                sequence_length=SEQUENCE_LENGTH,
                output_dim=1,
                device="cpu",
                random_seed=42,
            )
            result = trainer.fit(max_steps=TRAIN_STEPS)
            elapsed = time.perf_counter() - t0
            rmse = _extract_rmse(trainer)
            print(f"  RMSE: {rmse:.6f}   Time: {elapsed:.2f}s")
            results.append((name, rmse, elapsed))
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            print(f"  FAILED: {exc}")
            results.append((name, None, elapsed))

    _print_table(results)


def _print_table(results):
    print(f"\n{'='*70}")
    print("  Temporal Neural Forecast Comparison")
    print(f"{'='*70}")
    print(f"  Data: synthetic sin(0.1*t) + N(0, 0.15^2), {TRAIN_STEPS} steps")
    print(f"  {'Model':<16} {'RMSE':>10} {'Time (s)':>10}")
    print(f"  {'-'*16} {'-'*10} {'-'*10}")
    for name, rmse, elapsed in results:
        rmse_str = f"{rmse:.6f}" if rmse is not None else "FAILED"
        print(f"  {name:<16} {rmse_str:>10} {elapsed:>10.2f}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
