"""Canonical data-pipeline entry for temporal model comparison."""

from .data_generator import build_data_view, create_lag_features, generate_synthetic_series


def build_pipeline(**kwargs):
    """Build the temporal train/validation DataView."""

    return build_data_view(**kwargs)


__all__ = ["build_data_view", "build_pipeline", "create_lag_features", "generate_synthetic_series"]
