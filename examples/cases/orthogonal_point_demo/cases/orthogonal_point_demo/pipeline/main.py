"""Canonical data-pipeline entry for the orthogonal point demo."""

from .example_pipeline import build_data_view, build_orthogonal_point_demo_data_view


def build_pipeline(**kwargs):
    """Build the train/validation DataView consumed by the Case."""

    return build_orthogonal_point_demo_data_view(**kwargs)


__all__ = ["build_data_view", "build_orthogonal_point_demo_data_view", "build_pipeline"]
