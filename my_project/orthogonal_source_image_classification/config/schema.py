from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ImageClassificationConfig:
    dataset_keys: tuple[str, ...] = ("digits",)
    train_ratio: float = 0.75
    seed: int = 42
    representation_max_features: int = 55
    representation_candidate_keep_top: int = 120
    representation_max_pair_abs_corr: float = 0.985
    max_sources: int = 16
    candidate_keep_top: int = 220
    max_pair_abs_corr: float = 0.76
    max_rows: int = 0
    output_dir: str = "runs/orthogonal_source_image_classification"


__all__ = ["ImageClassificationConfig"]
