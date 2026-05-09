from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EtfQuantIntervalConfig:
    """Configuration for the ETF interval proxy scaffold."""

    dataset_url: str = (
        "https://huggingface.co/datasets/P2SAMAPA/p2-etf-rough-path-forecaster-results/"
        "resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet"
    )
    hf_repo_id: str = "P2SAMAPA/p2-etf-rough-path-forecaster-results"
    hf_filename: str = "default/train/0000.parquet"
    hf_revision: str = "refs/convert/parquet"
    cache_dir: str = "runs/etf_quant_interval_proxy/cache"
    allow_public_sample_fallback: bool = True
    dataset_label: str = "p2samapa_etf_rough_path_returns"
    horizon: int = 5
    lookback: int = 20
    train_ratio: float = 0.72
    max_rows: int = 0
    seed: int = 42
    interval_alpha: float = 0.10
    max_sources: int = 28
    candidate_keep_top: int = 180
    max_pair_abs_corr: float = 0.68
    min_abs_target_corr: float = 0.01
    rolling_window: int = 40
    output_dir: str = "runs/etf_quant_interval_proxy"


__all__ = ["EtfQuantIntervalConfig"]
