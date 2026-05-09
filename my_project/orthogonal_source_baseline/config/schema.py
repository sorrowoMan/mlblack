from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OrthogonalSourceBaselineConfig:
    benchmark_keys: tuple[str, ...] = (
        "ohm_like",
        "ideal_gas_like",
        "arrhenius_gate_like",
        "periodic_gate_like",
        "redundant_proxy_control",
        "coupled_reaction_transport_like",
    )
    n_total: int = 360
    train_ratio: float = 0.8
    noise_std: float = 0.025
    seed: int = 42
    max_sources: int = 10
    max_pair_abs_corr: float = 0.72
    max_rows: int = 60000
    output_dir: str = "runs/orthogonal_source_baseline"


__all__ = ["OrthogonalSourceBaselineConfig"]
