from __future__ import annotations

from dataclasses import dataclass, field

from core.symbolic.feature_space.regime_router import RegimePolicy, Strict4RouterSpec


@dataclass(frozen=True)
class IntervalCallbackConfig:
    interval_alpha: float
    interval_method: str
    interval_calib_ratio: float
    interval_quantile_l2: float
    regime_branch_mode: bool
    regime_gate_idx: tuple[int, int, int, int] | None
    base_regime_min_branch_train: int
    regime_branch_parallel_workers: int
    regime_policy: RegimePolicy = field(default_factory=Strict4RouterSpec)

    @property
    def strict4_branch_mode(self) -> bool:
        return self.regime_branch_mode

    @property
    def strict4_gate_idx(self) -> tuple[int, int, int, int] | None:
        return self.regime_gate_idx

    @property
    def base_strict4_min_branch_train(self) -> int:
        return self.base_regime_min_branch_train

    @property
    def strict4_branch_parallel_workers(self) -> int:
        return self.regime_branch_parallel_workers

    @property
    def strict4_regime_policy(self) -> RegimePolicy:
        return self.regime_policy

    @property
    def strict4_router_spec(self) -> RegimePolicy:
        return self.regime_policy


@dataclass(frozen=True)
class FitPredictCallbackConfig:
    random_seed: int | None
    inner_opt_enabled: bool
    inner_opt_adam_steps: int
    inner_opt_adam_lr: float
    inner_opt_lbfgs_steps: int
    inner_opt_lbfgs_lr: float
    inner_opt_accept_rmse_tol: float
    inner_opt_accept_rel_tol: float
    inner_opt_guard_patience: int
    inner_opt_guard_check_interval: int
    inner_opt_alt_freeze_readout: bool
    inner_opt_grad_clip_norm: float
    inner_opt_residual_clip_q: float


__all__ = [
    "IntervalCallbackConfig",
    "FitPredictCallbackConfig",
]
