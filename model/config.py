from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class InnerOptConfig:
    enabled: bool = True
    adam_steps: int = 80
    adam_lr: float = 1e-2
    lbfgs_steps: int = 25
    lbfgs_lr: float = 0.8
    accept_rmse_tol: float = 0.0
    accept_rel_tol: float = 0.01
    guard_patience: int = 3
    guard_check_interval: int = 10
    alt_freeze_readout: bool = True
    grad_clip_norm: float = 1.0
    residual_clip_q: float = 0.98


@dataclass(frozen=True)
class IntervalConfig:
    alpha: float = 0.1
    method: str = "native_quantile_cqr"
    calib_ratio: float = 0.2
    quantile_l2: float = 1e-4


@dataclass(frozen=True)
class XgboostBaselineConfig:
    n_estimators: int = 360
    max_depth: int = 6
    learning_rate: float = 0.05
    subsample: float = 0.9
    colsample_bytree: float = 0.9
    tree_method: str = "hist"
    random_seed: int = 42


@dataclass(frozen=True)
class ModelConfig:
    inner_opt: InnerOptConfig = InnerOptConfig()
    interval: IntervalConfig = IntervalConfig()
    xgb: XgboostBaselineConfig = XgboostBaselineConfig()


def as_three_layer_kwargs(inner_cfg: InnerOptConfig) -> dict[str, float | int | bool]:
    return {
        "inner_opt_enabled": bool(inner_cfg.enabled),
        "inner_opt_adam_steps": int(max(0, inner_cfg.adam_steps)),
        "inner_opt_adam_lr": float(max(1e-8, inner_cfg.adam_lr)),
        "inner_opt_lbfgs_steps": int(max(0, inner_cfg.lbfgs_steps)),
        "inner_opt_lbfgs_lr": float(max(1e-8, inner_cfg.lbfgs_lr)),
        "inner_opt_accept_rmse_tol": float(max(0.0, inner_cfg.accept_rmse_tol)),
        "inner_opt_accept_rel_tol": float(max(0.0, inner_cfg.accept_rel_tol)),
        "inner_opt_guard_patience": int(max(1, inner_cfg.guard_patience)),
        "inner_opt_guard_check_interval": int(max(1, inner_cfg.guard_check_interval)),
        "inner_opt_alt_freeze_readout": bool(inner_cfg.alt_freeze_readout),
        "inner_opt_grad_clip_norm": float(max(0.0, inner_cfg.grad_clip_norm)),
        "inner_opt_residual_clip_q": float(inner_cfg.residual_clip_q),
    }


__all__ = [
    "InnerOptConfig",
    "IntervalConfig",
    "XgboostBaselineConfig",
    "ModelConfig",
    "as_three_layer_kwargs",
]
