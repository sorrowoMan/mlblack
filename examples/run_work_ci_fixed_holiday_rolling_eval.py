from __future__ import annotations

import argparse
import concurrent.futures
import json
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.run_work_ci_fixed_holiday_piecewise_demo import (  # noqa: E402
    _build_regime_index,
    _col_index,
    _fit_artifact,
    _gate_key,
    _metrics,
    _select_training_indices_for_regime,
    _slice_cols,
)
from examples.path_defaults import default_work_ci_csv


STRICT4_REGIME_ORDER: tuple[tuple[int, int, int, int], ...] = (
    (1, 1, 0, 0),  # holiday day/window (near)
    (1, 0, 1, 0),  # holiday day/window (mid)
    (0, 0, 0, 1),  # non-work weekend
    (0, 0, 0, 0),  # regular day
)
STRICT3_REGIME_ORDER: tuple[tuple[int, int, int, int], ...] = (
    (1, 1, 0, 0),  # holiday_any (near+mid merged)
    (0, 0, 0, 1),  # non-work weekend
    (0, 0, 0, 0),  # regular day
)
STRICT3_REGIME_ALIAS: dict[tuple[int, int, int, int], str] = {
    (1, 1, 0, 0): "holiday_any",
    (0, 0, 0, 1): "weekend",
    (0, 0, 0, 0): "regular",
}
STRICT4_REGIME_ALIAS: dict[tuple[int, int, int, int], str] = {
    (1, 1, 0, 0): "holiday_near",
    (1, 0, 1, 0): "holiday_mid",
    (0, 0, 0, 1): "weekend",
    (0, 0, 0, 0): "regular",
}
STRICT4_ALIAS_TO_REGIME: dict[str, tuple[int, int, int, int]] = {
    v: k for k, v in STRICT4_REGIME_ALIAS.items()
}
STRICT4_PARALLEL_MODES: frozenset[str] = frozenset({"serial", "thread", "process"})
STRICT4_GPU_STRATEGIES: frozenset[str] = frozenset({"none", "fixed", "round_robin", "auto"})
STRICT4_BRANCH_OVERRIDE_FIELDS: tuple[str, ...] = (
    "local_search_force_linear_base",
    "local_search_topk_features",
    "local_search_max_added_terms",
    "local_search_max_pair_terms",
    "local_search_max_candidates_per_iter",
    "local_search_candidate_keep_top",
    "local_search_ridge_l2",
    "local_search_unary_ops",
    "local_search_nested_unary_patterns",
    "local_search_overfit_guard_enabled",
    "local_search_overfit_guard_val_ratio",
    "local_search_overfit_guard_min_val_samples",
    "local_search_overfit_guard_min_val_rmse_gain",
    "local_search_overfit_guard_max_gap_increase",
    "local_search_overfit_guard_patience",
    "local_search_interaction_budget_mode",
    "local_search_interaction_diag_threshold",
    "local_search_interaction_diag_topk_features",
    "local_search_interaction_pair_budget_boost",
    "local_search_interaction_grad_projection_budget_boost",
    "local_search_inner_opt_enabled",
    "local_search_inner_opt_method",
    "local_search_inner_opt_device",
    "local_search_inner_opt_adam_steps",
    "local_search_inner_opt_adam_lr",
    "local_search_inner_opt_lbfgs_steps",
    "local_search_inner_opt_lbfgs_lr",
    "local_search_inner_opt_l2",
    "local_search_inner_opt_accept_rmse_tol",
    "blend_kappa",
)


def _jsonable(v: Any) -> Any:
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, Mapping):
        return {str(k): _jsonable(val) for k, val in v.items()}
    if isinstance(v, (list, tuple, set)):
        return [_jsonable(x) for x in v]
    return str(v)


def _parse_csv_list(raw: str) -> tuple[str, ...]:
    txt = str(raw).strip()
    if not txt:
        return tuple()
    return tuple(p.strip() for p in txt.split(",") if p.strip())


def _normalize_parallel_mode(raw: str) -> str:
    key = str(raw or "serial").strip().lower()
    if key not in STRICT4_PARALLEL_MODES:
        raise ValueError(
            f"Unsupported strict4 parallel mode: '{raw}'. Allowed={sorted(STRICT4_PARALLEL_MODES)}"
        )
    return key


def _normalize_gpu_strategy(raw: str) -> str:
    key = str(raw or "none").strip().lower()
    if key not in STRICT4_GPU_STRATEGIES:
        raise ValueError(
            f"Unsupported strict4 gpu strategy: '{raw}'. Allowed={sorted(STRICT4_GPU_STRATEGIES)}"
        )
    return key


def _normalize_cuda_device_token(raw: str) -> str:
    txt = str(raw).strip().lower()
    if not txt:
        raise ValueError("strict4 gpu device token must not be empty")
    if txt in {"cpu", "cuda"}:
        return txt
    if txt.isdigit():
        return f"cuda:{int(txt)}"
    if txt.startswith("gpu:") and txt.split(":", 1)[1].isdigit():
        return f"cuda:{int(txt.split(':', 1)[1])}"
    if txt.startswith("cuda:") and txt.split(":", 1)[1].isdigit():
        return f"cuda:{int(txt.split(':', 1)[1])}"
    raise ValueError(f"Unsupported strict4 gpu device token: '{raw}'")


def _parse_gpu_devices(raw: str) -> tuple[str, ...]:
    tokens = _parse_csv_list(raw)
    out: list[str] = []
    for item in tokens:
        tok = _normalize_cuda_device_token(item)
        if tok not in out:
            out.append(tok)
    return tuple(out)


def _discover_cuda_devices() -> tuple[str, ...]:
    try:
        import torch
    except Exception:
        return tuple()

    try:
        if not bool(torch.cuda.is_available()):
            return tuple()
        n = int(torch.cuda.device_count())
    except Exception:
        return tuple()

    if n <= 0:
        return tuple()
    return tuple(f"cuda:{i}" for i in range(n))


def _resolve_parallel_workers(*, n_tasks: int, requested: int) -> int:
    if n_tasks <= 1:
        return 1
    req = int(max(1, requested))
    return int(min(req, n_tasks))


def _select_gpu_device_for_branch(
    *,
    strategy: str,
    configured_devices: tuple[str, ...],
    branch_order: int,
) -> str | None:
    key = str(strategy).strip().lower()
    pool = tuple(configured_devices)

    if key == "none":
        return None
    if key == "fixed":
        return str(pool[0]) if pool else "cuda:0"
    if key in {"round_robin", "auto"}:
        effective_pool = pool if pool else _discover_cuda_devices()
        if not effective_pool:
            return None
        return str(effective_pool[int(branch_order) % len(effective_pool)])
    raise ValueError(f"Unsupported strict4 gpu strategy: {strategy}")


def _fit_local_regime_task(
    *,
    trainer_params: Mapping[str, Any],
    X_slice: np.ndarray,
    y_slice: np.ndarray,
    feature_names: tuple[str, ...],
):
    return _fit_artifact(
        trainer_key="symbolic_stagewise",
        trainer_params=dict(trainer_params),
        X=np.asarray(X_slice, dtype=float),
        y=np.asarray(y_slice, dtype=float),
        feature_names=tuple(feature_names),
    )


def _safe_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(np.mean(np.asarray(values, dtype=float)))


def _safe_std(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(np.std(np.asarray(values, dtype=float), ddof=0))


def _safe_median(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(np.median(np.asarray(values, dtype=float)))


def _time_val_split_indices(
    *,
    n_samples: int,
    val_ratio: float,
    min_val_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    n = int(max(0, n_samples))
    if n <= 2:
        return np.asarray([], dtype=int), np.asarray([], dtype=int)
    n_val = int(max(int(min_val_samples), int(round(float(val_ratio) * float(n)))))
    n_val = min(n_val, n - 1)
    n_fit = n - n_val
    if n_fit <= 0 or n_val <= 0:
        return np.asarray([], dtype=int), np.asarray([], dtype=int)
    idx_fit = np.arange(0, n_fit, dtype=int)
    idx_val = np.arange(n_fit, n, dtype=int)
    return idx_fit, idx_val


@dataclass(frozen=True)
class RollingSpec:
    gate_features: tuple[str, ...]
    param_features: tuple[str, ...]
    regime_mode: str
    min_leaf: int
    merge_rare_holiday_regimes: bool
    blend_with_global: bool
    blend_kappa: float
    local_search_force_linear_base: str
    local_search_topk_features: int
    local_search_max_added_terms: int
    local_search_max_pair_terms: int
    local_search_max_candidates_per_iter: int
    local_search_candidate_keep_top: int
    local_search_ridge_l2: float
    local_search_unary_ops: tuple[str, ...]
    local_search_nested_unary_patterns: tuple[str, ...]
    local_search_overfit_guard_enabled: bool
    local_search_overfit_guard_val_ratio: float
    local_search_overfit_guard_min_val_samples: int
    local_search_overfit_guard_min_val_rmse_gain: float
    local_search_overfit_guard_max_gap_increase: float
    local_search_overfit_guard_patience: int
    local_search_interaction_budget_mode: str
    local_search_interaction_diag_threshold: float
    local_search_interaction_diag_topk_features: int
    local_search_interaction_pair_budget_boost: float
    local_search_interaction_grad_projection_budget_boost: float
    small_sample_guard_threshold: int
    blend_global_backbone_mode: str
    blend_global_backbone_val_ratio: float
    blend_global_backbone_min_val_samples: int
    blend_global_backbone_margin: float
    local_search_inner_opt_enabled: bool
    local_search_inner_opt_method: str
    local_search_inner_opt_device: str
    local_search_inner_opt_adam_steps: int
    local_search_inner_opt_adam_lr: float
    local_search_inner_opt_lbfgs_steps: int
    local_search_inner_opt_lbfgs_lr: float
    local_search_inner_opt_l2: float
    local_search_inner_opt_accept_rmse_tol: float
    strict4_parallel_mode: str
    strict4_max_workers: int
    strict4_gpu_strategy: str
    strict4_gpu_devices: tuple[str, ...]
    strict4_branch_hparams: dict[str, dict[str, Any]]
    strict4_dynamic_merge_enabled: bool
    strict4_dynamic_merge_min_samples: int


def _hamming_fixed4(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    return int(sum(int(x != y) for x, y in zip(a, b)))


def _normalize_fixed4_key(raw_key: tuple[int, ...]) -> tuple[int, int, int, int]:
    if len(raw_key) >= 4:
        return tuple(int(v > 0) for v in raw_key[:4])  # type: ignore[return-value]
    padded = list(int(v > 0) for v in raw_key)
    while len(padded) < 4:
        padded.append(0)
    return tuple(padded[:4])  # type: ignore[return-value]


def _map_to_strict4_regime(raw_key: tuple[int, ...]) -> tuple[int, int, int, int]:
    k = _normalize_fixed4_key(raw_key)
    if k in STRICT4_REGIME_ORDER:
        return k
    # Hard semantic routing before nearest-neighbor fallback.
    if k[0] > 0 and k[1] > 0:
        return (1, 1, 0, 0)
    if k[0] > 0 and k[2] > 0:
        return (1, 0, 1, 0)
    if k[3] > 0:
        return (0, 0, 0, 1)
    if k[0] == 0 and k[1] == 0 and k[2] == 0:
        return (0, 0, 0, 0)
    return min(STRICT4_REGIME_ORDER, key=lambda proto: _hamming_fixed4(proto, k))


def _map_keys_strict4(keys: tuple[tuple[int, ...], ...]) -> tuple[tuple[int, int, int, int], ...]:
    return tuple(_map_to_strict4_regime(k) for k in keys)


def _map_to_strict3_regime(raw_key: tuple[int, ...]) -> tuple[int, int, int, int]:
    k = _normalize_fixed4_key(raw_key)
    # any holiday signal -> holiday_any
    if k[0] > 0 or k[1] > 0 or k[2] > 0:
        return (1, 1, 0, 0)
    if k[3] > 0:
        return (0, 0, 0, 1)
    return (0, 0, 0, 0)


def _map_keys_strict3(keys: tuple[tuple[int, ...], ...]) -> tuple[tuple[int, int, int, int], ...]:
    return tuple(_map_to_strict3_regime(k) for k in keys)


def _strict4_alias_for_regime_key(regime_key: tuple[int, ...]) -> str:
    k = _normalize_fixed4_key(regime_key)
    return str(STRICT4_REGIME_ALIAS.get(k, ",".join(map(str, k))))


def _parse_strict4_alias_from_text(raw_key: str) -> str | None:
    txt = str(raw_key).strip().lower()
    if txt in STRICT4_ALIAS_TO_REGIME:
        return txt
    cleaned = (
        txt.replace("(", "")
        .replace(")", "")
        .replace("[", "")
        .replace("]", "")
        .replace(" ", "")
    )
    if not cleaned:
        return None
    parts = [p for p in cleaned.split(",") if p != ""]
    if len(parts) != 4:
        return None
    try:
        vals = tuple(int(int(p) > 0) for p in parts)
    except Exception:
        return None
    return STRICT4_REGIME_ALIAS.get(vals)  # type: ignore[arg-type]


def _normalize_ops_like(value: Any, *, field: str) -> tuple[str, ...]:
    if isinstance(value, str):
        return tuple(_parse_csv_list(value))
    if isinstance(value, (list, tuple, set)):
        out: list[str] = []
        for item in value:
            s = str(item).strip()
            if s:
                out.append(s)
        return tuple(out)
    raise ValueError(f"Field '{field}' must be csv string or list/tuple/set.")


def _parse_strict4_branch_hparams_json(raw: str) -> dict[str, dict[str, Any]]:
    txt = str(raw).strip()
    if not txt:
        return {}
    payload = txt
    path = Path(txt).expanduser()
    if path.exists() and path.is_file():
        payload = path.read_text(encoding="utf-8-sig")
    obj = json.loads(payload)
    if not isinstance(obj, Mapping):
        raise ValueError("strict4 branch hparams json must be an object/dict.")

    out: dict[str, dict[str, Any]] = {}
    for raw_key, raw_cfg in obj.items():
        alias = _parse_strict4_alias_from_text(str(raw_key))
        if alias is None:
            raise ValueError(
                f"Unknown strict4 branch key '{raw_key}'. Use one of "
                f"{sorted(STRICT4_ALIAS_TO_REGIME.keys())} or tuple like '(1,1,0,0)'."
            )
        if not isinstance(raw_cfg, Mapping):
            raise ValueError(f"strict4 branch '{raw_key}' config must be a dict/object.")
        cfg: dict[str, Any] = {}
        unknown = sorted(set(str(k) for k in raw_cfg.keys()) - set(STRICT4_BRANCH_OVERRIDE_FIELDS))
        if unknown:
            raise ValueError(
                f"strict4 branch '{raw_key}' has unsupported fields: {unknown}. "
                f"Allowed={list(STRICT4_BRANCH_OVERRIDE_FIELDS)}"
            )
        if "local_search_force_linear_base" in raw_cfg:
            linear_mode = str(raw_cfg["local_search_force_linear_base"]).strip().lower()
            if linear_mode not in {"auto", "on", "off"}:
                raise ValueError(
                    f"strict4 branch '{raw_key}' has invalid local_search_force_linear_base='{linear_mode}', "
                    "expected auto|on|off."
                )
            cfg["local_search_force_linear_base"] = linear_mode
        if "local_search_topk_features" in raw_cfg:
            cfg["local_search_topk_features"] = int(max(1, int(raw_cfg["local_search_topk_features"])))
        if "local_search_max_added_terms" in raw_cfg:
            cfg["local_search_max_added_terms"] = int(max(0, int(raw_cfg["local_search_max_added_terms"])))
        if "local_search_max_pair_terms" in raw_cfg:
            cfg["local_search_max_pair_terms"] = int(max(0, int(raw_cfg["local_search_max_pair_terms"])))
        if "local_search_max_candidates_per_iter" in raw_cfg:
            cfg["local_search_max_candidates_per_iter"] = int(max(1, int(raw_cfg["local_search_max_candidates_per_iter"])))
        if "local_search_candidate_keep_top" in raw_cfg:
            cfg["local_search_candidate_keep_top"] = int(max(1, int(raw_cfg["local_search_candidate_keep_top"])))
        if "local_search_ridge_l2" in raw_cfg:
            cfg["local_search_ridge_l2"] = float(max(0.0, float(raw_cfg["local_search_ridge_l2"])))
        if "local_search_unary_ops" in raw_cfg:
            cfg["local_search_unary_ops"] = _normalize_ops_like(raw_cfg["local_search_unary_ops"], field="local_search_unary_ops")
        if "local_search_nested_unary_patterns" in raw_cfg:
            cfg["local_search_nested_unary_patterns"] = _normalize_ops_like(
                raw_cfg["local_search_nested_unary_patterns"], field="local_search_nested_unary_patterns"
            )
        if "local_search_overfit_guard_enabled" in raw_cfg:
            cfg["local_search_overfit_guard_enabled"] = bool(raw_cfg["local_search_overfit_guard_enabled"])
        if "local_search_overfit_guard_val_ratio" in raw_cfg:
            cfg["local_search_overfit_guard_val_ratio"] = float(
                np.clip(float(raw_cfg["local_search_overfit_guard_val_ratio"]), 0.0, 0.9)
            )
        if "local_search_overfit_guard_min_val_samples" in raw_cfg:
            cfg["local_search_overfit_guard_min_val_samples"] = int(
                max(1, int(raw_cfg["local_search_overfit_guard_min_val_samples"]))
            )
        if "local_search_overfit_guard_min_val_rmse_gain" in raw_cfg:
            cfg["local_search_overfit_guard_min_val_rmse_gain"] = float(
                max(0.0, float(raw_cfg["local_search_overfit_guard_min_val_rmse_gain"]))
            )
        if "local_search_overfit_guard_max_gap_increase" in raw_cfg:
            cfg["local_search_overfit_guard_max_gap_increase"] = float(
                max(0.0, float(raw_cfg["local_search_overfit_guard_max_gap_increase"]))
            )
        if "local_search_overfit_guard_patience" in raw_cfg:
            cfg["local_search_overfit_guard_patience"] = int(
                max(0, int(raw_cfg["local_search_overfit_guard_patience"]))
            )
        if "local_search_interaction_budget_mode" in raw_cfg:
            mode = str(raw_cfg["local_search_interaction_budget_mode"]).strip().lower()
            if mode not in {"fixed", "interaction_first"}:
                raise ValueError(
                    f"strict4 branch '{raw_key}' has invalid local_search_interaction_budget_mode='{mode}', "
                    "expected fixed|interaction_first."
                )
            cfg["local_search_interaction_budget_mode"] = mode
        if "local_search_interaction_diag_threshold" in raw_cfg:
            cfg["local_search_interaction_diag_threshold"] = float(
                max(0.0, float(raw_cfg["local_search_interaction_diag_threshold"]))
            )
        if "local_search_interaction_diag_topk_features" in raw_cfg:
            cfg["local_search_interaction_diag_topk_features"] = int(
                max(1, int(raw_cfg["local_search_interaction_diag_topk_features"]))
            )
        if "local_search_interaction_pair_budget_boost" in raw_cfg:
            cfg["local_search_interaction_pair_budget_boost"] = float(
                max(1.0, float(raw_cfg["local_search_interaction_pair_budget_boost"]))
            )
        if "local_search_interaction_grad_projection_budget_boost" in raw_cfg:
            cfg["local_search_interaction_grad_projection_budget_boost"] = float(
                max(1.0, float(raw_cfg["local_search_interaction_grad_projection_budget_boost"]))
            )
        if "local_search_inner_opt_enabled" in raw_cfg:
            cfg["local_search_inner_opt_enabled"] = bool(raw_cfg["local_search_inner_opt_enabled"])
        if "local_search_inner_opt_method" in raw_cfg:
            mode = str(raw_cfg["local_search_inner_opt_method"]).strip().lower()
            if mode not in {"adam_lbfgs", "adam", "lbfgs"}:
                raise ValueError(
                    f"strict4 branch '{raw_key}' has invalid local_search_inner_opt_method='{mode}', "
                    "expected adam_lbfgs|adam|lbfgs."
                )
            cfg["local_search_inner_opt_method"] = mode
        if "local_search_inner_opt_device" in raw_cfg:
            cfg["local_search_inner_opt_device"] = str(raw_cfg["local_search_inner_opt_device"]).strip().lower()
        if "local_search_inner_opt_adam_steps" in raw_cfg:
            cfg["local_search_inner_opt_adam_steps"] = int(max(0, int(raw_cfg["local_search_inner_opt_adam_steps"])))
        if "local_search_inner_opt_adam_lr" in raw_cfg:
            cfg["local_search_inner_opt_adam_lr"] = float(max(1e-8, float(raw_cfg["local_search_inner_opt_adam_lr"])))
        if "local_search_inner_opt_lbfgs_steps" in raw_cfg:
            cfg["local_search_inner_opt_lbfgs_steps"] = int(max(0, int(raw_cfg["local_search_inner_opt_lbfgs_steps"])))
        if "local_search_inner_opt_lbfgs_lr" in raw_cfg:
            cfg["local_search_inner_opt_lbfgs_lr"] = float(max(1e-8, float(raw_cfg["local_search_inner_opt_lbfgs_lr"])))
        if "local_search_inner_opt_l2" in raw_cfg:
            cfg["local_search_inner_opt_l2"] = float(max(0.0, float(raw_cfg["local_search_inner_opt_l2"])))
        if "local_search_inner_opt_accept_rmse_tol" in raw_cfg:
            cfg["local_search_inner_opt_accept_rmse_tol"] = float(
                max(0.0, float(raw_cfg["local_search_inner_opt_accept_rmse_tol"]))
            )
        if "blend_kappa" in raw_cfg:
            cfg["blend_kappa"] = float(max(1e-6, float(raw_cfg["blend_kappa"])))
        out[alias] = cfg
    return out


def _resolve_local_branch_settings(
    *,
    spec: RollingSpec,
    regime_key: tuple[int, ...] | None,
    param_count: int,
    train_sample_count: int | None = None,
) -> dict[str, Any]:
    base: dict[str, Any] = {
        "local_search_force_linear_base": str(spec.local_search_force_linear_base).strip().lower(),
        "local_search_topk_features": int(spec.local_search_topk_features),
        "local_search_max_added_terms": int(spec.local_search_max_added_terms),
        "local_search_max_pair_terms": int(spec.local_search_max_pair_terms),
        "local_search_max_candidates_per_iter": int(spec.local_search_max_candidates_per_iter),
        "local_search_candidate_keep_top": int(spec.local_search_candidate_keep_top),
        "local_search_ridge_l2": float(spec.local_search_ridge_l2),
        "local_search_unary_ops": tuple(spec.local_search_unary_ops),
        "local_search_nested_unary_patterns": tuple(spec.local_search_nested_unary_patterns),
        "local_search_overfit_guard_enabled": bool(spec.local_search_overfit_guard_enabled),
        "local_search_overfit_guard_val_ratio": float(spec.local_search_overfit_guard_val_ratio),
        "local_search_overfit_guard_min_val_samples": int(spec.local_search_overfit_guard_min_val_samples),
        "local_search_overfit_guard_min_val_rmse_gain": float(spec.local_search_overfit_guard_min_val_rmse_gain),
        "local_search_overfit_guard_max_gap_increase": float(spec.local_search_overfit_guard_max_gap_increase),
        "local_search_overfit_guard_patience": int(spec.local_search_overfit_guard_patience),
        "local_search_interaction_budget_mode": str(spec.local_search_interaction_budget_mode),
        "local_search_interaction_diag_threshold": float(spec.local_search_interaction_diag_threshold),
        "local_search_interaction_diag_topk_features": int(spec.local_search_interaction_diag_topk_features),
        "local_search_interaction_pair_budget_boost": float(spec.local_search_interaction_pair_budget_boost),
        "local_search_interaction_grad_projection_budget_boost": float(
            spec.local_search_interaction_grad_projection_budget_boost
        ),
        "local_search_inner_opt_enabled": bool(spec.local_search_inner_opt_enabled),
        "local_search_inner_opt_method": str(spec.local_search_inner_opt_method),
        "local_search_inner_opt_device": str(spec.local_search_inner_opt_device),
        "local_search_inner_opt_adam_steps": int(spec.local_search_inner_opt_adam_steps),
        "local_search_inner_opt_adam_lr": float(spec.local_search_inner_opt_adam_lr),
        "local_search_inner_opt_lbfgs_steps": int(spec.local_search_inner_opt_lbfgs_steps),
        "local_search_inner_opt_lbfgs_lr": float(spec.local_search_inner_opt_lbfgs_lr),
        "local_search_inner_opt_l2": float(spec.local_search_inner_opt_l2),
        "local_search_inner_opt_accept_rmse_tol": float(spec.local_search_inner_opt_accept_rmse_tol),
        "blend_kappa": float(spec.blend_kappa),
        "branch_alias": "default",
    }
    regime_mode = str(spec.regime_mode).strip().lower()
    if regime_mode == "strict4" and regime_key is not None:
        alias = _strict4_alias_for_regime_key(regime_key)
        base["branch_alias"] = alias
        override = dict(spec.strict4_branch_hparams.get(alias, {}))
        for k, v in override.items():
            base[k] = v

    n_fit = int(train_sample_count) if train_sample_count is not None else -1
    small_guard_threshold = int(max(0, int(spec.small_sample_guard_threshold)))
    small_guard_applied = bool(small_guard_threshold > 0 and n_fit > 0 and n_fit <= small_guard_threshold)
    if small_guard_applied:
        base["local_search_force_linear_base"] = "on"
        base["local_search_max_added_terms"] = min(int(base["local_search_max_added_terms"]), 4)
        base["local_search_max_pair_terms"] = min(int(base["local_search_max_pair_terms"]), 2)
        base["local_search_max_candidates_per_iter"] = min(int(base["local_search_max_candidates_per_iter"]), 120)
        base["local_search_candidate_keep_top"] = min(int(base["local_search_candidate_keep_top"]), 4)
        base["local_search_ridge_l2"] = max(float(base["local_search_ridge_l2"]), 1e-2)
        base["local_search_unary_ops"] = ("sin", "cos", "tanh")
        base["local_search_nested_unary_patterns"] = ("none",)
        base["local_search_overfit_guard_enabled"] = True
        base["local_search_overfit_guard_val_ratio"] = 0.3
        base["local_search_overfit_guard_min_val_samples"] = 64
        base["local_search_overfit_guard_min_val_rmse_gain"] = max(
            float(base["local_search_overfit_guard_min_val_rmse_gain"]), 0.005
        )
        base["local_search_overfit_guard_max_gap_increase"] = min(
            float(base["local_search_overfit_guard_max_gap_increase"]), 0.005
        )
        base["local_search_overfit_guard_patience"] = min(int(base["local_search_overfit_guard_patience"]), 1)
        base["local_search_interaction_budget_mode"] = "fixed"

    unary_ops = tuple(base["local_search_unary_ops"]) if len(tuple(base["local_search_unary_ops"])) > 0 else (
        "square",
        "sin",
        "cos",
        "tanh",
    )
    nested_patterns = tuple(base["local_search_nested_unary_patterns"])
    stage_params = {
        "force_linear_base": str(base["local_search_force_linear_base"]),
        "keep_search_trace": False,
        "auto_val_ratio": 0.2,
        "auto_min_val_samples": 32,
        "auto_random_seed": 42,
        "search_ridge_l2": float(max(0.0, float(base["local_search_ridge_l2"]))),
        "search_max_added_terms": int(max(0, int(base["local_search_max_added_terms"]))),
        "search_topk_features": min(int(max(1, int(base["local_search_topk_features"]))), int(max(1, param_count))),
        "search_max_pair_terms": int(max(0, int(base["local_search_max_pair_terms"]))),
        "search_max_candidates_per_iter": int(max(1, int(base["local_search_max_candidates_per_iter"]))),
        "search_candidate_keep_top": int(max(1, int(base["local_search_candidate_keep_top"]))),
        "search_include_hinge": True,
        "search_hinge_quantiles": [0.25, 0.5, 0.75],
        "search_unary_ops": list(unary_ops),
        "search_nested_unary_patterns": list(nested_patterns),
        "search_interaction_budget_mode": str(base["local_search_interaction_budget_mode"]),
        "search_interaction_diag_threshold": float(max(0.0, float(base["local_search_interaction_diag_threshold"]))),
        "search_interaction_diag_topk_features": min(
            int(max(1, int(base["local_search_interaction_diag_topk_features"]))),
            int(max(1, param_count)),
        ),
        "search_interaction_pair_budget_boost": float(max(1.0, float(base["local_search_interaction_pair_budget_boost"]))),
        "search_interaction_grad_projection_budget_boost": float(
            max(1.0, float(base["local_search_interaction_grad_projection_budget_boost"]))
        ),
        "search_overfit_guard_enabled": bool(base["local_search_overfit_guard_enabled"]),
        "search_overfit_guard_val_ratio": float(np.clip(float(base["local_search_overfit_guard_val_ratio"]), 0.0, 0.9)),
        "search_overfit_guard_min_val_samples": int(max(1, int(base["local_search_overfit_guard_min_val_samples"]))),
        "search_overfit_guard_min_val_rmse_gain": float(max(0.0, float(base["local_search_overfit_guard_min_val_rmse_gain"]))),
        "search_overfit_guard_max_gap_increase": float(max(0.0, float(base["local_search_overfit_guard_max_gap_increase"]))),
        "search_overfit_guard_patience": int(max(0, int(base["local_search_overfit_guard_patience"]))),
        "search_enable_prune": True,
        "search_prune_rmse_tolerance": 1e-6,
        "search_prune_max_removed_per_iter": 1,
        "search_path_memory_enabled": False,
        "search_min_actual_rmse_gain": 0.0,
        "search_inner_opt_enabled": bool(base["local_search_inner_opt_enabled"]),
        "search_inner_opt_method": str(base["local_search_inner_opt_method"]),
        "search_inner_opt_device": str(base["local_search_inner_opt_device"]),
        "search_inner_opt_adam_steps": int(max(0, int(base["local_search_inner_opt_adam_steps"]))),
        "search_inner_opt_adam_lr": float(max(1e-8, float(base["local_search_inner_opt_adam_lr"]))),
        "search_inner_opt_lbfgs_steps": int(max(0, int(base["local_search_inner_opt_lbfgs_steps"]))),
        "search_inner_opt_lbfgs_lr": float(max(1e-8, float(base["local_search_inner_opt_lbfgs_lr"]))),
        "search_inner_opt_l2": float(max(0.0, float(base["local_search_inner_opt_l2"]))),
        "search_inner_opt_accept_rmse_tol": float(max(0.0, float(base["local_search_inner_opt_accept_rmse_tol"]))),
    }
    out = {
        "branch_alias": str(base["branch_alias"]),
        "blend_kappa": float(max(1e-6, float(base["blend_kappa"]))),
        "settings_raw": {
            "local_search_force_linear_base": str(base["local_search_force_linear_base"]),
            "local_search_topk_features": int(max(1, int(base["local_search_topk_features"]))),
            "local_search_max_added_terms": int(max(0, int(base["local_search_max_added_terms"]))),
            "local_search_max_pair_terms": int(max(0, int(base["local_search_max_pair_terms"]))),
            "local_search_max_candidates_per_iter": int(max(1, int(base["local_search_max_candidates_per_iter"]))),
            "local_search_candidate_keep_top": int(max(1, int(base["local_search_candidate_keep_top"]))),
            "local_search_ridge_l2": float(max(0.0, float(base["local_search_ridge_l2"]))),
            "local_search_unary_ops": tuple(unary_ops),
            "local_search_nested_unary_patterns": tuple(nested_patterns),
            "local_search_overfit_guard_enabled": bool(base["local_search_overfit_guard_enabled"]),
            "local_search_overfit_guard_val_ratio": float(
                np.clip(float(base["local_search_overfit_guard_val_ratio"]), 0.0, 0.9)
            ),
            "local_search_overfit_guard_min_val_samples": int(
                max(1, int(base["local_search_overfit_guard_min_val_samples"]))
            ),
            "local_search_overfit_guard_min_val_rmse_gain": float(
                max(0.0, float(base["local_search_overfit_guard_min_val_rmse_gain"]))
            ),
            "local_search_overfit_guard_max_gap_increase": float(
                max(0.0, float(base["local_search_overfit_guard_max_gap_increase"]))
            ),
            "local_search_overfit_guard_patience": int(max(0, int(base["local_search_overfit_guard_patience"]))),
            "local_search_interaction_budget_mode": str(base["local_search_interaction_budget_mode"]),
            "local_search_interaction_diag_threshold": float(
                max(0.0, float(base["local_search_interaction_diag_threshold"]))
            ),
            "local_search_interaction_diag_topk_features": int(
                max(1, int(base["local_search_interaction_diag_topk_features"]))
            ),
            "local_search_interaction_pair_budget_boost": float(
                max(1.0, float(base["local_search_interaction_pair_budget_boost"]))
            ),
            "local_search_interaction_grad_projection_budget_boost": float(
                max(1.0, float(base["local_search_interaction_grad_projection_budget_boost"]))
            ),
            "local_search_inner_opt_enabled": bool(base["local_search_inner_opt_enabled"]),
            "local_search_inner_opt_method": str(base["local_search_inner_opt_method"]),
            "local_search_inner_opt_device": str(base["local_search_inner_opt_device"]),
            "local_search_inner_opt_adam_steps": int(max(0, int(base["local_search_inner_opt_adam_steps"]))),
            "local_search_inner_opt_adam_lr": float(max(1e-8, float(base["local_search_inner_opt_adam_lr"]))),
            "local_search_inner_opt_lbfgs_steps": int(max(0, int(base["local_search_inner_opt_lbfgs_steps"]))),
            "local_search_inner_opt_lbfgs_lr": float(max(1e-8, float(base["local_search_inner_opt_lbfgs_lr"]))),
            "local_search_inner_opt_l2": float(max(0.0, float(base["local_search_inner_opt_l2"]))),
            "local_search_inner_opt_accept_rmse_tol": float(max(0.0, float(base["local_search_inner_opt_accept_rmse_tol"]))),
            "blend_kappa": float(max(1e-6, float(base["blend_kappa"]))),
            "small_sample_guard_threshold": int(small_guard_threshold),
            "small_sample_guard_applied": bool(small_guard_applied),
            "train_sample_count": int(n_fit),
        },
        "stage_params": stage_params,
    }
    return out


def _load_table(
    *,
    csv_path: str,
    target_col: str,
    date_col: str,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...], np.ndarray]:
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"CSV is empty: {csv_path}")
    if target_col not in df.columns:
        raise ValueError(f"target_col '{target_col}' not found")
    if date_col not in df.columns:
        raise ValueError(f"date_col '{date_col}' not found")

    fold_cols = [c for c in df.columns if str(c).startswith("test_fold_")]
    drop_cols = set(fold_cols)
    drop_cols.add(target_col)
    drop_cols.add(date_col)
    feature_cols = [str(c) for c in df.columns if c not in drop_cols]
    if not feature_cols:
        raise ValueError("No feature columns selected after dropping date/target/fold columns")

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    if df[date_col].isna().any():
        bad = int(df[date_col].isna().sum())
        raise ValueError(f"Found {bad} invalid dates in '{date_col}'")
    df = df.sort_values(date_col).reset_index(drop=True)

    X_df = df[feature_cols].copy()
    y_sr = df[target_col].copy()
    for c in feature_cols:
        X_df[c] = pd.to_numeric(X_df[c], errors="coerce")
    y_sr = pd.to_numeric(y_sr, errors="coerce")

    if X_df.isna().any().any() or y_sr.isna().any():
        bad_cols = [c for c in feature_cols if X_df[c].isna().any()]
        if y_sr.isna().any():
            bad_cols.append(target_col)
        raise ValueError(f"Found NaN after numeric conversion. Columns: {bad_cols}")

    X_all = X_df.to_numpy(dtype=float)
    y_all = y_sr.to_numpy(dtype=float).reshape(-1, 1)
    dates = df[date_col].to_numpy()
    return X_all, y_all, tuple(feature_cols), dates


def _build_rolling_splits(
    *,
    n_samples: int,
    min_train_size: int,
    test_size: int,
    step_size: int,
    split_mode: str,
    train_window_size: int | None,
) -> list[dict[str, int]]:
    if n_samples <= 0:
        return []
    if min_train_size <= 0 or test_size <= 0 or step_size <= 0:
        raise ValueError("min_train_size/test_size/step_size must be > 0")
    if min_train_size + test_size > n_samples:
        raise ValueError("min_train_size + test_size exceeds total samples")

    mode = str(split_mode).strip().lower()
    if mode not in {"expanding", "sliding"}:
        raise ValueError(f"Unsupported split_mode: {split_mode}")
    if mode == "sliding" and (train_window_size is None or int(train_window_size) <= 0):
        raise ValueError("train_window_size must be > 0 for sliding mode")

    out: list[dict[str, int]] = []
    train_end = int(min_train_size)
    split_id = 0
    while train_end + int(test_size) <= int(n_samples):
        test_start = int(train_end)
        test_end = int(test_start + int(test_size))
        if mode == "expanding":
            train_start = 0
        else:
            train_start = max(0, int(train_end) - int(train_window_size))
        if train_start >= train_end:
            break
        out.append(
            {
                "split_id": int(split_id),
                "train_start": int(train_start),
                "train_end": int(train_end),
                "test_start": int(test_start),
                "test_end": int(test_end),
            }
        )
        split_id += 1
        train_end += int(step_size)
    return out


def _evaluate_one_split(
    *,
    split_tag: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: tuple[str, ...],
    spec: RollingSpec,
) -> dict[str, Any]:
    gate_idx = _col_index(feature_names, spec.gate_features)
    param_idx = _col_index(feature_names, spec.param_features)
    if len(gate_idx) == 0:
        raise ValueError("No holiday gate features found in dataset.")
    if len(param_idx) == 0:
        raise ValueError("No parameter features found in dataset.")

    gate_train = _slice_cols(X_train, gate_idx)
    gate_test = _slice_cols(X_test, gate_idx)
    X_train_param = _slice_cols(X_train, param_idx)
    X_test_param = _slice_cols(X_test, param_idx)
    param_names = tuple(feature_names[i] for i in param_idx)
    gate_names = tuple(feature_names[i] for i in gate_idx)

    key_train = _gate_key(gate_train)
    key_test = _gate_key(gate_test)
    regime_mode = str(spec.regime_mode).strip().lower()
    if regime_mode == "strict4":
        key_train = _map_keys_strict4(key_train)
        key_test = _map_keys_strict4(key_test)
    elif regime_mode == "strict3":
        key_train = _map_keys_strict3(key_train)
        key_test = _map_keys_strict3(key_test)
    count_train = Counter(key_train)
    count_test = Counter(key_test)

    t0 = time.perf_counter()
    xgb = _fit_artifact(
        trainer_key="xgboost",
        trainer_params={
            "artifact_id": f"rolling_xgb_{split_tag}",
            "n_estimators": 360,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "tree_method": "hist",
            "random_seed": 42,
        },
        X=X_train_param,
        y=y_train,
        feature_names=param_names,
    )
    pred_xgb = np.asarray(xgb.predict(X_test_param), dtype=float).reshape(-1, 1)
    xgb_metrics = _metrics(y_test, pred_xgb)
    xgb_sec = float(time.perf_counter() - t0)

    t0 = time.perf_counter()
    resolved_global = _resolve_local_branch_settings(
        spec=spec,
        regime_key=None,
        param_count=len(param_names),
        train_sample_count=int(X_train_param.shape[0]),
    )
    global_stage = _fit_artifact(
        trainer_key="symbolic_stagewise",
        trainer_params={
            "artifact_id": f"rolling_global_stage_{split_tag}",
            **resolved_global["stage_params"],
        },
        X=X_train_param,
        y=y_train,
        feature_names=param_names,
    )
    pred_global = np.asarray(global_stage.predict(X_test_param), dtype=float).reshape(-1, 1)
    global_stage_metrics = _metrics(y_test, pred_global)
    global_stage_sec = float(time.perf_counter() - t0)
    pred_global_for_blend = np.asarray(pred_global, dtype=float).reshape(-1, 1)
    global_backbone_choice: dict[str, Any] = {
        "mode": str(spec.blend_global_backbone_mode),
        "selected": "symbolic_stagewise",
        "active": False,
        "val_rmse_symbolic": None,
        "val_rmse_xgboost": None,
        "margin": float(spec.blend_global_backbone_margin),
    }

    backbone_mode = str(spec.blend_global_backbone_mode).strip().lower()
    if backbone_mode == "best_of_symbolic_xgboost":
        idx_fit, idx_val = _time_val_split_indices(
            n_samples=int(X_train_param.shape[0]),
            val_ratio=float(np.clip(spec.blend_global_backbone_val_ratio, 0.05, 0.45)),
            min_val_samples=int(max(16, spec.blend_global_backbone_min_val_samples)),
        )
        if int(idx_fit.size) > 0 and int(idx_val.size) > 0:
            xgb_val = _fit_artifact(
                trainer_key="xgboost",
                trainer_params={
                    "artifact_id": f"rolling_xgb_backbone_val_{split_tag}",
                    "n_estimators": 360,
                    "max_depth": 6,
                    "learning_rate": 0.05,
                    "subsample": 0.9,
                    "colsample_bytree": 0.9,
                    "tree_method": "hist",
                    "random_seed": 42,
                },
                X=X_train_param[idx_fit],
                y=y_train[idx_fit],
                feature_names=param_names,
            )
            resolved_global_val = _resolve_local_branch_settings(
                spec=spec,
                regime_key=None,
                param_count=len(param_names),
                train_sample_count=int(idx_fit.size),
            )
            sym_val = _fit_artifact(
                trainer_key="symbolic_stagewise",
                trainer_params={
                    "artifact_id": f"rolling_sym_backbone_val_{split_tag}",
                    **resolved_global_val["stage_params"],
                },
                X=X_train_param[idx_fit],
                y=y_train[idx_fit],
                feature_names=param_names,
            )
            y_val = np.asarray(y_train[idx_val], dtype=float).reshape(-1, 1)
            pred_xgb_val = np.asarray(xgb_val.predict(X_train_param[idx_val]), dtype=float).reshape(-1, 1)
            pred_sym_val = np.asarray(sym_val.predict(X_train_param[idx_val]), dtype=float).reshape(-1, 1)
            rmse_xgb_val = float(_metrics(y_val, pred_xgb_val)["rmse"])
            rmse_sym_val = float(_metrics(y_val, pred_sym_val)["rmse"])
            choose_xgb = bool(rmse_xgb_val + float(spec.blend_global_backbone_margin) < rmse_sym_val)
            if choose_xgb:
                pred_global_for_blend = np.asarray(pred_xgb, dtype=float).reshape(-1, 1)
            global_backbone_choice = {
                "mode": str(spec.blend_global_backbone_mode),
                "selected": ("xgboost" if choose_xgb else "symbolic_stagewise"),
                "active": True,
                "val_n_fit": int(idx_fit.size),
                "val_n_val": int(idx_val.size),
                "val_rmse_symbolic": float(rmse_sym_val),
                "val_rmse_xgboost": float(rmse_xgb_val),
                "margin": float(spec.blend_global_backbone_margin),
            }

    t0 = time.perf_counter()
    local_models: dict[tuple[int, ...], Any] = {}
    local_effective_samples: dict[tuple[int, ...], int] = {}
    local_blend_kappa_by_regime: dict[tuple[int, ...], float] = {}
    strict4_gpu_assignments: dict[str, str] = {}
    strict4_parallel_mode_effective = "serial"
    strict4_parallel_workers_effective = 1
    local_training_jobs: list[dict[str, Any]] = []
    regime_index = _build_regime_index(key_train)
    if regime_mode == "strict4":
        regime_keys_all = tuple(STRICT4_REGIME_ORDER)
    elif regime_mode == "strict3":
        regime_keys_all = tuple(STRICT3_REGIME_ORDER)
    else:
        regime_keys_all = tuple(sorted(set(key_train) | set(key_test)))
    regime_training_detail: dict[str, Any] = {}
    rare_merge_group = ((1, 1, 0, 0), (1, 0, 1, 0))
    merged_group_used = False
    strict4_dynamic_merge_used = False
    strict4_dynamic_merge_detail: dict[str, Any] = {}

    if spec.merge_rare_holiday_regimes and regime_mode not in {"strict4", "strict3"}:
        group_keys = tuple(k for k in rare_merge_group if k in regime_keys_all)
        if len(group_keys) >= 2:
            parts = [np.asarray(regime_index.get(k, np.asarray([], dtype=int)), dtype=int) for k in group_keys]
            parts = [p for p in parts if int(p.size) > 0]
            if parts:
                idx_merge = np.unique(np.concatenate(parts, axis=0).astype(int, copy=False))
                if int(idx_merge.size) > 0:
                    resolved = _resolve_local_branch_settings(
                        spec=spec,
                        regime_key=None,
                        param_count=len(param_names),
                        train_sample_count=int(idx_merge.size),
                    )
                    shared_art = _fit_artifact(
                        trainer_key="symbolic_stagewise",
                        trainer_params={
                            "artifact_id": f"rolling_local_shared_1100_1010_{split_tag}",
                            **resolved["stage_params"],
                        },
                        X=X_train_param[idx_merge],
                        y=y_train[idx_merge],
                        feature_names=param_names,
                    )
                    used_from = {
                        str(k): int(np.asarray(regime_index.get(k, np.asarray([], dtype=int)), dtype=int).size)
                        for k in group_keys
                    }
                    merged_group_used = True
                    for k in group_keys:
                        local_models[k] = shared_art
                        local_effective_samples[k] = int(idx_merge.size)
                        local_blend_kappa_by_regime[k] = float(resolved["blend_kappa"])
                        regime_training_detail[str(k)] = {
                            "target": list(k),
                            "exact_count": int(
                                np.asarray(regime_index.get(k, np.asarray([], dtype=int)), dtype=int).size
                            ),
                            "used_count": int(idx_merge.size),
                            "used_from": used_from,
                            "shared_model_with": [list(x) for x in group_keys],
                            "local_hparams_used": _jsonable(resolved["settings_raw"]),
                        }

    if regime_mode == "strict4" and bool(spec.strict4_dynamic_merge_enabled):
        k_near = (1, 1, 0, 0)
        k_mid = (1, 0, 1, 0)
        idx_near = np.asarray(regime_index.get(k_near, np.asarray([], dtype=int)), dtype=int)
        idx_mid = np.asarray(regime_index.get(k_mid, np.asarray([], dtype=int)), dtype=int)
        n_near = int(idx_near.size)
        n_mid = int(idx_mid.size)
        min_n = int(max(1, int(spec.strict4_dynamic_merge_min_samples)))
        should_merge = (
            (n_near > 0 or n_mid > 0)
            and (n_near < min_n or n_mid < min_n)
        )
        if should_merge:
            idx_merge = np.unique(
                np.concatenate(
                    [
                        np.asarray(idx_near, dtype=int),
                        np.asarray(idx_mid, dtype=int),
                    ],
                    axis=0,
                ).astype(int, copy=False)
            )
            if int(idx_merge.size) > 0:
                resolved = _resolve_local_branch_settings(
                    spec=spec,
                    regime_key=None,
                    param_count=len(param_names),
                    train_sample_count=int(idx_merge.size),
                )
                shared_art = _fit_artifact(
                    trainer_key="symbolic_stagewise",
                    trainer_params={
                        "artifact_id": f"rolling_local_shared_strict4_holiday_{split_tag}",
                        **resolved["stage_params"],
                    },
                    X=X_train_param[idx_merge],
                    y=y_train[idx_merge],
                    feature_names=param_names,
                )
                used_from = {
                    str(k_near): int(n_near),
                    str(k_mid): int(n_mid),
                }
                strict4_dynamic_merge_used = True
                strict4_dynamic_merge_detail = {
                    "enabled": True,
                    "min_samples": int(min_n),
                    "holiday_near_count": int(n_near),
                    "holiday_mid_count": int(n_mid),
                    "merged_count": int(idx_merge.size),
                    "used_from": used_from,
                }
                for k in (k_near, k_mid):
                    local_models[k] = shared_art
                    local_effective_samples[k] = int(idx_merge.size)
                    local_blend_kappa_by_regime[k] = float(resolved["blend_kappa"])
                    regime_training_detail[str(k)] = {
                        "target": list(k),
                        "exact_count": int(np.asarray(regime_index.get(k, np.asarray([], dtype=int)), dtype=int).size),
                        "used_count": int(idx_merge.size),
                        "used_from": used_from,
                        "strict_exact": True,
                        "dynamic_merge_holiday": True,
                        "shared_model_with": [list(k_near), list(k_mid)],
                        "local_hparams_used": _jsonable(resolved["settings_raw"]),
                    }

    for branch_order, k in enumerate(regime_keys_all):
        if k in local_models:
            continue
        if regime_mode in {"strict4", "strict3"}:
            idx_exact = np.asarray(regime_index.get(k, np.asarray([], dtype=int)), dtype=int)
            resolved = _resolve_local_branch_settings(
                spec=spec,
                regime_key=k,
                param_count=len(param_names),
                train_sample_count=int(idx_exact.size),
            )
            if int(idx_exact.size) <= 0:
                regime_training_detail[str(k)] = {
                    "target": list(k),
                    "exact_count": 0,
                    "used_count": 0,
                    "used_from": {},
                    "strict_exact": True,
                    "skipped": True,
                    "local_hparams_used": _jsonable(resolved["settings_raw"]),
                }
                continue
            idx_use = idx_exact
            use_detail = {
                "target": list(k),
                "exact_count": int(idx_exact.size),
                "used_count": int(idx_exact.size),
                "used_from": {str(k): int(idx_exact.size)},
                "strict_exact": True,
                "skipped": False,
                "local_hparams_used": _jsonable(resolved["settings_raw"]),
            }
        else:
            idx_use, use_detail = _select_training_indices_for_regime(
                target_key=k,
                regime_index=regime_index,
                min_leaf=int(spec.min_leaf),
            )
            resolved = _resolve_local_branch_settings(
                spec=spec,
                regime_key=None,
                param_count=len(param_names),
                train_sample_count=int(idx_use.size),
            )
            use_detail = dict(use_detail)
            use_detail["local_hparams_used"] = _jsonable(resolved["settings_raw"])
        trainer_params = {
            "artifact_id": f"rolling_local_{'_'.join(map(str, k))}_{split_tag}",
            **dict(resolved["stage_params"]),
        }
        if bool(trainer_params.get("search_inner_opt_enabled", False)):
            assigned_device = _select_gpu_device_for_branch(
                strategy=str(spec.strict4_gpu_strategy),
                configured_devices=tuple(spec.strict4_gpu_devices),
                branch_order=int(branch_order),
            )
            if assigned_device is not None:
                trainer_params["search_inner_opt_device"] = str(assigned_device)
                use_detail["assigned_inner_opt_device"] = str(assigned_device)
                strict4_gpu_assignments[str(k)] = str(assigned_device)

        if regime_mode in {"strict4", "strict3"}:
            local_training_jobs.append(
                {
                    "regime_key": tuple(k),
                    "idx_use": np.asarray(idx_use, dtype=int),
                    "trainer_params": dict(trainer_params),
                    "blend_kappa": float(resolved["blend_kappa"]),
                    "used_count": int(use_detail.get("used_count", int(idx_use.size))),
                    "use_detail": dict(use_detail),
                }
            )
            continue

        art = _fit_local_regime_task(
            trainer_params=dict(trainer_params),
            X_slice=X_train_param[idx_use],
            y_slice=y_train[idx_use],
            feature_names=param_names,
        )
        local_models[k] = art
        local_effective_samples[k] = int(use_detail.get("used_count", int(idx_use.size)))
        local_blend_kappa_by_regime[k] = float(resolved["blend_kappa"])
        regime_training_detail[str(k)] = use_detail

    if regime_mode in {"strict4", "strict3"} and local_training_jobs:
        requested_mode = _normalize_parallel_mode(spec.strict4_parallel_mode)
        requested_workers = int(max(1, int(spec.strict4_max_workers)))
        if requested_mode == "serial":
            strict4_parallel_mode_effective = "serial"
            strict4_parallel_workers_effective = 1
            for job in local_training_jobs:
                idx_use = np.asarray(job["idx_use"], dtype=int)
                art = _fit_local_regime_task(
                    trainer_params=dict(job["trainer_params"]),
                    X_slice=X_train_param[idx_use],
                    y_slice=y_train[idx_use],
                    feature_names=param_names,
                )
                rk = tuple(job["regime_key"])
                local_models[rk] = art
                local_effective_samples[rk] = int(job["used_count"])
                local_blend_kappa_by_regime[rk] = float(job["blend_kappa"])
                regime_training_detail[str(rk)] = dict(job["use_detail"])
        else:
            strict4_parallel_workers_effective = _resolve_parallel_workers(
                n_tasks=len(local_training_jobs),
                requested=requested_workers,
            )
            if strict4_parallel_workers_effective <= 1:
                strict4_parallel_mode_effective = "serial"
                strict4_parallel_workers_effective = 1
                for job in local_training_jobs:
                    idx_use = np.asarray(job["idx_use"], dtype=int)
                    art = _fit_local_regime_task(
                        trainer_params=dict(job["trainer_params"]),
                        X_slice=X_train_param[idx_use],
                        y_slice=y_train[idx_use],
                        feature_names=param_names,
                    )
                    rk = tuple(job["regime_key"])
                    local_models[rk] = art
                    local_effective_samples[rk] = int(job["used_count"])
                    local_blend_kappa_by_regime[rk] = float(job["blend_kappa"])
                    regime_training_detail[str(rk)] = dict(job["use_detail"])
            else:
                strict4_parallel_mode_effective = requested_mode
                executor_cls: Any = (
                    concurrent.futures.ThreadPoolExecutor
                    if requested_mode == "thread"
                    else concurrent.futures.ProcessPoolExecutor
                )
                with executor_cls(max_workers=int(strict4_parallel_workers_effective)) as executor:
                    fut2job: dict[concurrent.futures.Future[Any], dict[str, Any]] = {}
                    for job in local_training_jobs:
                        idx_use = np.asarray(job["idx_use"], dtype=int)
                        fut = executor.submit(
                            _fit_local_regime_task,
                            trainer_params=dict(job["trainer_params"]),
                            X_slice=np.asarray(X_train_param[idx_use], dtype=float),
                            y_slice=np.asarray(y_train[idx_use], dtype=float),
                            feature_names=param_names,
                        )
                        fut2job[fut] = job
                    for fut in concurrent.futures.as_completed(fut2job):
                        job = fut2job[fut]
                        rk = tuple(job["regime_key"])
                        try:
                            art = fut.result()
                        except Exception as exc:
                            raise RuntimeError(
                                f"strict4 local branch training failed for regime={rk}: {type(exc).__name__}: {exc}"
                            ) from exc
                        local_models[rk] = art
                        local_effective_samples[rk] = int(job["used_count"])
                        local_blend_kappa_by_regime[rk] = float(job["blend_kappa"])
                        regime_training_detail[str(rk)] = dict(job["use_detail"])

    pred_piece = np.zeros_like(y_test, dtype=float)
    pred_blend = np.zeros_like(y_test, dtype=float)
    blend_weight = np.zeros((y_test.shape[0],), dtype=float)
    fallback_count = 0
    for i, k in enumerate(key_test):
        x_row = X_test_param[i : i + 1, :]
        pg = float(pred_global_for_blend[i, 0])
        art = local_models.get(k)
        if art is None:
            fallback_count += 1
            pl = pg
            alpha = 0.0
        else:
            pl = float(np.asarray(art.predict(x_row), dtype=float).reshape(-1)[0])
            n_eff = float(local_effective_samples.get(k, 0))
            if spec.blend_with_global:
                kappa_eff = float(local_blend_kappa_by_regime.get(k, float(spec.blend_kappa)))
                alpha = float(n_eff / (n_eff + kappa_eff))
            else:
                alpha = 1.0
        pred_piece[i, 0] = pl
        pred_blend[i, 0] = float(alpha * pl + (1.0 - alpha) * pg)
        blend_weight[i] = float(alpha)

    piece_metrics = _metrics(y_test, pred_piece)
    piece_blend_metrics = _metrics(y_test, pred_blend)
    selected_global_metrics = _metrics(y_test, pred_global_for_blend)
    piece_sec = float(time.perf_counter() - t0)

    return {
        "feature_names_used": {
            "gate": list(gate_names),
            "param": list(param_names),
        },
        "regime_mode": regime_mode,
        "regime_keys_all": [list(k) for k in regime_keys_all],
        "train_regime_counts": {str(k): int(v) for k, v in count_train.items()},
        "test_regime_counts": {str(k): int(v) for k, v in count_test.items()},
        "rare_merge_group": [list(k) for k in rare_merge_group],
        "rare_merge_group_used": bool(merged_group_used and regime_mode not in {"strict4", "strict3"}),
        "local_models_trained": [str(k) for k in local_models.keys()],
        "local_effective_samples": {str(k): int(v) for k, v in local_effective_samples.items()},
        "regime_training_detail": regime_training_detail,
        "strict4_parallel_runtime": {
            "requested_mode": str(spec.strict4_parallel_mode),
            "effective_mode": str(strict4_parallel_mode_effective),
            "requested_workers": int(max(1, int(spec.strict4_max_workers))),
            "effective_workers": int(max(1, int(strict4_parallel_workers_effective))),
            "gpu_strategy": str(spec.strict4_gpu_strategy),
            "gpu_devices": list(spec.strict4_gpu_devices),
            "gpu_assigned_by_regime": strict4_gpu_assignments,
        },
        "strict4_dynamic_merge": {
            "enabled": bool(spec.strict4_dynamic_merge_enabled),
            "min_samples": int(max(1, int(spec.strict4_dynamic_merge_min_samples))),
            "used": bool(strict4_dynamic_merge_used),
            "detail": dict(strict4_dynamic_merge_detail),
        },
        "prediction_fallback_count": int(fallback_count),
        "blend": {
            "enabled": bool(spec.blend_with_global),
            "kappa": float(spec.blend_kappa),
            "kappa_mode": (
                "per_regime"
                if regime_mode in {"strict4", "strict3"} and len(spec.strict4_branch_hparams) > 0
                else "global"
            ),
            "kappa_by_regime": {
                str(k): float(local_blend_kappa_by_regime.get(k, float(spec.blend_kappa))) for k in regime_keys_all
            },
            "mean_alpha_test": float(np.mean(blend_weight)) if int(blend_weight.size) > 0 else 0.0,
            "min_alpha_test": float(np.min(blend_weight)) if int(blend_weight.size) > 0 else 0.0,
            "max_alpha_test": float(np.max(blend_weight)) if int(blend_weight.size) > 0 else 0.0,
            "global_backbone": _jsonable(global_backbone_choice),
        },
        "strict4_branch_hparams_input": _jsonable(spec.strict4_branch_hparams),
        "global_hparams_used": _jsonable(resolved_global["settings_raw"]),
        "metrics": {
            "xgboost_global": {"metrics_test": xgb_metrics, "duration_sec": xgb_sec},
            "symbolic_stagewise_global": {"metrics_test": global_stage_metrics, "duration_sec": global_stage_sec},
            "global_backbone_selected": {"metrics_test": selected_global_metrics, "duration_sec": global_stage_sec},
            "symbolic_stagewise_fixed_piecewise": {"metrics_test": piece_metrics, "duration_sec": piece_sec},
            "symbolic_stagewise_fixed_piecewise_blended": {
                "metrics_test": piece_blend_metrics,
                "duration_sec": piece_sec,
            },
        },
        "delta_piecewise_vs_global_stagewise_rmse": float(piece_metrics["rmse"] - global_stage_metrics["rmse"]),
        "delta_piecewise_blended_vs_global_stagewise_rmse": float(
            piece_blend_metrics["rmse"] - global_stage_metrics["rmse"]
        ),
        "delta_piecewise_vs_xgboost_rmse": float(piece_metrics["rmse"] - xgb_metrics["rmse"]),
        "delta_piecewise_blended_vs_xgboost_rmse": float(piece_blend_metrics["rmse"] - xgb_metrics["rmse"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Rolling time-split evaluation for fixed holiday piecewise models.")
    parser.add_argument(
        "--csv-path",
        type=str,
        default=default_work_ci_csv(),
    )
    parser.add_argument("--target-col", type=str, default="ci")
    parser.add_argument("--date-col", type=str, default="date")
    parser.add_argument("--split-mode", type=str, default="expanding", choices=["expanding", "sliding"])
    parser.add_argument("--min-train-size", type=int, default=600)
    parser.add_argument("--test-size", type=int, default=120)
    parser.add_argument("--step-size", type=int, default=120)
    parser.add_argument("--train-window-size", type=int, default=720)
    parser.add_argument("--regime-mode", type=str, default="legacy", choices=["legacy", "strict4", "strict3"])

    parser.add_argument("--min-leaf", type=int, default=64)
    parser.add_argument("--blend-kappa", type=float, default=512.0)
    parser.add_argument("--disable-merge-rare-holiday-regimes", action="store_true")
    parser.add_argument("--disable-confidence-blend", action="store_true")
    parser.add_argument("--local-search-force-linear-base", type=str, default="auto", choices=["auto", "on", "off"])
    parser.add_argument("--local-search-topk-features", type=int, default=8)
    parser.add_argument("--local-search-max-added-terms", type=int, default=12)
    parser.add_argument("--local-search-max-pair-terms", type=int, default=16)
    parser.add_argument("--local-search-max-candidates-per-iter", type=int, default=500)
    parser.add_argument("--local-search-candidate-keep-top", type=int, default=12)
    parser.add_argument("--local-search-ridge-l2", type=float, default=1e-4)
    parser.add_argument("--local-search-unary-ops", type=str, default="square,sin,cos,tanh")
    parser.add_argument("--local-search-nested-unary-patterns", type=str, default="sin(square),cos(square)")
    parser.add_argument("--local-search-overfit-guard-enabled", action="store_true")
    parser.add_argument("--local-search-overfit-guard-val-ratio", type=float, default=0.2)
    parser.add_argument("--local-search-overfit-guard-min-val-samples", type=int, default=64)
    parser.add_argument("--local-search-overfit-guard-min-val-rmse-gain", type=float, default=0.0)
    parser.add_argument("--local-search-overfit-guard-max-gap-increase", type=float, default=0.05)
    parser.add_argument("--local-search-overfit-guard-patience", type=int, default=3)
    parser.add_argument(
        "--local-search-interaction-budget-mode",
        type=str,
        default="fixed",
        choices=["fixed", "interaction_first"],
    )
    parser.add_argument("--local-search-interaction-diag-threshold", type=float, default=1.15)
    parser.add_argument("--local-search-interaction-diag-topk-features", type=int, default=8)
    parser.add_argument("--local-search-interaction-pair-budget-boost", type=float, default=2.0)
    parser.add_argument("--local-search-interaction-grad-projection-budget-boost", type=float, default=1.5)
    parser.add_argument("--local-search-inner-opt-enabled", action="store_true")
    parser.add_argument(
        "--local-search-inner-opt-method",
        type=str,
        default="adam_lbfgs",
        choices=["adam_lbfgs", "adam", "lbfgs"],
    )
    parser.add_argument("--local-search-inner-opt-device", type=str, default="auto")
    parser.add_argument("--local-search-inner-opt-adam-steps", type=int, default=120)
    parser.add_argument("--local-search-inner-opt-adam-lr", type=float, default=5e-3)
    parser.add_argument("--local-search-inner-opt-lbfgs-steps", type=int, default=60)
    parser.add_argument("--local-search-inner-opt-lbfgs-lr", type=float, default=0.8)
    parser.add_argument("--local-search-inner-opt-l2", type=float, default=0.0)
    parser.add_argument("--local-search-inner-opt-accept-rmse-tol", type=float, default=1e-6)
    parser.add_argument("--small-sample-guard-threshold", type=int, default=0)
    parser.add_argument(
        "--blend-global-backbone-mode",
        type=str,
        default="symbolic_only",
        choices=["symbolic_only", "best_of_symbolic_xgboost"],
    )
    parser.add_argument("--blend-global-backbone-val-ratio", type=float, default=0.2)
    parser.add_argument("--blend-global-backbone-min-val-samples", type=int, default=64)
    parser.add_argument("--blend-global-backbone-margin", type=float, default=0.0)
    parser.add_argument(
        "--strict4-branch-hparams-json",
        type=str,
        default="",
        help=(
            "JSON object or JSON file path for strict4 per-branch overrides. "
            "keys: holiday_near, holiday_mid, weekend, regular."
        ),
    )
    parser.add_argument(
        "--strict4-parallel-mode",
        type=str,
        default="serial",
        choices=["serial", "thread", "process"],
        help="Parallel mode for strict4 branch local model training.",
    )
    parser.add_argument("--strict4-max-workers", type=int, default=1)
    parser.add_argument(
        "--strict4-gpu-strategy",
        type=str,
        default="none",
        choices=["none", "fixed", "round_robin", "auto"],
        help="GPU assignment strategy for strict4 branch local model training.",
    )
    parser.add_argument(
        "--strict4-gpu-devices",
        type=str,
        default="",
        help="CSV GPU devices for strict4 strategy, e.g. '0,1' or 'cuda:0,cuda:1'.",
    )
    parser.add_argument(
        "--strict4-dynamic-merge-enabled",
        action="store_true",
        help="Enable dynamic merge for strict4 holiday_near/holiday_mid when one branch is sample-starved.",
    )
    parser.add_argument(
        "--strict4-dynamic-merge-min-samples",
        type=int,
        default=64,
        help="Threshold for strict4 dynamic merge trigger (holiday_near or holiday_mid < threshold).",
    )
    args = parser.parse_args()
    regime_mode = str(args.regime_mode).strip().lower()
    merge_rare_enabled = (not bool(args.disable_merge_rare_holiday_regimes)) and regime_mode not in {"strict4", "strict3"}
    strict4_branch_hparams = _parse_strict4_branch_hparams_json(str(args.strict4_branch_hparams_json))
    strict4_parallel_mode = _normalize_parallel_mode(str(args.strict4_parallel_mode))
    strict4_gpu_strategy = _normalize_gpu_strategy(str(args.strict4_gpu_strategy))
    strict4_gpu_devices = _parse_gpu_devices(str(args.strict4_gpu_devices))

    spec = RollingSpec(
        gate_features=(
            "is_holiday_day_or_window",
            "is_holiday_near",
            "is_holiday_mid",
            "is_nonwork_weekend",
        ),
        param_features=(
            "avg_occ",
            "avg_speed",
            "total_flow",
            "aqi",
            "wind",
            "is_bad_weather",
            "weather_dummy",
            "life_impact",
        ),
        regime_mode=regime_mode,
        min_leaf=int(max(20, args.min_leaf)),
        merge_rare_holiday_regimes=bool(merge_rare_enabled),
        blend_with_global=not bool(args.disable_confidence_blend),
        blend_kappa=float(max(1e-6, args.blend_kappa)),
        local_search_force_linear_base=str(args.local_search_force_linear_base).strip().lower(),
        local_search_topk_features=int(max(1, args.local_search_topk_features)),
        local_search_max_added_terms=int(max(0, args.local_search_max_added_terms)),
        local_search_max_pair_terms=int(max(0, args.local_search_max_pair_terms)),
        local_search_max_candidates_per_iter=int(max(1, args.local_search_max_candidates_per_iter)),
        local_search_candidate_keep_top=int(max(1, args.local_search_candidate_keep_top)),
        local_search_ridge_l2=float(max(0.0, args.local_search_ridge_l2)),
        local_search_unary_ops=tuple(_parse_csv_list(args.local_search_unary_ops)),
        local_search_nested_unary_patterns=tuple(_parse_csv_list(args.local_search_nested_unary_patterns)),
        local_search_overfit_guard_enabled=bool(args.local_search_overfit_guard_enabled),
        local_search_overfit_guard_val_ratio=float(np.clip(args.local_search_overfit_guard_val_ratio, 0.0, 0.9)),
        local_search_overfit_guard_min_val_samples=int(max(1, args.local_search_overfit_guard_min_val_samples)),
        local_search_overfit_guard_min_val_rmse_gain=float(max(0.0, args.local_search_overfit_guard_min_val_rmse_gain)),
        local_search_overfit_guard_max_gap_increase=float(max(0.0, args.local_search_overfit_guard_max_gap_increase)),
        local_search_overfit_guard_patience=int(max(0, args.local_search_overfit_guard_patience)),
        local_search_interaction_budget_mode=str(args.local_search_interaction_budget_mode).strip().lower(),
        local_search_interaction_diag_threshold=float(max(0.0, args.local_search_interaction_diag_threshold)),
        local_search_interaction_diag_topk_features=int(max(1, args.local_search_interaction_diag_topk_features)),
        local_search_interaction_pair_budget_boost=float(max(1.0, args.local_search_interaction_pair_budget_boost)),
        local_search_interaction_grad_projection_budget_boost=float(
            max(1.0, args.local_search_interaction_grad_projection_budget_boost)
        ),
        small_sample_guard_threshold=int(max(0, args.small_sample_guard_threshold)),
        blend_global_backbone_mode=str(args.blend_global_backbone_mode).strip().lower(),
        blend_global_backbone_val_ratio=float(np.clip(args.blend_global_backbone_val_ratio, 0.05, 0.45)),
        blend_global_backbone_min_val_samples=int(max(16, args.blend_global_backbone_min_val_samples)),
        blend_global_backbone_margin=float(max(0.0, args.blend_global_backbone_margin)),
        local_search_inner_opt_enabled=bool(args.local_search_inner_opt_enabled),
        local_search_inner_opt_method=str(args.local_search_inner_opt_method).strip().lower(),
        local_search_inner_opt_device=str(args.local_search_inner_opt_device).strip().lower(),
        local_search_inner_opt_adam_steps=int(max(0, args.local_search_inner_opt_adam_steps)),
        local_search_inner_opt_adam_lr=float(max(1e-8, args.local_search_inner_opt_adam_lr)),
        local_search_inner_opt_lbfgs_steps=int(max(0, args.local_search_inner_opt_lbfgs_steps)),
        local_search_inner_opt_lbfgs_lr=float(max(1e-8, args.local_search_inner_opt_lbfgs_lr)),
        local_search_inner_opt_l2=float(max(0.0, args.local_search_inner_opt_l2)),
        local_search_inner_opt_accept_rmse_tol=float(max(0.0, args.local_search_inner_opt_accept_rmse_tol)),
        strict4_parallel_mode=str(strict4_parallel_mode),
        strict4_max_workers=int(max(1, int(args.strict4_max_workers))),
        strict4_gpu_strategy=str(strict4_gpu_strategy),
        strict4_gpu_devices=tuple(strict4_gpu_devices),
        strict4_branch_hparams=strict4_branch_hparams,
        strict4_dynamic_merge_enabled=bool(args.strict4_dynamic_merge_enabled),
        strict4_dynamic_merge_min_samples=int(max(1, int(args.strict4_dynamic_merge_min_samples))),
    )

    X_all, y_all, feature_names, dates = _load_table(
        csv_path=str(args.csv_path),
        target_col=str(args.target_col),
        date_col=str(args.date_col),
    )
    n_total = int(X_all.shape[0])
    splits = _build_rolling_splits(
        n_samples=n_total,
        min_train_size=int(args.min_train_size),
        test_size=int(args.test_size),
        step_size=int(args.step_size),
        split_mode=str(args.split_mode),
        train_window_size=int(args.train_window_size),
    )
    if not splits:
        raise ValueError("No valid rolling splits generated. Check split parameters.")

    fold_rows: list[dict[str, Any]] = []
    for s in splits:
        sid = int(s["split_id"])
        tr0, tr1 = int(s["train_start"]), int(s["train_end"])
        te0, te1 = int(s["test_start"]), int(s["test_end"])
        tag = f"s{sid}_tr{tr0}_{tr1}_te{te0}_{te1}"
        print(
            f"[split {sid}] train={tr0}:{tr1} ({str(dates[tr0])[:10]}..{str(dates[tr1 - 1])[:10]}) "
            f"test={te0}:{te1} ({str(dates[te0])[:10]}..{str(dates[te1 - 1])[:10]})"
        )

        result = _evaluate_one_split(
            split_tag=tag,
            X_train=np.asarray(X_all[tr0:tr1], dtype=float),
            y_train=np.asarray(y_all[tr0:tr1], dtype=float),
            X_test=np.asarray(X_all[te0:te1], dtype=float),
            y_test=np.asarray(y_all[te0:te1], dtype=float),
            feature_names=feature_names,
            spec=spec,
        )
        row = {
            "split_id": sid,
            "train_range": {
                "start_idx": tr0,
                "end_idx_exclusive": tr1,
                "start_date": str(dates[tr0])[:10],
                "end_date": str(dates[tr1 - 1])[:10],
                "n_train": int(tr1 - tr0),
            },
            "test_range": {
                "start_idx": te0,
                "end_idx_exclusive": te1,
                "start_date": str(dates[te0])[:10],
                "end_date": str(dates[te1 - 1])[:10],
                "n_test": int(te1 - te0),
            },
            "summary": result,
        }
        fold_rows.append(row)

    def _metric_arr(path: list[str]) -> list[float]:
        out: list[float] = []
        for r in fold_rows:
            cur: Any = r
            for p in path:
                cur = cur[p]
            out.append(float(cur))
        return out

    xgb_rmse = _metric_arr(["summary", "metrics", "xgboost_global", "metrics_test", "rmse"])
    global_rmse = _metric_arr(["summary", "metrics", "symbolic_stagewise_global", "metrics_test", "rmse"])
    piece_rmse = _metric_arr(["summary", "metrics", "symbolic_stagewise_fixed_piecewise", "metrics_test", "rmse"])
    blend_rmse = _metric_arr(["summary", "metrics", "symbolic_stagewise_fixed_piecewise_blended", "metrics_test", "rmse"])

    aggregate = {
        "xgboost_global_rmse": {
            "mean": _safe_mean(xgb_rmse),
            "std": _safe_std(xgb_rmse),
            "median": _safe_median(xgb_rmse),
            "min": float(np.min(np.asarray(xgb_rmse, dtype=float))),
            "max": float(np.max(np.asarray(xgb_rmse, dtype=float))),
        },
        "symbolic_stagewise_global_rmse": {
            "mean": _safe_mean(global_rmse),
            "std": _safe_std(global_rmse),
            "median": _safe_median(global_rmse),
            "min": float(np.min(np.asarray(global_rmse, dtype=float))),
            "max": float(np.max(np.asarray(global_rmse, dtype=float))),
        },
        "symbolic_stagewise_fixed_piecewise_rmse": {
            "mean": _safe_mean(piece_rmse),
            "std": _safe_std(piece_rmse),
            "median": _safe_median(piece_rmse),
            "min": float(np.min(np.asarray(piece_rmse, dtype=float))),
            "max": float(np.max(np.asarray(piece_rmse, dtype=float))),
        },
        "symbolic_stagewise_fixed_piecewise_blended_rmse": {
            "mean": _safe_mean(blend_rmse),
            "std": _safe_std(blend_rmse),
            "median": _safe_median(blend_rmse),
            "min": float(np.min(np.asarray(blend_rmse, dtype=float))),
            "max": float(np.max(np.asarray(blend_rmse, dtype=float))),
        },
        "wins": {
            "blend_better_than_xgboost_count": int(
                sum(1 for b, x in zip(blend_rmse, xgb_rmse) if float(b) < float(x))
            ),
            "blend_better_than_global_count": int(
                sum(1 for b, g in zip(blend_rmse, global_rmse) if float(b) < float(g))
            ),
            "n_splits": int(len(fold_rows)),
        },
    }

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = ROOT / "examples" / "out" / f"work_ci_fixed_holiday_rolling_eval_{stamp}"
    out_root.mkdir(parents=True, exist_ok=True)
    out_path = out_root / "summary.json"

    summary = {
        "timestamp": datetime.now().isoformat(),
        "output_root": str(out_root),
        "dataset": {
            "source": str(args.csv_path),
            "target_col": str(args.target_col),
            "date_col": str(args.date_col),
            "n_total": int(n_total),
            "date_min": str(dates[0])[:10],
            "date_max": str(dates[-1])[:10],
            "feature_count": int(len(feature_names)),
        },
        "rolling_config": {
            "split_mode": str(args.split_mode),
            "min_train_size": int(args.min_train_size),
            "test_size": int(args.test_size),
            "step_size": int(args.step_size),
            "train_window_size": int(args.train_window_size),
            "n_splits": int(len(splits)),
        },
        "model_config": _jsonable(asdict(spec)),
        "splits": _jsonable(fold_rows),
        "aggregate": _jsonable(aggregate),
    }
    out_path.write_text(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2), encoding="utf-8")

    print("WORK_CI_FIXED_HOLIDAY_ROLLING_EVAL_DONE")
    print(f"output_root={out_root}")
    print("rolling rmse mean:")
    print(f"  xgboost_global={aggregate['xgboost_global_rmse']['mean']:.6f}")
    print(f"  symbolic_stagewise_global={aggregate['symbolic_stagewise_global_rmse']['mean']:.6f}")
    print(f"  symbolic_stagewise_fixed_piecewise={aggregate['symbolic_stagewise_fixed_piecewise_rmse']['mean']:.6f}")
    print(
        "  symbolic_stagewise_fixed_piecewise_blended="
        f"{aggregate['symbolic_stagewise_fixed_piecewise_blended_rmse']['mean']:.6f}"
    )
    print(
        "wins: "
        f"blend<xgboost {aggregate['wins']['blend_better_than_xgboost_count']}/{aggregate['wins']['n_splits']}, "
        f"blend<global {aggregate['wins']['blend_better_than_global_count']}/{aggregate['wins']['n_splits']}"
    )
    print(f"summary={out_path}")


if __name__ == "__main__":
    main()
