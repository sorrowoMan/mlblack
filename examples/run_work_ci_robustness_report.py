from __future__ import annotations

import argparse
import itertools
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
from examples.work_ci_reader import WorkCiIntervalReader  # noqa: E402

_ART_COUNTER = itertools.count(1)


def _artifact_id(prefix: str, tag: str) -> str:
    return f"{prefix}_{tag}_{next(_ART_COUNTER)}"


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


def _parse_float_list(raw: str) -> tuple[float, ...]:
    txt = str(raw).strip()
    if not txt:
        return tuple()
    out: list[float] = []
    for p in txt.split(","):
        pp = p.strip()
        if not pp:
            continue
        out.append(float(pp))
    return tuple(out)


@dataclass(frozen=True)
class ModelSpec:
    gate_features: tuple[str, ...]
    param_features: tuple[str, ...]
    min_leaf: int
    merge_rare_holiday_regimes: bool
    blend_with_global: bool
    blend_kappa: float
    local_search_topk_features: int
    local_search_max_added_terms: int
    local_search_max_pair_terms: int
    local_search_max_candidates_per_iter: int
    local_search_candidate_keep_top: int
    local_search_unary_ops: tuple[str, ...]
    local_search_nested_unary_patterns: tuple[str, ...]


@dataclass
class PiecewiseBlendedModel:
    gate_idx: tuple[int, ...]
    param_idx: tuple[int, ...]
    global_artifact: Any
    local_models: dict[tuple[int, ...], Any]
    local_effective_samples: dict[tuple[int, ...], int]
    blend_kappa: float
    blend_with_global: bool
    training_detail: dict[str, Any]

    def predict(self, X: np.ndarray) -> np.ndarray:
        x = np.asarray(X, dtype=float)
        x_param = _slice_cols(x, self.param_idx)
        x_gate = _slice_cols(x, self.gate_idx)
        keys = _gate_key(x_gate)

        pred_global = np.asarray(self.global_artifact.predict(x_param), dtype=float).reshape(-1, 1)
        out = np.zeros((x.shape[0], 1), dtype=float)
        for i, k in enumerate(keys):
            pg = float(pred_global[i, 0])
            art = self.local_models.get(k)
            if art is None:
                pl = pg
                alpha = 0.0
            else:
                pl = float(np.asarray(art.predict(x_param[i : i + 1, :]), dtype=float).reshape(-1)[0])
                n_eff = float(self.local_effective_samples.get(k, 0))
                if self.blend_with_global:
                    alpha = float(n_eff / (n_eff + float(self.blend_kappa)))
                else:
                    alpha = 1.0
            out[i, 0] = float(alpha * pl + (1.0 - alpha) * pg)
        return out


def _load_fold_split(
    *,
    csv_path: str,
    target_col: str,
    test_fold_col: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[str, ...]]:
    reader = WorkCiIntervalReader(
        csv_path=str(csv_path),
        target_col=str(target_col),
        test_fold_col=str(test_fold_col),
    )
    bundle = reader.read()
    tr = bundle.train
    te = bundle.test
    if te is None:
        raise ValueError("No test split from WorkCiIntervalReader")
    feature_names = tuple(str(v) for v in tr.feature_names)
    return (
        np.asarray(tr.X_train, dtype=float),
        np.asarray(tr.y_train, dtype=float).reshape(-1, 1),
        np.asarray(te.X_train, dtype=float),
        np.asarray(te.y_train, dtype=float).reshape(-1, 1),
        feature_names,
    )


def _load_full_table(
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

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    if df[date_col].isna().any():
        raise ValueError(f"Found invalid dates in '{date_col}'")
    df = df.sort_values(date_col).reset_index(drop=True)

    fold_cols = [c for c in df.columns if str(c).startswith("test_fold_")]
    drop_cols = set(fold_cols)
    drop_cols.add(target_col)
    drop_cols.add(date_col)
    feature_cols = [str(c) for c in df.columns if c not in drop_cols]
    if not feature_cols:
        raise ValueError("No feature columns available after dropping date/target/test_fold_*")

    X_df = df[feature_cols].copy()
    y_sr = pd.to_numeric(df[target_col], errors="coerce")
    for c in feature_cols:
        X_df[c] = pd.to_numeric(X_df[c], errors="coerce")
    if X_df.isna().any().any() or y_sr.isna().any():
        bad_cols = [c for c in feature_cols if X_df[c].isna().any()]
        if y_sr.isna().any():
            bad_cols.append(target_col)
        raise ValueError(f"Found NaN after numeric conversion. Columns: {bad_cols}")

    X_all = X_df.to_numpy(dtype=float)
    y_all = y_sr.to_numpy(dtype=float).reshape(-1, 1)
    dates = df[date_col].to_numpy()
    return X_all, y_all, tuple(feature_cols), dates


def _build_model(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: tuple[str, ...],
    spec: ModelSpec,
    tag: str,
) -> PiecewiseBlendedModel:
    gate_idx = _col_index(feature_names, spec.gate_features)
    param_idx = _col_index(feature_names, spec.param_features)
    if len(gate_idx) == 0:
        raise ValueError("No holiday gate features found in dataset.")
    if len(param_idx) == 0:
        raise ValueError("No parameter features found in dataset.")

    gate_train = _slice_cols(X_train, gate_idx)
    x_train_param = _slice_cols(X_train, param_idx)
    param_names = tuple(feature_names[i] for i in param_idx)
    key_train = _gate_key(gate_train)

    global_art = _fit_artifact(
        trainer_key="symbolic_stagewise",
        trainer_params={
            "artifact_id": _artifact_id("robust_global_stage", tag),
            "force_linear_base": "auto",
            "keep_search_trace": False,
            "auto_val_ratio": 0.2,
            "auto_min_val_samples": 64,
            "auto_random_seed": 42,
            "search_max_added_terms": 8,
            "search_topk_features": min(8, len(param_names)),
            "search_max_pair_terms": 8,
            "search_max_candidates_per_iter": 180,
            "search_candidate_keep_top": 8,
            "search_include_hinge": True,
            "search_hinge_quantiles": [0.25, 0.5, 0.75],
            "search_unary_ops": ["square", "sin", "cos", "tanh"],
            "search_nested_unary_patterns": ["sin(square)", "cos(square)"],
            "search_enable_prune": True,
            "search_prune_rmse_tolerance": 1e-6,
            "search_prune_max_removed_per_iter": 1,
            "search_path_memory_enabled": False,
            "search_min_actual_rmse_gain": 0.0,
        },
        X=x_train_param,
        y=y_train,
        feature_names=param_names,
    )

    local_unary_ops = (
        tuple(spec.local_search_unary_ops) if len(spec.local_search_unary_ops) > 0 else ("square", "sin", "cos", "tanh")
    )
    local_nested = tuple(spec.local_search_nested_unary_patterns)

    local_models: dict[tuple[int, ...], Any] = {}
    local_effective_samples: dict[tuple[int, ...], int] = {}
    regime_index = _build_regime_index(key_train)
    regime_keys_all = tuple(sorted(set(key_train)))
    regime_training_detail: dict[str, Any] = {}
    rare_merge_group = ((1, 1, 0, 0), (1, 0, 1, 0))
    merged_group_used = False

    if spec.merge_rare_holiday_regimes:
        group_keys = tuple(k for k in rare_merge_group if k in regime_keys_all)
        if len(group_keys) >= 2:
            parts = [np.asarray(regime_index.get(k, np.asarray([], dtype=int)), dtype=int) for k in group_keys]
            parts = [p for p in parts if int(p.size) > 0]
            if parts:
                idx_merge = np.unique(np.concatenate(parts, axis=0).astype(int, copy=False))
                if int(idx_merge.size) > 0:
                    shared_art = _fit_artifact(
                        trainer_key="symbolic_stagewise",
                        trainer_params={
                            "artifact_id": _artifact_id("robust_local_shared", tag),
                            "force_linear_base": "auto",
                            "keep_search_trace": False,
                            "auto_val_ratio": 0.2,
                            "auto_min_val_samples": 32,
                            "auto_random_seed": 42,
                            "search_max_added_terms": int(spec.local_search_max_added_terms),
                            "search_topk_features": min(int(spec.local_search_topk_features), len(param_names)),
                            "search_max_pair_terms": int(spec.local_search_max_pair_terms),
                            "search_max_candidates_per_iter": int(spec.local_search_max_candidates_per_iter),
                            "search_candidate_keep_top": int(spec.local_search_candidate_keep_top),
                            "search_include_hinge": True,
                            "search_hinge_quantiles": [0.25, 0.5, 0.75],
                            "search_unary_ops": list(local_unary_ops),
                            "search_nested_unary_patterns": list(local_nested),
                            "search_enable_prune": True,
                            "search_prune_rmse_tolerance": 1e-6,
                            "search_prune_max_removed_per_iter": 1,
                            "search_path_memory_enabled": False,
                            "search_min_actual_rmse_gain": 0.0,
                        },
                        X=x_train_param[idx_merge],
                        y=y_train[idx_merge],
                        feature_names=param_names,
                    )
                    merged_group_used = True
                    used_from = {
                        str(k): int(np.asarray(regime_index.get(k, np.asarray([], dtype=int)), dtype=int).size)
                        for k in group_keys
                    }
                    for k in group_keys:
                        local_models[k] = shared_art
                        local_effective_samples[k] = int(idx_merge.size)
                        regime_training_detail[str(k)] = {
                            "target": list(k),
                            "exact_count": int(
                                np.asarray(regime_index.get(k, np.asarray([], dtype=int)), dtype=int).size
                            ),
                            "used_count": int(idx_merge.size),
                            "used_from": used_from,
                            "shared_model_with": [list(x) for x in group_keys],
                        }

    for k in regime_keys_all:
        if k in local_models:
            continue
        idx_use, use_detail = _select_training_indices_for_regime(
            target_key=k,
            regime_index=regime_index,
            min_leaf=int(spec.min_leaf),
        )
        art = _fit_artifact(
            trainer_key="symbolic_stagewise",
            trainer_params={
                "artifact_id": _artifact_id("robust_local", tag),
                "force_linear_base": "auto",
                "keep_search_trace": False,
                "auto_val_ratio": 0.2,
                "auto_min_val_samples": 32,
                "auto_random_seed": 42,
                "search_max_added_terms": int(spec.local_search_max_added_terms),
                "search_topk_features": min(int(spec.local_search_topk_features), len(param_names)),
                "search_max_pair_terms": int(spec.local_search_max_pair_terms),
                "search_max_candidates_per_iter": int(spec.local_search_max_candidates_per_iter),
                "search_candidate_keep_top": int(spec.local_search_candidate_keep_top),
                "search_include_hinge": True,
                "search_hinge_quantiles": [0.25, 0.5, 0.75],
                "search_unary_ops": list(local_unary_ops),
                "search_nested_unary_patterns": list(local_nested),
                "search_enable_prune": True,
                "search_prune_rmse_tolerance": 1e-6,
                "search_prune_max_removed_per_iter": 1,
                "search_path_memory_enabled": False,
                "search_min_actual_rmse_gain": 0.0,
            },
            X=x_train_param[idx_use],
            y=y_train[idx_use],
            feature_names=param_names,
        )
        local_models[k] = art
        local_effective_samples[k] = int(use_detail.get("used_count", int(idx_use.size)))
        regime_training_detail[str(k)] = use_detail

    training_detail = {
        "gate_features_used": [feature_names[i] for i in gate_idx],
        "param_features_used": [feature_names[i] for i in param_idx],
        "train_regime_counts": {str(k): int(v) for k, v in dict(Counter(key_train)).items()},
        "rare_merge_group": [list(k) for k in rare_merge_group],
        "rare_merge_group_used": bool(merged_group_used),
        "local_effective_samples": {str(k): int(v) for k, v in local_effective_samples.items()},
        "regime_training_detail": regime_training_detail,
    }

    return PiecewiseBlendedModel(
        gate_idx=gate_idx,
        param_idx=param_idx,
        global_artifact=global_art,
        local_models=local_models,
        local_effective_samples=local_effective_samples,
        blend_kappa=float(spec.blend_kappa),
        blend_with_global=bool(spec.blend_with_global),
        training_detail=training_detail,
    )


def _compute_param_stats(
    *,
    X_train: np.ndarray,
    param_idx: tuple[int, ...],
) -> dict[str, np.ndarray]:
    x_param = _slice_cols(X_train, param_idx)
    med = np.median(x_param, axis=0)
    std = np.std(x_param, axis=0)
    nonbinary_mask = np.zeros((x_param.shape[1],), dtype=bool)
    for j in range(x_param.shape[1]):
        vals = np.unique(x_param[:, j])
        is_binary = bool(np.all(np.isclose(vals, 0.0) | np.isclose(vals, 1.0)))
        nonbinary_mask[j] = (not is_binary)
    return {
        "median": np.asarray(med, dtype=float),
        "std": np.asarray(std, dtype=float),
        "nonbinary_mask": np.asarray(nonbinary_mask, dtype=bool),
    }


def _evaluate_noise_stability(
    *,
    model: PiecewiseBlendedModel,
    X_test: np.ndarray,
    y_test: np.ndarray,
    param_stats: dict[str, np.ndarray],
    noise_levels: tuple[float, ...],
    repeats: int,
    random_seed: int,
    clean_rmse: float,
) -> dict[str, Any]:
    std = np.asarray(param_stats["std"], dtype=float)
    nonbinary_mask = np.asarray(param_stats["nonbinary_mask"], dtype=bool)
    rows: list[dict[str, Any]] = []

    for level in noise_levels:
        rmse_list: list[float] = []
        for r in range(int(repeats)):
            rng = np.random.default_rng(int(random_seed) + 10000 * int(level * 1000) + r)
            x = np.asarray(X_test, dtype=float).copy()
            x_param = _slice_cols(x, model.param_idx)
            for j in range(x_param.shape[1]):
                if not bool(nonbinary_mask[j]):
                    continue
                sigma = float(level) * float(max(std[j], 1e-8))
                x_param[:, j] = x_param[:, j] + rng.normal(0.0, sigma, size=x_param.shape[0])
            x[:, list(model.param_idx)] = x_param
            pred = model.predict(x)
            rmse = float(_metrics(y_test, pred)["rmse"])
            rmse_list.append(rmse)

        mean_rmse = float(np.mean(np.asarray(rmse_list, dtype=float)))
        rows.append(
            {
                "noise_level": float(level),
                "rmse_mean": mean_rmse,
                "rmse_std": float(np.std(np.asarray(rmse_list, dtype=float), ddof=0)),
                "degradation_ratio_vs_clean": float((mean_rmse - float(clean_rmse)) / max(float(clean_rmse), 1e-12)),
                "rmse_runs": [float(v) for v in rmse_list],
            }
        )

    return {
        "clean_rmse": float(clean_rmse),
        "noise_levels": [float(v) for v in noise_levels],
        "repeats": int(repeats),
        "rows": rows,
        "avg_degradation_ratio": float(np.mean(np.asarray([r["degradation_ratio_vs_clean"] for r in rows], dtype=float)))
        if rows
        else 0.0,
    }


def _evaluate_missing_tolerance(
    *,
    model: PiecewiseBlendedModel,
    X_test: np.ndarray,
    y_test: np.ndarray,
    param_stats: dict[str, np.ndarray],
    missing_rates: tuple[float, ...],
    repeats: int,
    random_seed: int,
    clean_rmse: float,
) -> dict[str, Any]:
    med = np.asarray(param_stats["median"], dtype=float)
    rows: list[dict[str, Any]] = []

    for rate in missing_rates:
        rmse_list: list[float] = []
        for r in range(int(repeats)):
            rng = np.random.default_rng(int(random_seed) + 200000 + 10000 * int(rate * 1000) + r)
            x = np.asarray(X_test, dtype=float).copy()
            x_param = _slice_cols(x, model.param_idx)
            mask = rng.random(size=x_param.shape) < float(rate)
            if np.any(mask):
                fill = np.broadcast_to(med.reshape(1, -1), x_param.shape)
                x_param = np.where(mask, fill, x_param)
            x[:, list(model.param_idx)] = x_param
            pred = model.predict(x)
            rmse = float(_metrics(y_test, pred)["rmse"])
            rmse_list.append(rmse)

        mean_rmse = float(np.mean(np.asarray(rmse_list, dtype=float)))
        rows.append(
            {
                "missing_rate": float(rate),
                "rmse_mean": mean_rmse,
                "rmse_std": float(np.std(np.asarray(rmse_list, dtype=float), ddof=0)),
                "degradation_ratio_vs_clean": float((mean_rmse - float(clean_rmse)) / max(float(clean_rmse), 1e-12)),
                "rmse_runs": [float(v) for v in rmse_list],
            }
        )

    return {
        "clean_rmse": float(clean_rmse),
        "missing_rates": [float(v) for v in missing_rates],
        "repeats": int(repeats),
        "rows": rows,
        "avg_degradation_ratio": float(np.mean(np.asarray([r["degradation_ratio_vs_clean"] for r in rows], dtype=float)))
        if rows
        else 0.0,
    }


def _evaluate_drift_resistance(
    *,
    csv_path: str,
    target_col: str,
    date_col: str,
    spec: ModelSpec,
    drift_train_size: int,
    drift_window_size: int,
    drift_step_size: int,
) -> dict[str, Any]:
    X_all, y_all, feature_names, dates = _load_full_table(
        csv_path=str(csv_path),
        target_col=str(target_col),
        date_col=str(date_col),
    )
    n = int(X_all.shape[0])
    train_n = int(max(1, min(int(drift_train_size), n - 1)))
    if train_n + int(drift_window_size) > n:
        raise ValueError("drift_train_size + drift_window_size exceeds total sample count")

    model = _build_model(
        X_train=np.asarray(X_all[:train_n], dtype=float),
        y_train=np.asarray(y_all[:train_n], dtype=float),
        feature_names=feature_names,
        spec=spec,
        tag="drift",
    )

    holiday_idx: int | None = None
    if "is_holiday_day_or_window" in feature_names:
        holiday_idx = int(feature_names.index("is_holiday_day_or_window"))

    windows: list[dict[str, Any]] = []
    start = int(train_n)
    win_id = 0
    while start + int(drift_window_size) <= n:
        end = int(start + int(drift_window_size))
        xw = np.asarray(X_all[start:end], dtype=float)
        yw = np.asarray(y_all[start:end], dtype=float)
        pred = model.predict(xw)
        m = _metrics(yw, pred)
        if holiday_idx is None:
            holiday_rate = float("nan")
        else:
            holiday_rate = float(np.mean((xw[:, holiday_idx] > 0.5).astype(float)))
        windows.append(
            {
                "window_id": int(win_id),
                "start_idx": int(start),
                "end_idx_exclusive": int(end),
                "start_date": str(dates[start])[:10],
                "end_date": str(dates[end - 1])[:10],
                "n_test": int(end - start),
                "holiday_rate": holiday_rate,
                "rmse": float(m["rmse"]),
                "mae": float(m["mae"]),
                "r2": float(m["r2"]),
            }
        )
        start += int(drift_step_size)
        win_id += 1

    if not windows:
        raise ValueError("No drift windows generated. Check drift window config.")

    rmse_arr = np.asarray([w["rmse"] for w in windows], dtype=float)
    first_rmse = float(rmse_arr[0])
    last_rmse = float(rmse_arr[-1])
    slope = 0.0
    if rmse_arr.size >= 2:
        x = np.arange(rmse_arr.size, dtype=float)
        slope = float(np.polyfit(x, rmse_arr, 1)[0])

    return {
        "training_range": {
            "n_train": int(train_n),
            "start_date": str(dates[0])[:10],
            "end_date": str(dates[train_n - 1])[:10],
        },
        "window_config": {
            "window_size": int(drift_window_size),
            "step_size": int(drift_step_size),
            "n_windows": int(len(windows)),
        },
        "windows": windows,
        "summary": {
            "first_window_rmse": first_rmse,
            "last_window_rmse": last_rmse,
            "degradation_ratio_last_vs_first": float((last_rmse - first_rmse) / max(first_rmse, 1e-12)),
            "rmse_mean": float(np.mean(rmse_arr)),
            "rmse_std": float(np.std(rmse_arr, ddof=0)),
            "rmse_min": float(np.min(rmse_arr)),
            "rmse_max": float(np.max(rmse_arr)),
            "rmse_slope_per_window": float(slope),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run robustness report (noise/missing/drift) for work CI piecewise model.")
    parser.add_argument(
        "--csv-path",
        type=str,
        default=default_work_ci_csv(),
    )
    parser.add_argument("--target-col", type=str, default="ci")
    parser.add_argument("--date-col", type=str, default="date")
    parser.add_argument("--test-fold-col", type=str, default="test_fold_10")

    parser.add_argument("--min-leaf", type=int, default=64)
    parser.add_argument("--blend-kappa", type=float, default=512.0)
    parser.add_argument("--disable-merge-rare-holiday-regimes", action="store_true")
    parser.add_argument("--disable-confidence-blend", action="store_true")
    parser.add_argument("--local-search-topk-features", type=int, default=8)
    parser.add_argument("--local-search-max-added-terms", type=int, default=12)
    parser.add_argument("--local-search-max-pair-terms", type=int, default=16)
    parser.add_argument("--local-search-max-candidates-per-iter", type=int, default=500)
    parser.add_argument("--local-search-candidate-keep-top", type=int, default=12)
    parser.add_argument("--local-search-unary-ops", type=str, default="square,sin,cos,tanh")
    parser.add_argument("--local-search-nested-unary-patterns", type=str, default="sin(square),cos(square)")

    parser.add_argument("--noise-levels", type=str, default="0.01,0.03,0.05,0.1")
    parser.add_argument("--missing-rates", type=str, default="0.05,0.1,0.2")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--random-seed", type=int, default=42)

    parser.add_argument("--drift-train-size", type=int, default=960)
    parser.add_argument("--drift-window-size", type=int, default=120)
    parser.add_argument("--drift-step-size", type=int, default=120)
    args = parser.parse_args()

    spec = ModelSpec(
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
        min_leaf=int(max(20, args.min_leaf)),
        merge_rare_holiday_regimes=not bool(args.disable_merge_rare_holiday_regimes),
        blend_with_global=not bool(args.disable_confidence_blend),
        blend_kappa=float(max(1e-6, args.blend_kappa)),
        local_search_topk_features=int(max(1, args.local_search_topk_features)),
        local_search_max_added_terms=int(max(0, args.local_search_max_added_terms)),
        local_search_max_pair_terms=int(max(0, args.local_search_max_pair_terms)),
        local_search_max_candidates_per_iter=int(max(1, args.local_search_max_candidates_per_iter)),
        local_search_candidate_keep_top=int(max(1, args.local_search_candidate_keep_top)),
        local_search_unary_ops=tuple(_parse_csv_list(args.local_search_unary_ops)),
        local_search_nested_unary_patterns=tuple(_parse_csv_list(args.local_search_nested_unary_patterns)),
    )

    noise_levels = tuple(float(v) for v in _parse_float_list(args.noise_levels))
    missing_rates = tuple(float(v) for v in _parse_float_list(args.missing_rates))
    if not noise_levels:
        raise ValueError("noise_levels is empty")
    if not missing_rates:
        raise ValueError("missing_rates is empty")

    t0 = time.perf_counter()
    xtr, ytr, xte, yte, feature_names = _load_fold_split(
        csv_path=str(args.csv_path),
        target_col=str(args.target_col),
        test_fold_col=str(args.test_fold_col),
    )
    model = _build_model(
        X_train=xtr,
        y_train=ytr,
        feature_names=feature_names,
        spec=spec,
        tag="fold_eval",
    )

    pred_clean = model.predict(xte)
    clean_metrics = _metrics(yte, pred_clean)
    clean_rmse = float(clean_metrics["rmse"])
    param_stats = _compute_param_stats(X_train=xtr, param_idx=model.param_idx)

    noise_report = _evaluate_noise_stability(
        model=model,
        X_test=xte,
        y_test=yte,
        param_stats=param_stats,
        noise_levels=noise_levels,
        repeats=int(max(1, args.repeats)),
        random_seed=int(args.random_seed),
        clean_rmse=clean_rmse,
    )
    missing_report = _evaluate_missing_tolerance(
        model=model,
        X_test=xte,
        y_test=yte,
        param_stats=param_stats,
        missing_rates=missing_rates,
        repeats=int(max(1, args.repeats)),
        random_seed=int(args.random_seed),
        clean_rmse=clean_rmse,
    )
    drift_report = _evaluate_drift_resistance(
        csv_path=str(args.csv_path),
        target_col=str(args.target_col),
        date_col=str(args.date_col),
        spec=spec,
        drift_train_size=int(args.drift_train_size),
        drift_window_size=int(args.drift_window_size),
        drift_step_size=int(args.drift_step_size),
    )
    elapsed = float(time.perf_counter() - t0)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = ROOT / "examples" / "out" / f"work_ci_robustness_report_{stamp}"
    out_root.mkdir(parents=True, exist_ok=True)
    out_path = out_root / "summary.json"

    summary = {
        "timestamp": datetime.now().isoformat(),
        "output_root": str(out_root),
        "runtime_sec": elapsed,
        "dataset": {
            "source": str(args.csv_path),
            "target_col": str(args.target_col),
            "date_col": str(args.date_col),
            "test_fold_col": str(args.test_fold_col),
            "n_train": int(xtr.shape[0]),
            "n_test": int(xte.shape[0]),
        },
        "model_config": _jsonable(asdict(spec)),
        "model_training_detail": _jsonable(model.training_detail),
        "clean_test_metrics": _jsonable(clean_metrics),
        "noise_stability": _jsonable(noise_report),
        "missing_tolerance": _jsonable(missing_report),
        "drift_resistance": _jsonable(drift_report),
    }
    out_path.write_text(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2), encoding="utf-8")

    print("WORK_CI_ROBUSTNESS_REPORT_DONE")
    print(f"output_root={out_root}")
    print(f"clean_rmse={clean_rmse:.6f}")
    print(f"noise_avg_degradation={float(noise_report['avg_degradation_ratio']):.6f}")
    print(f"missing_avg_degradation={float(missing_report['avg_degradation_ratio']):.6f}")
    print(
        "drift_last_vs_first="
        f"{float(drift_report['summary']['degradation_ratio_last_vs_first']):.6f}"
    )
    print(f"summary={out_path}")


if __name__ == "__main__":
    main()
