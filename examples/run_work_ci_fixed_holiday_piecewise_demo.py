from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import TrainerAssemblySpec, build_trainer
from core.common.contracts import ProcessedDataset
from examples.path_defaults import default_work_ci_csv
from examples.work_ci_reader import WorkCiIntervalReader


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


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    yt = np.asarray(y_true, dtype=float).reshape(-1)
    yp = np.asarray(y_pred, dtype=float).reshape(-1)
    err = yp - yt
    mse = float(np.mean(err**2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(err)))
    ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
    r2 = float("nan") if ss_tot <= 1e-12 else float(1.0 - np.sum(err**2) / ss_tot)
    return {"rmse": rmse, "mae": mae, "r2": r2}


@dataclass(frozen=True)
class PiecewiseSpec:
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
    local_search_max_arity: int
    local_search_max_expr_depth: int
    local_search_overfit_guard_enabled: bool
    local_search_overfit_guard_val_ratio: float
    local_search_overfit_guard_min_val_samples: int
    local_search_overfit_guard_random_seed: int
    local_search_overfit_guard_min_val_rmse_gain: float
    local_search_overfit_guard_max_gap_increase: float
    local_search_overfit_guard_patience: int
    local_search_overfit_guard_snapshot_min_improve: float
    local_search_overfit_guard_tabu_rounds: int
    local_search_overfit_guard_replace_topk: int
    local_search_overfit_guard_replace_drop_topk: int
    local_search_enable_grad_residual_projection: bool
    local_grad_projection_topk_focus: int
    local_grad_projection_partner_pool: int
    local_grad_projection_topk_partners: int
    local_grad_projection_topk_unary: int
    local_grad_projection_partner_orders: tuple[int, ...]
    local_grad_projection_enable_pair_dictionary: bool
    local_grad_projection_min_abs_corr: float
    local_grad_projection_max_generated: int


def _col_index(feature_names: tuple[str, ...], cols: tuple[str, ...]) -> tuple[int, ...]:
    idx = []
    for c in cols:
        if c in feature_names:
            idx.append(feature_names.index(c))
    return tuple(int(i) for i in idx)


def _slice_cols(X: np.ndarray, idx: tuple[int, ...]) -> np.ndarray:
    x = np.asarray(X, dtype=float)
    return np.asarray(x[:, list(idx)], dtype=float)


def _fit_artifact(
    *,
    trainer_key: str,
    trainer_params: Mapping[str, Any],
    X: np.ndarray,
    y: np.ndarray,
    feature_names: tuple[str, ...],
):
    ds = ProcessedDataset(
        X_train=np.asarray(X, dtype=float),
        y_train=np.asarray(y, dtype=float).reshape(-1, 1),
        feature_names=feature_names,
        target_names=("ci",),
        metadata={"dataset": "work_ci_fixed_holiday_piecewise"},
    )
    spec = TrainerAssemblySpec(
        trainer_key=str(trainer_key),
        trainer_params=dict(trainer_params),
        pipeline_key="identity",
        pipeline_params={},
        biases=(),
    )
    trainer = build_trainer(spec)
    return trainer.fit(ds)


def _parse_csv_list(raw: str) -> tuple[str, ...]:
    txt = str(raw).strip()
    if not txt:
        return tuple()
    return tuple(p.strip() for p in txt.split(",") if p.strip())


def _parse_csv_int_list(raw: str, *, min_value: int = 1) -> tuple[int, ...]:
    items = _parse_csv_list(raw)
    out: list[int] = []
    for s in items:
        try:
            v = int(s)
        except Exception:
            continue
        out.append(int(max(min_value, v)))
    uniq = sorted(set(out))
    return tuple(int(v) for v in uniq)


def _gate_key(mat: np.ndarray) -> tuple[tuple[int, ...], ...]:
    g = np.asarray(mat, dtype=float)
    out = []
    for r in g:
        out.append(tuple(int(v > 0.5) for v in r))
    return tuple(out)


def _hamming(a: tuple[int, ...], b: tuple[int, ...]) -> int:
    return int(sum(int(x != y) for x, y in zip(a, b)))


def _build_regime_index(keys: tuple[tuple[int, ...], ...]) -> dict[tuple[int, ...], np.ndarray]:
    reg: dict[tuple[int, ...], list[int]] = {}
    for i, k in enumerate(keys):
        reg.setdefault(k, []).append(int(i))
    out: dict[tuple[int, ...], np.ndarray] = {}
    for k, idx in reg.items():
        out[k] = np.asarray(sorted(idx), dtype=int)
    return out


def _select_training_indices_for_regime(
    *,
    target_key: tuple[int, ...],
    regime_index: dict[tuple[int, ...], np.ndarray],
    min_leaf: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    exact = np.asarray(regime_index.get(target_key, np.asarray([], dtype=int)), dtype=int)
    selected: list[np.ndarray] = []
    details: dict[str, Any] = {
        "target": list(target_key),
        "exact_count": int(exact.size),
        "used_count": 0,
        "used_from": {},
    }

    if exact.size > 0:
        selected.append(exact)
        details["used_from"][str(target_key)] = int(exact.size)

    total = int(exact.size)
    need = max(0, int(min_leaf) - total)
    if need > 0:
        # Backfill from nearest regimes by Hamming distance, then by larger pool size.
        candidates = sorted(
            ((k, v) for k, v in regime_index.items() if k != target_key and int(v.size) > 0),
            key=lambda kv: (_hamming(target_key, kv[0]), -int(kv[1].size)),
        )
        for k, idx in candidates:
            if need <= 0:
                break
            take_n = min(int(idx.size), int(need))
            if take_n <= 0:
                continue
            take = np.asarray(idx[:take_n], dtype=int)
            selected.append(take)
            details["used_from"][str(k)] = int(take_n)
            total += int(take_n)
            need = max(0, int(min_leaf) - total)

    if not selected:
        # Extremely defensive fallback: use full train set.
        all_idx = np.concatenate([v for _, v in sorted(regime_index.items(), key=lambda kv: str(kv[0]))], axis=0)
        selected = [np.asarray(all_idx, dtype=int)]
        details["used_from"] = {"ALL": int(selected[0].size)}

    out_idx = np.concatenate(selected, axis=0).astype(int, copy=False)
    out_idx = np.unique(out_idx)
    details["used_count"] = int(out_idx.size)
    return out_idx, details


def _load_work(
    *,
    csv_path: str,
    target_col: str,
    test_fold_col: str,
) -> ProcessedDataset:
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
    if not isinstance(tr, ProcessedDataset) or not isinstance(te, ProcessedDataset):
        raise TypeError("Expected ProcessedDataset from WorkCiIntervalReader")
    return ProcessedDataset(
        X_train=np.asarray(tr.X_train, dtype=float),
        y_train=np.asarray(tr.y_train, dtype=float),
        X_test=np.asarray(te.X_train, dtype=float),
        y_test=np.asarray(te.y_train, dtype=float),
        feature_names=tr.feature_names,
        target_names=tr.target_names,
        metadata={
            "dataset": "work_ci_fixed_holiday_piecewise",
            "source": str(csv_path),
            "target_col": str(target_col),
            "test_fold_col": str(test_fold_col),
            "n_train": int(tr.X_train.shape[0]),
            "n_test": int(te.X_train.shape[0]),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Fixed holiday piecewise symbolic training on work CI.")
    parser.add_argument(
        "--csv-path",
        type=str,
        default=default_work_ci_csv(),
    )
    parser.add_argument("--target-col", type=str, default="ci")
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
    parser.add_argument("--local-search-max-arity", type=int, default=3)
    parser.add_argument("--local-search-max-expr-depth", type=int, default=8)
    parser.add_argument("--local-search-overfit-guard-enabled", action="store_true")
    parser.add_argument("--local-search-overfit-guard-val-ratio", type=float, default=0.2)
    parser.add_argument("--local-search-overfit-guard-min-val-samples", type=int, default=64)
    parser.add_argument("--local-search-overfit-guard-random-seed", type=int, default=42)
    parser.add_argument("--local-search-overfit-guard-min-val-rmse-gain", type=float, default=0.0)
    parser.add_argument("--local-search-overfit-guard-max-gap-increase", type=float, default=0.05)
    parser.add_argument("--local-search-overfit-guard-patience", type=int, default=3)
    parser.add_argument("--local-search-overfit-guard-snapshot-min-improve", type=float, default=0.0)
    parser.add_argument("--local-search-overfit-guard-tabu-rounds", type=int, default=2)
    parser.add_argument("--local-search-overfit-guard-replace-topk", type=int, default=3)
    parser.add_argument("--local-search-overfit-guard-replace-drop-topk", type=int, default=3)
    parser.add_argument("--local-disable-grad-residual-projection", action="store_true")
    parser.add_argument("--local-grad-projection-topk-focus", type=int, default=3)
    parser.add_argument("--local-grad-projection-partner-pool", type=int, default=8)
    parser.add_argument("--local-grad-projection-topk-partners", type=int, default=3)
    parser.add_argument("--local-grad-projection-topk-unary", type=int, default=2)
    parser.add_argument("--local-grad-projection-partner-orders", type=str, default="1,2")
    parser.add_argument("--local-disable-grad-projection-pair-dictionary", action="store_true")
    parser.add_argument("--local-grad-projection-min-abs-corr", type=float, default=0.05)
    parser.add_argument("--local-grad-projection-max-generated", type=int, default=120)
    args = parser.parse_args()

    ds = _load_work(
        csv_path=args.csv_path,
        target_col=args.target_col,
        test_fold_col=args.test_fold_col,
    )
    X_train = np.asarray(ds.X_train, dtype=float)
    y_train = np.asarray(ds.y_train, dtype=float).reshape(-1, 1)
    X_test = np.asarray(ds.X_test, dtype=float)
    y_test = np.asarray(ds.y_test, dtype=float).reshape(-1, 1)
    feature_names = tuple(str(v) for v in (ds.feature_names or tuple(f"x{i}" for i in range(X_train.shape[1]))))

    spec = PiecewiseSpec(
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
        local_search_max_arity=int(max(1, args.local_search_max_arity)),
        local_search_max_expr_depth=int(max(1, args.local_search_max_expr_depth)),
        local_search_overfit_guard_enabled=bool(args.local_search_overfit_guard_enabled),
        local_search_overfit_guard_val_ratio=float(np.clip(args.local_search_overfit_guard_val_ratio, 0.0, 0.9)),
        local_search_overfit_guard_min_val_samples=int(max(1, args.local_search_overfit_guard_min_val_samples)),
        local_search_overfit_guard_random_seed=int(args.local_search_overfit_guard_random_seed),
        local_search_overfit_guard_min_val_rmse_gain=float(args.local_search_overfit_guard_min_val_rmse_gain),
        local_search_overfit_guard_max_gap_increase=float(args.local_search_overfit_guard_max_gap_increase),
        local_search_overfit_guard_patience=int(max(0, args.local_search_overfit_guard_patience)),
        local_search_overfit_guard_snapshot_min_improve=float(
            max(0.0, args.local_search_overfit_guard_snapshot_min_improve)
        ),
        local_search_overfit_guard_tabu_rounds=int(max(0, args.local_search_overfit_guard_tabu_rounds)),
        local_search_overfit_guard_replace_topk=int(max(0, args.local_search_overfit_guard_replace_topk)),
        local_search_overfit_guard_replace_drop_topk=int(max(0, args.local_search_overfit_guard_replace_drop_topk)),
        local_search_enable_grad_residual_projection=not bool(args.local_disable_grad_residual_projection),
        local_grad_projection_topk_focus=int(max(1, args.local_grad_projection_topk_focus)),
        local_grad_projection_partner_pool=int(max(2, args.local_grad_projection_partner_pool)),
        local_grad_projection_topk_partners=int(max(1, args.local_grad_projection_topk_partners)),
        local_grad_projection_topk_unary=int(max(1, args.local_grad_projection_topk_unary)),
        local_grad_projection_partner_orders=_parse_csv_int_list(args.local_grad_projection_partner_orders, min_value=1),
        local_grad_projection_enable_pair_dictionary=not bool(args.local_disable_grad_projection_pair_dictionary),
        local_grad_projection_min_abs_corr=float(max(0.0, args.local_grad_projection_min_abs_corr)),
        local_grad_projection_max_generated=int(max(0, args.local_grad_projection_max_generated)),
    )

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
    count_train = Counter(key_train)
    count_test = Counter(key_test)
    local_unary_ops = tuple(spec.local_search_unary_ops) if len(spec.local_search_unary_ops) > 0 else ("square", "sin", "cos", "tanh")
    local_nested_unary_patterns = tuple(spec.local_search_nested_unary_patterns)
    local_structural_control = {
        "search_max_arity": int(spec.local_search_max_arity),
        "search_max_expr_depth": int(spec.local_search_max_expr_depth),
        "search_overfit_guard_enabled": bool(spec.local_search_overfit_guard_enabled),
        "search_overfit_guard_val_ratio": float(spec.local_search_overfit_guard_val_ratio),
        "search_overfit_guard_min_val_samples": int(spec.local_search_overfit_guard_min_val_samples),
        "search_overfit_guard_random_seed": int(spec.local_search_overfit_guard_random_seed),
        "search_overfit_guard_min_val_rmse_gain": float(spec.local_search_overfit_guard_min_val_rmse_gain),
        "search_overfit_guard_max_gap_increase": float(spec.local_search_overfit_guard_max_gap_increase),
        "search_overfit_guard_patience": int(spec.local_search_overfit_guard_patience),
        "search_overfit_guard_snapshot_min_improve": float(spec.local_search_overfit_guard_snapshot_min_improve),
        "search_overfit_guard_tabu_rounds": int(spec.local_search_overfit_guard_tabu_rounds),
        "search_overfit_guard_replace_topk": int(spec.local_search_overfit_guard_replace_topk),
        "search_overfit_guard_replace_drop_topk": int(spec.local_search_overfit_guard_replace_drop_topk),
        "search_enable_grad_residual_projection": bool(spec.local_search_enable_grad_residual_projection),
        "search_grad_projection_topk_focus": int(spec.local_grad_projection_topk_focus),
        "search_grad_projection_partner_pool": int(spec.local_grad_projection_partner_pool),
        "search_grad_projection_topk_partners": int(spec.local_grad_projection_topk_partners),
        "search_grad_projection_topk_unary": int(spec.local_grad_projection_topk_unary),
        "search_grad_projection_partner_orders": tuple(int(v) for v in spec.local_grad_projection_partner_orders),
        "search_grad_projection_enable_pair_dictionary": bool(spec.local_grad_projection_enable_pair_dictionary),
        "search_grad_projection_min_abs_corr": float(spec.local_grad_projection_min_abs_corr),
        "search_grad_projection_max_generated": int(spec.local_grad_projection_max_generated),
    }

    # Global baselines (same param feature set).
    t0 = time.perf_counter()
    xgb = _fit_artifact(
        trainer_key="xgboost",
        trainer_params={
            "artifact_id": "work_ci_piecewise_xgb_v1",
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
    xgb_metrics = _metrics(y_test, np.asarray(xgb.predict(X_test_param), dtype=float).reshape(-1, 1))
    xgb_sec = float(time.perf_counter() - t0)

    t0 = time.perf_counter()
    global_stage = _fit_artifact(
        trainer_key="symbolic_stagewise",
        trainer_params={
            "artifact_id": "work_ci_piecewise_global_stagewise_v1",
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
            **local_structural_control,
        },
        X=X_train_param,
        y=y_train,
        feature_names=param_names,
    )
    global_stage_metrics = _metrics(y_test, np.asarray(global_stage.predict(X_test_param), dtype=float).reshape(-1, 1))
    global_stage_sec = float(time.perf_counter() - t0)

    # Fixed piecewise local symbolic models (with rare-regime merge + confidence blending).
    t0 = time.perf_counter()
    local_models: dict[tuple[int, ...], Any] = {}
    local_effective_samples: dict[tuple[int, ...], int] = {}
    regime_index = _build_regime_index(key_train)
    regime_keys_all = tuple(sorted(set(key_train) | set(key_test)))
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
                            "artifact_id": "work_ci_piecewise_local_merged_1100_1010",
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
                            "search_nested_unary_patterns": list(local_nested_unary_patterns),
                            "search_enable_prune": True,
                            "search_prune_rmse_tolerance": 1e-6,
                            "search_prune_max_removed_per_iter": 1,
                            "search_path_memory_enabled": False,
                            "search_min_actual_rmse_gain": 0.0,
                            **local_structural_control,
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
                "artifact_id": f"work_ci_piecewise_local_{'_'.join(map(str, k))}",
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
                "search_nested_unary_patterns": list(local_nested_unary_patterns),
                "search_enable_prune": True,
                "search_prune_rmse_tolerance": 1e-6,
                "search_prune_max_removed_per_iter": 1,
                "search_path_memory_enabled": False,
                "search_min_actual_rmse_gain": 0.0,
                **local_structural_control,
            },
            X=X_train_param[idx_use],
            y=y_train[idx_use],
            feature_names=param_names,
        )
        local_models[k] = art
        local_effective_samples[k] = int(use_detail.get("used_count", int(idx_use.size)))
        regime_training_detail[str(k)] = use_detail

    pred_global = np.asarray(global_stage.predict(X_test_param), dtype=float).reshape(-1, 1)
    pred_piece = np.zeros_like(y_test, dtype=float)
    pred_blend = np.zeros_like(y_test, dtype=float)
    blend_weight = np.zeros((y_test.shape[0],), dtype=float)
    fallback_count = 0
    for i, k in enumerate(key_test):
        x_row = X_test_param[i : i + 1, :]
        pg = float(pred_global[i, 0])
        art = local_models.get(k)
        if art is None:
            fallback_count += 1
            pl = pg
            alpha = 0.0
        else:
            pl = float(np.asarray(art.predict(x_row), dtype=float).reshape(-1)[0])
            n_eff = float(local_effective_samples.get(k, 0))
            if spec.blend_with_global:
                alpha = float(n_eff / (n_eff + float(spec.blend_kappa)))
            else:
                alpha = 1.0
        pred_piece[i, 0] = pl
        pred_blend[i, 0] = float(alpha * pl + (1.0 - alpha) * pg)
        blend_weight[i] = float(alpha)
    piece_metrics = _metrics(y_test, pred_piece)
    piece_blend_metrics = _metrics(y_test, pred_blend)
    piece_sec = float(time.perf_counter() - t0)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = ROOT / "examples" / "out" / f"work_ci_fixed_holiday_piecewise_demo_{stamp}"
    out_root.mkdir(parents=True, exist_ok=True)

    summary = {
        "timestamp": datetime.now().isoformat(),
        "output_root": str(out_root),
        "dataset": _jsonable(ds.metadata),
        "piecewise_spec": _jsonable(asdict(spec)),
        "rare_merge_group": [list(k) for k in rare_merge_group],
        "rare_merge_group_used": bool(merged_group_used),
        "gate_feature_names_used": list(gate_names),
        "param_feature_names_used": list(param_names),
        "train_regime_counts": {str(k): int(v) for k, v in count_train.items()},
        "test_regime_counts": {str(k): int(v) for k, v in count_test.items()},
        "local_models_trained": [str(k) for k in local_models.keys()],
        "local_models_skipped": {},
        "local_effective_samples": {str(k): int(v) for k, v in local_effective_samples.items()},
        "regime_training_detail": regime_training_detail,
        "prediction_fallback_count": int(fallback_count),
        "blend": {
            "enabled": bool(spec.blend_with_global),
            "kappa": float(spec.blend_kappa),
            "mean_alpha_test": float(np.mean(blend_weight)) if int(blend_weight.size) > 0 else 0.0,
            "min_alpha_test": float(np.min(blend_weight)) if int(blend_weight.size) > 0 else 0.0,
            "max_alpha_test": float(np.max(blend_weight)) if int(blend_weight.size) > 0 else 0.0,
        },
        "metrics": {
            "xgboost_global": {"metrics_test": xgb_metrics, "duration_sec": xgb_sec},
            "symbolic_stagewise_global": {"metrics_test": global_stage_metrics, "duration_sec": global_stage_sec},
            "symbolic_stagewise_fixed_piecewise": {"metrics_test": piece_metrics, "duration_sec": piece_sec},
            "symbolic_stagewise_fixed_piecewise_blended": {"metrics_test": piece_blend_metrics, "duration_sec": piece_sec},
        },
        "delta_piecewise_vs_global_stagewise_rmse": float(piece_metrics["rmse"] - global_stage_metrics["rmse"]),
        "delta_piecewise_blended_vs_global_stagewise_rmse": float(
            piece_blend_metrics["rmse"] - global_stage_metrics["rmse"]
        ),
        "delta_piecewise_vs_xgboost_rmse": float(piece_metrics["rmse"] - xgb_metrics["rmse"]),
        "delta_piecewise_blended_vs_xgboost_rmse": float(piece_blend_metrics["rmse"] - xgb_metrics["rmse"]),
    }

    summary_path = out_root / "summary.json"
    summary_path.write_text(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2), encoding="utf-8")

    print("WORK_CI_FIXED_HOLIDAY_PIECEWISE_DEMO_DONE")
    print(f"output_root={out_root}")
    print("test rmse:")
    print(f"  xgboost_global={float(xgb_metrics['rmse']):.6f}")
    print(f"  symbolic_stagewise_global={float(global_stage_metrics['rmse']):.6f}")
    print(f"  symbolic_stagewise_fixed_piecewise={float(piece_metrics['rmse']):.6f}")
    print(f"  symbolic_stagewise_fixed_piecewise_blended={float(piece_blend_metrics['rmse']):.6f}")
    print(f"summary={summary_path}")


if __name__ == "__main__":
    main()
