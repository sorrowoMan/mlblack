from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.common.contracts import ProcessedDataset
from core.trainers.symbolic_stagewise_trainer import (
    SymbolicStagewiseSurrogateTrainer,
    SymbolicStagewiseTrainerConfig,
)
from core.trainers.xgboost_trainer import XGBoostSurrogateTrainer, XGBoostTrainerConfig
from examples.path_defaults import (
    apply_env_defaults,
    default_outputs_dir,
    default_reports_dir,
    default_work_ci_csv_no_flow_speed_occ_lag,
)


def _jsonable(v: Any) -> Any:
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, set)):
        return [_jsonable(x) for x in v]
    return str(v)


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt = np.asarray(y_true, dtype=float).reshape(-1)
    yp = np.asarray(y_pred, dtype=float).reshape(-1)
    return float(np.sqrt(np.mean((yp - yt) ** 2)))


def _summ(values: Sequence[float]) -> dict[str, float]:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=0)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _build_feature_cols(df: pd.DataFrame, *, target_col: str, date_col: str) -> list[str]:
    fold_cols = [c for c in df.columns if str(c).startswith("test_fold_")]
    drop = set(fold_cols)
    drop.add(str(target_col))
    if date_col in df.columns:
        drop.add(str(date_col))
    cols = [str(c) for c in df.columns if c not in drop]
    if not cols:
        raise ValueError("No feature columns available")
    return cols


def _split_plan_cv(df: pd.DataFrame, *, fold_start: int, fold_end: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for k in range(int(fold_start), int(fold_end) + 1):
        fold_col = f"test_fold_{k}"
        if fold_col not in df.columns:
            raise ValueError(f"Missing fold column: {fold_col}")
        test_mask = pd.to_numeric(df[fold_col], errors="coerce").fillna(0).astype(int).to_numpy() == 1
        train_mask = ~test_mask
        if int(np.sum(test_mask)) == 0:
            raise ValueError(f"No test rows for {fold_col}")
        out.append(
            {
                "scope": "cv",
                "split_id": int(k),
                "split_name": str(fold_col),
                "train_idx": np.where(train_mask)[0],
                "test_idx": np.where(test_mask)[0],
            }
        )
    return out


def _split_plan_rolling(
    df: pd.DataFrame,
    *,
    min_train_size: int,
    test_size: int,
    step_size: int,
    mode: str,
    train_window_size: int,
) -> list[dict[str, Any]]:
    n = int(df.shape[0])
    if min_train_size + test_size > n:
        raise ValueError("rolling min_train_size + test_size exceeds sample count")
    if min_train_size <= 0 or test_size <= 0 or step_size <= 0:
        raise ValueError("rolling sizes must be positive")

    key = str(mode).strip().lower()
    if key not in {"expanding", "sliding"}:
        raise ValueError("rolling mode must be expanding|sliding")
    if key == "sliding" and train_window_size <= 0:
        raise ValueError("train_window_size must be positive for sliding")

    out: list[dict[str, Any]] = []
    split_id = 0
    train_end = int(min_train_size)
    while train_end + int(test_size) <= n:
        if key == "expanding":
            train_start = 0
        else:
            train_start = max(0, int(train_end) - int(train_window_size))
        test_start = int(train_end)
        test_end = int(test_start + int(test_size))
        out.append(
            {
                "scope": "rolling",
                "split_id": int(split_id),
                "split_name": f"rolling_{split_id:02d}",
                "train_idx": np.arange(train_start, train_end, dtype=int),
                "test_idx": np.arange(test_start, test_end, dtype=int),
                "train_start": int(train_start),
                "train_end": int(train_end),
                "test_start": int(test_start),
                "test_end": int(test_end),
            }
        )
        split_id += 1
        train_end += int(step_size)
    return out


@dataclass(frozen=True)
class ModelSpec:
    key: str
    description: str


def _fit_predict(
    *,
    model_key: str,
    seed: int,
    feature_names: tuple[str, ...],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    cache_db_root: Path,
) -> dict[str, Any]:
    train_ds = ProcessedDataset(
        X_train=np.asarray(X_train, dtype=float),
        y_train=np.asarray(y_train, dtype=float),
        feature_names=feature_names,
        target_names=("ci",),
    )

    t0 = time.perf_counter()
    if model_key == "xgboost":
        trainer = XGBoostSurrogateTrainer(
            config=XGBoostTrainerConfig(
                artifact_id=f"xgboost_seed{int(seed)}",
                n_estimators=360,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                tree_method="hist",
                random_seed=int(seed),
            )
        )
    elif model_key == "stagewise_base":
        trainer = SymbolicStagewiseSurrogateTrainer(
            config=SymbolicStagewiseTrainerConfig(
                artifact_id=f"stagewise_base_seed{int(seed)}",
                force_linear_base="auto",
                keep_search_trace=False,
                auto_val_ratio=0.2,
                auto_min_val_samples=64,
                auto_random_seed=int(seed),
                search_max_added_terms=8,
                search_topk_features=min(8, int(len(feature_names))),
                search_max_pair_terms=8,
                search_max_candidates_per_iter=220,
                search_candidate_keep_top=10,
                search_enable_prune=True,
                search_prune_rmse_tolerance=1e-6,
                search_prune_max_removed_per_iter=1,
                search_path_memory_enabled=False,
                search_graph_cache_enabled=True,
                search_graph_cache_backend="memory",
                search_inner_opt_enabled=False,
            )
        )
    elif model_key == "stagewise_new":
        db_path = cache_db_root / "expression_graph_cache.sqlite3"
        trainer = SymbolicStagewiseSurrogateTrainer(
            config=SymbolicStagewiseTrainerConfig(
                artifact_id=f"stagewise_new_seed{int(seed)}",
                force_linear_base="auto",
                keep_search_trace=False,
                auto_val_ratio=0.2,
                auto_min_val_samples=64,
                auto_random_seed=int(seed),
                search_max_added_terms=8,
                search_topk_features=min(8, int(len(feature_names))),
                search_max_pair_terms=8,
                search_max_candidates_per_iter=220,
                search_candidate_keep_top=10,
                search_enable_prune=True,
                search_prune_rmse_tolerance=1e-6,
                search_prune_max_removed_per_iter=1,
                search_path_memory_enabled=False,
                search_graph_cache_enabled=True,
                search_graph_cache_backend="sqlite",
                search_graph_cache_db_path=str(db_path),
                search_graph_cache_namespace="work_ci_stagewise_vs_xgb",
                search_inner_opt_enabled=True,
                search_inner_opt_method="adam_lbfgs",
                search_inner_opt_device="auto",
                search_inner_opt_random_seed=int(seed),
                search_inner_opt_adam_steps=120,
                search_inner_opt_adam_lr=5e-3,
                search_inner_opt_adam_weight_decay=0.0,
                search_inner_opt_lbfgs_steps=60,
                search_inner_opt_lbfgs_lr=0.8,
                search_inner_opt_l2=0.0,
                search_inner_opt_accept_rmse_tol=1e-6,
            )
        )
    else:
        raise ValueError(f"Unknown model_key: {model_key}")

    artifact = trainer.fit(train_ds)
    pred = np.asarray(artifact.predict(np.asarray(X_test, dtype=float)), dtype=float).reshape(-1)
    duration = float(time.perf_counter() - t0)

    strategy = dict(artifact.metadata.get("strategy", {}))
    inner_opt = dict(strategy.get("inner_opt", {})) if strategy else {}
    terms = int(strategy.get("terms", 0)) if strategy else 0

    return {
        "prediction": pred,
        "duration_sec": float(duration),
        "artifact_type": type(artifact).__name__,
        "terms": int(terms),
        "inner_opt": inner_opt,
    }


def main() -> None:
    apply_env_defaults()

    parser = argparse.ArgumentParser(
        description="Traffic benchmark: same data + same seed for xgboost vs symbolic_stagewise(base/new)."
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        default=default_work_ci_csv_no_flow_speed_occ_lag(),
    )
    parser.add_argument("--target-col", type=str, default="ci")
    parser.add_argument("--date-col", type=str, default="date")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fold-start", type=int, default=1)
    parser.add_argument("--fold-end", type=int, default=10)
    parser.add_argument("--rolling-min-train-size", type=int, default=960)
    parser.add_argument("--rolling-test-size", type=int, default=120)
    parser.add_argument("--rolling-step-size", type=int, default=120)
    parser.add_argument("--rolling-mode", type=str, default="expanding")
    parser.add_argument("--rolling-train-window-size", type=int, default=720)
    parser.add_argument(
        "--output-root",
        type=str,
        default=str(Path(default_outputs_dir()) / "work_ci_stagewise_vs_xgb_cv_rolling"),
    )
    parser.add_argument(
        "--report-json",
        type=str,
        default=str(Path(default_reports_dir()) / "work_ci_stagewise_vs_xgb_cv_rolling.json"),
    )
    args = parser.parse_args()

    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    report_path = Path(args.report_json).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    cache_db_root = output_root / "graph_cache"
    cache_db_root.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv_path)
    if args.date_col not in df.columns:
        raise ValueError(f"Missing date column: {args.date_col}")
    df = df.copy()
    df[args.date_col] = pd.to_datetime(df[args.date_col], errors="coerce")
    if df[args.date_col].isna().any():
        raise ValueError(f"Invalid date values in column {args.date_col}")
    df = df.sort_values(args.date_col).reset_index(drop=True)

    feature_cols = _build_feature_cols(df, target_col=str(args.target_col), date_col=str(args.date_col))
    for c in feature_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df[str(args.target_col)] = pd.to_numeric(df[str(args.target_col)], errors="coerce")
    if df[feature_cols].isna().any().any() or df[str(args.target_col)].isna().any():
        raise ValueError("NaN found after numeric conversion")

    X_all = df[feature_cols].to_numpy(dtype=float)
    y_all = df[str(args.target_col)].to_numpy(dtype=float).reshape(-1)
    feature_names = tuple(str(c) for c in feature_cols)

    models = (
        ModelSpec("xgboost", "xgboost baseline"),
        ModelSpec("stagewise_base", "symbolic_stagewise without inner-opt"),
        ModelSpec("stagewise_new", "symbolic_stagewise with Adam/LBFGS + sqlite cache"),
    )

    splits = _split_plan_cv(df, fold_start=int(args.fold_start), fold_end=int(args.fold_end))
    splits.extend(
        _split_plan_rolling(
            df,
            min_train_size=int(args.rolling_min_train_size),
            test_size=int(args.rolling_test_size),
            step_size=int(args.rolling_step_size),
            mode=str(args.rolling_mode),
            train_window_size=int(args.rolling_train_window_size),
        )
    )

    rows: list[dict[str, Any]] = []
    for sp in splits:
        scope = str(sp["scope"])
        split_name = str(sp["split_name"])
        tr_idx = np.asarray(sp["train_idx"], dtype=int)
        te_idx = np.asarray(sp["test_idx"], dtype=int)
        X_train = np.asarray(X_all[tr_idx], dtype=float)
        y_train = np.asarray(y_all[tr_idx], dtype=float)
        X_test = np.asarray(X_all[te_idx], dtype=float)
        y_test = np.asarray(y_all[te_idx], dtype=float)

        for ms in models:
            print(f"[{scope}] {split_name} -> {ms.key} ...")
            res = _fit_predict(
                model_key=str(ms.key),
                seed=int(args.seed),
                feature_names=feature_names,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                cache_db_root=cache_db_root,
            )
            rmse = _rmse(y_test, np.asarray(res["prediction"], dtype=float))
            row = {
                "scope": scope,
                "split_name": split_name,
                "split_id": int(sp["split_id"]),
                "model": str(ms.key),
                "description": str(ms.description),
                "seed": int(args.seed),
                "n_train": int(X_train.shape[0]),
                "n_test": int(X_test.shape[0]),
                "rmse": float(rmse),
                "duration_sec": float(res["duration_sec"]),
                "artifact_type": str(res["artifact_type"]),
                "terms": int(res["terms"]),
                "inner_opt": dict(res.get("inner_opt", {})),
            }
            rows.append(row)
            print(
                f"[{scope}] {split_name} {ms.key} rmse={row['rmse']:.4f} "
                f"time={row['duration_sec']:.2f}s terms={row['terms']}"
            )

    summary: dict[str, Any] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "csv_path": str(Path(args.csv_path).resolve()),
        "target_col": str(args.target_col),
        "date_col": str(args.date_col),
        "seed": int(args.seed),
        "n_samples": int(len(df)),
        "n_features": int(len(feature_cols)),
        "feature_cols": list(feature_cols),
        "rows": rows,
        "aggregate": {},
    }

    agg: dict[str, Any] = {}
    for model in {str(r["model"]) for r in rows}:
        agg[model] = {}
        for scope in ("cv", "rolling"):
            sub = [r for r in rows if str(r["model"]) == model and str(r["scope"]) == scope]
            agg[model][scope] = {
                "n_splits": int(len(sub)),
                "rmse": _summ([float(r["rmse"]) for r in sub]),
                "duration_sec": _summ([float(r["duration_sec"]) for r in sub]),
                "duration_total_sec": float(np.sum(np.asarray([float(r["duration_sec"]) for r in sub], dtype=float)))
                if sub
                else 0.0,
                "terms": _summ([float(r["terms"]) for r in sub]) if model != "xgboost" else None,
            }
    summary["aggregate"] = agg

    report_path.write_text(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"DONE report={report_path}")


if __name__ == "__main__":
    main()
