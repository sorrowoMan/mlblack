from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import TrainerAssemblySpec, build_trainer
from core.common.contracts import ProcessedDataset


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


def _build_problem(
    *,
    n_total: int = 2800,
    train_ratio: float = 0.8,
    noise_std: float = 0.08,
    shift_ratio: float = 0.65,
    seed: int = 17,
) -> tuple[ProcessedDataset, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    warmup = 24
    t = np.arange(n_total + warmup, dtype=float)

    daily_phase = 2.0 * np.pi * (t % 24.0) / 24.0
    weekly_phase = 2.0 * np.pi * (t % (24.0 * 7.0)) / (24.0 * 7.0)

    temp = 18.0 + 9.0 * np.sin(daily_phase) + 2.5 * np.sin(weekly_phase) + rng.normal(0.0, 0.8, size=t.shape[0])
    flow = 0.8 + 0.55 * np.sin(daily_phase - 0.9) + 0.25 * np.sin(weekly_phase + 0.3) + rng.normal(
        0.0, 0.12, size=t.shape[0]
    )
    humidity = 0.52 + 0.18 * np.cos(daily_phase + 0.4) + rng.normal(0.0, 0.04, size=t.shape[0])
    event = (((t.astype(int) // 24) % 7) >= 5).astype(float)

    shift_idx = int(round((n_total + warmup) * float(shift_ratio)))
    regime = (np.arange(n_total + warmup) >= shift_idx).astype(float)

    y = np.zeros((n_total + warmup,), dtype=float)
    y[:warmup] = 3.0 + rng.normal(0.0, 0.3, size=warmup)
    for i in range(warmup, n_total + warmup):
        lag1 = y[i - 1]
        lag24 = y[i - 24]
        base = (
            0.53 * lag1
            - 0.16 * lag24
            + 1.45 * np.sin(temp[i] / 7.5)
            + 1.05 * np.maximum(flow[i] - 0.55, 0.0)
            + 0.75 * event[i]
        )
        shift = regime[i] * (0.95 * np.cos(np.pi * humidity[i]) + 0.85 * (temp[i] * event[i]) / 30.0)
        y[i] = 2.4 + base + shift + rng.normal(0.0, float(noise_std))

    rows = np.arange(warmup, n_total + warmup, dtype=int)
    X = np.stack(
        [
            temp[rows],
            flow[rows],
            humidity[rows],
            event[rows],
            y[rows - 1],
            y[rows - 24],
            regime[rows],
            np.sin(2.0 * np.pi * (rows % 24) / 24.0),
        ],
        axis=1,
    )
    Y = y[rows].reshape(-1, 1)

    n = X.shape[0]
    cut = int(round(float(train_ratio) * float(n)))
    cut = max(100, min(cut, n - 100))

    X_train = np.asarray(X[:cut], dtype=float)
    y_train = np.asarray(Y[:cut], dtype=float)
    X_test = np.asarray(X[cut:], dtype=float)
    y_test = np.asarray(Y[cut:], dtype=float)

    ds = ProcessedDataset(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        feature_names=("temp", "flow", "humidity", "event", "lag1", "lag24", "regime", "hour_sin"),
        target_names=("target",),
        metadata={
            "dataset": "regime_lag_shift",
            "n_total": int(n),
            "n_train": int(X_train.shape[0]),
            "n_test": int(X_test.shape[0]),
            "noise_std": float(noise_std),
            "shift_ratio": float(shift_ratio),
            "split_mode": "chronological",
            "true_mechanism": "y_t depends on lag1, lag24 and nonlinear drivers with regime shift",
        },
    )
    return ds, X_test, y_test


def _fit(
    trainer_key: str,
    trainer_params: dict[str, Any],
    train_ds: ProcessedDataset,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    tag: str,
    out_root: Path,
) -> dict[str, Any]:
    spec = TrainerAssemblySpec(
        trainer_key=str(trainer_key),
        trainer_params=dict(trainer_params),
        pipeline_key="identity",
        pipeline_params={},
        biases=(),
    )

    t0 = time.perf_counter()
    trainer = build_trainer(spec)
    artifact = trainer.fit(train_ds)
    duration = float(time.perf_counter() - t0)

    pred = np.asarray(artifact.predict(X_test), dtype=float)
    metrics = _metrics(y_test, pred)

    art_dir = out_root / tag / "artifact"
    art_dir.mkdir(parents=True, exist_ok=True)
    artifact.save(str(art_dir))

    return {
        "tag": str(tag),
        "trainer_key": str(trainer_key),
        "duration_sec": float(duration),
        "metrics_test": dict(metrics),
        "artifact_id": str(getattr(artifact, "artifact_id", "unknown")),
        "artifact_dir": str(art_dir),
        "metadata": _jsonable(getattr(artifact, "metadata", {})),
    }


def _trace_summary(run: Mapping[str, Any]) -> dict[str, Any]:
    meta = dict(run.get("metadata", {}))
    trace = list(meta.get("search_trace", {}).get("iterations", []))
    removed = 0
    prior_outcomes = 0
    with_prior = 0
    for step in trace:
        pr = dict(step.get("pruning", {}))
        removed += int(len(list(pr.get("removed_terms", []))))
        tops = list(step.get("top_candidates", []))
        if tops:
            top0 = dict(tops[0])
            pp = dict(top0.get("path_prior", {}))
            if bool(pp.get("enabled", False)):
                with_prior += 1
                prior_outcomes += int(pp.get("outcomes", 0))
    return {
        "n_iterations": int(len(trace)),
        "n_removed_terms": int(removed),
        "n_steps_with_prior_info": int(with_prior),
        "sum_top_prior_outcomes": int(prior_outcomes),
    }


def main() -> None:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = ROOT / "examples" / "out" / f"new_problem_regime_lag_test_{stamp}"
    out_root.mkdir(parents=True, exist_ok=True)

    ds, X_test, y_test = _build_problem()

    memory_db = out_root / "path_memory.sqlite3"
    namespace = f"regime_lag_{stamp}"

    ridge = _fit(
        "ridge",
        {"artifact_id": "new_problem_ridge_v1", "l2": 1.0},
        ds,
        X_test,
        y_test,
        tag="ridge",
        out_root=out_root,
    )

    xgb = _fit(
        "xgboost",
        {
            "artifact_id": "new_problem_xgb_v1",
            "n_estimators": 320,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "tree_method": "hist",
            "random_seed": 42,
        },
        ds,
        X_test,
        y_test,
        tag="xgboost",
        out_root=out_root,
    )

    stage_cfg = {
        "artifact_id": "new_problem_stagewise_v1",
        "force_linear_base": "auto",
        "keep_search_trace": True,
        "auto_val_ratio": 0.2,
        "auto_min_val_samples": 64,
        "auto_random_seed": 42,
        "search_max_added_terms": 10,
        "search_topk_features": 8,
        "search_max_pair_terms": 12,
        "search_max_candidates_per_iter": 300,
        "search_candidate_keep_top": 10,
        "search_include_hinge": True,
        "search_hinge_quantiles": [0.25, 0.5, 0.75],
        "search_unary_ops": ["square", "sin", "cos", "tanh"],
        "search_nested_unary_patterns": ["sin(square)", "cos(square)"],
        "search_enable_prune": True,
        "search_prune_rmse_tolerance": 1e-6,
        "search_prune_max_removed_per_iter": 1,
        "search_path_memory_enabled": True,
        "search_path_memory_db_path": str(memory_db),
        "search_path_memory_namespace": str(namespace),
        "search_path_memory_prior_bonus": 0.06,
        "search_path_memory_tabu_penalty": 0.08,
        "search_path_memory_min_outcomes": 1,
        "search_min_actual_rmse_gain": 0.0,
    }

    stage_cold = _fit(
        "symbolic_stagewise",
        dict(stage_cfg),
        ds,
        X_test,
        y_test,
        tag="stagewise_cold",
        out_root=out_root,
    )
    stage_warm = _fit(
        "symbolic_stagewise",
        dict(stage_cfg),
        ds,
        X_test,
        y_test,
        tag="stagewise_warm",
        out_root=out_root,
    )

    cold_trace = _trace_summary(stage_cold)
    warm_trace = _trace_summary(stage_warm)

    summary = {
        "timestamp": datetime.now().isoformat(),
        "output_root": str(out_root),
        "memory_db": str(memory_db),
        "memory_namespace": str(namespace),
        "dataset": _jsonable(ds.metadata),
        "runs": {
            "ridge": ridge,
            "xgboost": xgb,
            "stagewise_cold": stage_cold,
            "stagewise_warm": stage_warm,
        },
        "trace_summary": {"cold": cold_trace, "warm": warm_trace},
        "observation": {
            "stagewise_vs_ridge_rmse_gain": float(ridge["metrics_test"]["rmse"] - stage_warm["metrics_test"]["rmse"]),
            "stagewise_vs_xgboost_rmse_gain": float(xgb["metrics_test"]["rmse"] - stage_warm["metrics_test"]["rmse"]),
            "warm_prior_signal_delta": int(warm_trace["sum_top_prior_outcomes"] - cold_trace["sum_top_prior_outcomes"]),
        },
    }

    summary_path = out_root / "summary.json"
    summary_path.write_text(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2), encoding="utf-8")

    print("NEW_PROBLEM_REGIME_LAG_TEST_DONE")
    print(f"output_root={out_root}")
    print(
        "rmse: ridge={:.6f} | xgboost={:.6f} | stage_cold={:.6f} | stage_warm={:.6f}".format(
            float(ridge["metrics_test"]["rmse"]),
            float(xgb["metrics_test"]["rmse"]),
            float(stage_cold["metrics_test"]["rmse"]),
            float(stage_warm["metrics_test"]["rmse"]),
        )
    )
    print(
        "trace prior_sum: cold={} | warm={}".format(
            int(cold_trace["sum_top_prior_outcomes"]),
            int(warm_trace["sum_top_prior_outcomes"]),
        )
    )
    print(f"summary={summary_path}")


if __name__ == "__main__":
    main()

