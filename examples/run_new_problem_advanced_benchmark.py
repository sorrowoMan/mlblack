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
from examples.run_new_problem_regime_lag_test import _build_problem


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
    m = _metrics(y_test, pred)

    art_dir = out_root / tag / "artifact"
    art_dir.mkdir(parents=True, exist_ok=True)
    artifact.save(str(art_dir))

    return {
        "tag": str(tag),
        "trainer_key": str(trainer_key),
        "duration_sec": float(duration),
        "metrics_test": dict(m),
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
    out_root = ROOT / "examples" / "out" / f"new_problem_advanced_benchmark_{stamp}"
    out_root.mkdir(parents=True, exist_ok=True)

    ds, X_test, y_test = _build_problem()

    memory_db = out_root / "path_memory.sqlite3"
    namespace = f"new_problem_adv_{stamp}"

    runs: dict[str, dict[str, Any]] = {}
    errors: dict[str, str] = {}

    candidates: list[tuple[str, str, dict[str, Any]]] = [
        (
            "ridge",
            "ridge",
            {
                "artifact_id": "adv_ridge_v1",
                "l2": 1.0,
            },
        ),
        (
            "xgboost",
            "xgboost",
            {
                "artifact_id": "adv_xgboost_v1",
                "n_estimators": 420,
                "max_depth": 6,
                "learning_rate": 0.04,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
                "tree_method": "hist",
                "random_seed": 42,
            },
        ),
        (
            "sklearn_mlp",
            "sklearn_mlp",
            {
                "artifact_id": "adv_sklearn_mlp_v1",
                "hidden_layer_sizes": [256, 128, 64],
                "max_iter": 600,
                "alpha": 1e-4,
                "learning_rate_init": 8e-4,
                "validation_fraction": 0.15,
                "n_iter_no_change": 30,
                "random_seed": 42,
                "early_stopping": True,
                "verbose": False,
            },
        ),
        (
            "mlp_torch",
            "mlp_torch",
            {
                "artifact_id": "adv_mlp_torch_v1",
                "hidden_dims": [256, 128, 64],
                "epochs": 220,
                "batch_size": 96,
                "lr": 8e-4,
                "weight_decay": 1e-4,
                "optimizer": "adamw",
                "val_ratio": 0.15,
                "early_stop_patience": 30,
                "random_seed": 42,
                "device": "auto",
                "verbose": False,
            },
        ),
        (
            "symbolic_torch",
            "symbolic_torch",
            {
                "artifact_id": "adv_symbolic_torch_v1",
                "version": "v2",
                "epochs": 260,
                "batch_size": 128,
                "lr": 1e-3,
                "weight_decay": 1e-4,
                "optimizer": "adamw",
                "val_ratio": 0.15,
                "early_stop_patience": 30,
                "random_seed": 42,
                "device": "auto",
                "v2_include_interactions": True,
                "v2_max_interactions": 24,
                "v2_topk_features": 8,
                "v2_include_hinge": True,
                "verbose": False,
            },
        ),
    ]

    for name, trainer_key, params in candidates:
        try:
            runs[name] = _fit(
                trainer_key,
                params,
                ds,
                X_test,
                y_test,
                tag=name,
                out_root=out_root,
            )
        except Exception as exc:
            errors[name] = f"{type(exc).__name__}: {exc}"

    stage_cfg = {
        "artifact_id": "adv_symbolic_stagewise_v1",
        "force_linear_base": "auto",
        "keep_search_trace": True,
        "auto_val_ratio": 0.2,
        "auto_min_val_samples": 64,
        "auto_random_seed": 42,
        "search_max_added_terms": 10,
        "search_topk_features": 8,
        "search_max_pair_terms": 12,
        "search_max_candidates_per_iter": 320,
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

    try:
        runs["symbolic_stagewise_cold"] = _fit(
            "symbolic_stagewise",
            dict(stage_cfg),
            ds,
            X_test,
            y_test,
            tag="symbolic_stagewise_cold",
            out_root=out_root,
        )
        runs["symbolic_stagewise_warm"] = _fit(
            "symbolic_stagewise",
            dict(stage_cfg),
            ds,
            X_test,
            y_test,
            tag="symbolic_stagewise_warm",
            out_root=out_root,
        )
    except Exception as exc:
        errors["symbolic_stagewise"] = f"{type(exc).__name__}: {exc}"

    cold_trace = (
        _trace_summary(runs["symbolic_stagewise_cold"]) if "symbolic_stagewise_cold" in runs else {}
    )
    warm_trace = (
        _trace_summary(runs["symbolic_stagewise_warm"]) if "symbolic_stagewise_warm" in runs else {}
    )

    leaderboard = []
    for name, run in runs.items():
        metric = dict(run.get("metrics_test", {}))
        rmse = metric.get("rmse")
        if isinstance(rmse, (int, float)):
            leaderboard.append((str(name), float(rmse)))
    leaderboard.sort(key=lambda x: x[1])

    summary = {
        "timestamp": datetime.now().isoformat(),
        "output_root": str(out_root),
        "memory_db": str(memory_db),
        "memory_namespace": str(namespace),
        "dataset": _jsonable(ds.metadata),
        "runs": runs,
        "errors": errors,
        "trace_summary": {
            "symbolic_stagewise_cold": cold_trace,
            "symbolic_stagewise_warm": warm_trace,
        },
        "leaderboard_test_rmse": [{"name": n, "rmse": v} for n, v in leaderboard],
    }

    summary_path = out_root / "summary.json"
    summary_path.write_text(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2), encoding="utf-8")

    print("NEW_PROBLEM_ADVANCED_BENCHMARK_DONE")
    print(f"output_root={out_root}")
    if leaderboard:
        print("leaderboard(test rmse):")
        for n, v in leaderboard:
            print(f"  {n:24s} rmse={v:.6f}")
    if errors:
        print("errors:")
        for k, v in errors.items():
            print(f"  {k}: {v}")
    print(f"summary={summary_path}")


if __name__ == "__main__":
    main()

