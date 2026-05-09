from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

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
    if isinstance(v, dict):
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
    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
    }


def _build_problem(
    *,
    n_total: int = 2200,
    train_ratio: float = 0.8,
    noise_std: float = 0.10,
    seed: int = 7,
) -> tuple[ProcessedDataset, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(seed))

    x0 = rng.uniform(-2.0, 2.0, size=n_total)
    x1 = rng.uniform(-1.8, 2.5, size=n_total)
    x2 = rng.uniform(-3.2, 3.2, size=n_total)
    x3 = rng.uniform(-2.2, 2.2, size=n_total)
    x4 = rng.normal(0.0, 1.0, size=n_total)

    y_true = (
        1.4
        + 2.2 * x0
        - 1.5 * x1
        + 3.1 * np.sin(x2)
        + 1.7 * (x0 * x1)
        + 1.0 * np.maximum(x1 - 0.25, 0.0)
        + 0.8 * np.cos(x3**2)
    )
    y = y_true + rng.normal(0.0, float(noise_std), size=n_total)

    X = np.stack([x0, x1, x2, x3, x4], axis=1)
    Y = y.reshape(-1, 1)

    idx = np.arange(n_total)
    rng.shuffle(idx)
    cut = int(round(float(train_ratio) * float(n_total)))
    cut = max(1, min(cut, n_total - 1))
    tr = idx[:cut]
    te = idx[cut:]

    X_train = np.asarray(X[tr], dtype=float)
    y_train = np.asarray(Y[tr], dtype=float)
    X_test = np.asarray(X[te], dtype=float)
    y_test = np.asarray(Y[te], dtype=float)

    ds = ProcessedDataset(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        feature_names=("x0", "x1", "x2", "x3", "x4"),
        target_names=("y",),
        metadata={
            "dataset": "framework_capability_test",
            "n_total": int(n_total),
            "n_train": int(X_train.shape[0]),
            "n_test": int(X_test.shape[0]),
            "noise_std": float(noise_std),
            "true_formula": "1.4 + 2.2*x0 -1.5*x1 +3.1*sin(x2) +1.7*x0*x1 +1.0*relu(x1-0.25) +0.8*cos(x3^2)",
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
    t1 = float(time.perf_counter() - t0)

    pred = np.asarray(artifact.predict(X_test), dtype=float)
    m = _metrics(y_test, pred)

    art_dir = out_root / tag / "artifact"
    art_dir.mkdir(parents=True, exist_ok=True)
    artifact.save(str(art_dir))

    out: dict[str, Any] = {
        "tag": str(tag),
        "trainer_key": str(trainer_key),
        "duration_sec": float(t1),
        "metrics_test": dict(m),
        "artifact_id": str(getattr(artifact, "artifact_id", "unknown")),
        "artifact_dir": str(art_dir),
        "metadata": _jsonable(getattr(artifact, "metadata", {})),
    }
    return out


def _summarize_trace(run: Mapping[str, Any]) -> dict[str, Any]:
    meta = dict(run.get("metadata", {}))
    trace = list(meta.get("search_trace", {}).get("iterations", []))
    removed = 0
    prior_seen_top = 0
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
                prior_seen_top += int(pp.get("outcomes", 0))

    return {
        "n_iterations": int(len(trace)),
        "n_removed_terms": int(removed),
        "n_steps_with_prior_info": int(with_prior),
        "sum_top_prior_outcomes": int(prior_seen_top),
    }


def main() -> None:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = ROOT / "examples" / "out" / f"framework_capability_test_{stamp}"
    out_root.mkdir(parents=True, exist_ok=True)

    ds, X_test, y_test = _build_problem()

    memory_db = out_root / "path_memory.sqlite3"
    namespace = f"cap_test_{stamp}"

    ridge = _fit(
        "ridge",
        {"artifact_id": "ridge_cap_test_v1", "l2": 1.0},
        ds,
        X_test,
        y_test,
        tag="ridge",
        out_root=out_root,
    )

    stagewise_cfg = {
        "artifact_id": "stagewise_cap_test_v1",
        "force_linear_base": "auto",
        "keep_search_trace": True,
        "auto_val_ratio": 0.2,
        "auto_min_val_samples": 64,
        "auto_random_seed": 42,
        "search_max_added_terms": 8,
        "search_topk_features": 5,
        "search_max_pair_terms": 8,
        "search_max_candidates_per_iter": 240,
        "search_candidate_keep_top": 8,
        "search_include_hinge": True,
        "search_hinge_quantiles": [0.3, 0.6],
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

    stagewise_cold = _fit(
        "symbolic_stagewise",
        dict(stagewise_cfg),
        ds,
        X_test,
        y_test,
        tag="stagewise_cold",
        out_root=out_root,
    )

    stagewise_warm = _fit(
        "symbolic_stagewise",
        dict(stagewise_cfg),
        ds,
        X_test,
        y_test,
        tag="stagewise_warm",
        out_root=out_root,
    )

    cold_trace = _summarize_trace(stagewise_cold)
    warm_trace = _summarize_trace(stagewise_warm)

    summary = {
        "timestamp": datetime.now().isoformat(),
        "output_root": str(out_root),
        "memory_db": str(memory_db),
        "memory_namespace": str(namespace),
        "dataset": _jsonable(ds.metadata),
        "runs": {
            "ridge": ridge,
            "stagewise_cold": stagewise_cold,
            "stagewise_warm": stagewise_warm,
        },
        "trace_summary": {
            "cold": cold_trace,
            "warm": warm_trace,
        },
        "capability_observation": {
            "nonlinear_gain_vs_ridge_rmse": float(
                ridge["metrics_test"]["rmse"] - stagewise_warm["metrics_test"]["rmse"]
            ),
            "warm_prior_signal_delta": int(warm_trace["sum_top_prior_outcomes"] - cold_trace["sum_top_prior_outcomes"]),
            "warm_removed_terms_delta": int(warm_trace["n_removed_terms"] - cold_trace["n_removed_terms"]),
        },
    }

    (out_root / "summary.json").write_text(
        json.dumps(_jsonable(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("FRAMEWORK CAPABILITY TEST DONE")
    print(f"output_root={out_root}")
    print(
        "ridge_rmse={:.6f} | cold_rmse={:.6f} | warm_rmse={:.6f}".format(
            float(ridge["metrics_test"]["rmse"]),
            float(stagewise_cold["metrics_test"]["rmse"]),
            float(stagewise_warm["metrics_test"]["rmse"]),
        )
    )
    print(
        "cold(trace iters={}, removed={}, prior_sum={})".format(
            int(cold_trace["n_iterations"]),
            int(cold_trace["n_removed_terms"]),
            int(cold_trace["sum_top_prior_outcomes"]),
        )
    )
    print(
        "warm(trace iters={}, removed={}, prior_sum={})".format(
            int(warm_trace["n_iterations"]),
            int(warm_trace["n_removed_terms"]),
            int(warm_trace["sum_top_prior_outcomes"]),
        )
    )
    print(f"summary={out_root / 'summary.json'}")


if __name__ == "__main__":
    main()

