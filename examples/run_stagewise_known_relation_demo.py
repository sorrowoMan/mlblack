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


def _build_known_relation_dataset(
    *,
    n_total: int = 3000,
    train_ratio: float = 0.8,
    noise_std: float = 0.15,
    seed: int = 42,
) -> tuple[ProcessedDataset, np.ndarray, np.ndarray, dict[str, Any]]:
    rng = np.random.default_rng(int(seed))

    x0 = rng.uniform(-2.0, 2.0, size=n_total)
    x1 = rng.uniform(-1.8, 2.5, size=n_total)
    x2 = rng.uniform(-3.2, 3.2, size=n_total)
    x3 = rng.uniform(-2.2, 2.2, size=n_total)
    x4 = rng.normal(0.0, 1.0, size=n_total)  # weak/noisy feature

    # Ground-truth relation: linear floor + nonlinear remainder
    y_true = (
        1.25
        + 2.10 * x0
        - 1.35 * x1
        + 3.40 * np.sin(x2)
        + 1.65 * (x0 * x1)
        + 1.10 * np.maximum(x1 - 0.30, 0.0)
        + 0.85 * np.cos(x3**2)
    )

    noise = rng.normal(0.0, float(noise_std), size=n_total)
    y = y_true + noise

    X = np.stack([x0, x1, x2, x3, x4], axis=1)
    Y = y.reshape(-1, 1)

    idx = np.arange(n_total)
    rng.shuffle(idx)
    cut = int(round(float(train_ratio) * float(n_total)))
    cut = max(1, min(cut, n_total - 1))

    tr_idx = idx[:cut]
    te_idx = idx[cut:]

    X_train = np.asarray(X[tr_idx], dtype=float)
    y_train = np.asarray(Y[tr_idx], dtype=float)
    X_test = np.asarray(X[te_idx], dtype=float)
    y_test = np.asarray(Y[te_idx], dtype=float)

    ds = ProcessedDataset(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        feature_names=("x0", "x1", "x2", "x3", "x4"),
        target_names=("y",),
        metadata={
            "dataset": "known_relation_demo",
            "n_total": int(n_total),
            "n_train": int(X_train.shape[0]),
            "n_test": int(X_test.shape[0]),
            "noise_std": float(noise_std),
        },
    )

    formula = {
        "expression": "y=1.25+2.10*x0-1.35*x1+3.40*sin(x2)+1.65*(x0*x1)+1.10*relu(x1-0.30)+0.85*cos(x3^2)+noise",
        "components": {
            "linear_floor": ["1.25", "2.10*x0", "-1.35*x1"],
            "nonlinear_remainder": [
                "3.40*sin(x2)",
                "1.65*(x0*x1)",
                "1.10*relu(x1-0.30)",
                "0.85*cos(x3^2)",
            ],
        },
    }

    return ds, X_test, y_test, formula


def _fit_one(
    trainer_key: str,
    trainer_params: dict[str, Any],
    train_ds: ProcessedDataset,
    X_test: np.ndarray,
    y_test: np.ndarray,
    out_dir: Path,
    *,
    run_tag: str | None = None,
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
    elapsed = float(time.perf_counter() - t0)

    pred_test = np.asarray(artifact.predict(X_test), dtype=float)
    test_metrics = _metrics(y_test, pred_test)

    tag = str(run_tag or trainer_key)
    artifact_dir = out_dir / tag / "artifact"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact.save(str(artifact_dir))

    out: dict[str, Any] = {
        "status": "ok",
        "trainer_key": str(trainer_key),
        "duration_sec": elapsed,
        "test_metrics": test_metrics,
        "artifact_dir": str(artifact_dir),
        "artifact_id": str(getattr(artifact, "artifact_id", "unknown")),
        "model_meta": _jsonable(getattr(artifact, "metadata", {}).get("model", {})),
    }

    if hasattr(artifact, "expression"):
        try:
            out["expression"] = str(artifact.expression(target_index=0, precision=8, use_feature_names=True))
        except Exception as exc:
            out["expression_error"] = f"{type(exc).__name__}: {exc}"

    return out


def main() -> None:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = ROOT / "examples" / "out" / f"known_relation_stagewise_{stamp}"
    out_root.mkdir(parents=True, exist_ok=True)

    train_ds, X_test, y_test, true_formula = _build_known_relation_dataset(
        n_total=3000,
        train_ratio=0.8,
        noise_std=0.15,
        seed=42,
    )

    runs: dict[str, Any] = {}

    stagewise_params = {
        "artifact_id": "stagewise_known_relation_v1",
        "force_linear_base": True,
        "keep_search_trace": True,
        "search_max_added_terms": 10,
        "search_topk_features": 5,
        "search_max_pair_terms": 8,
        "search_include_hinge": True,
        "search_hinge_quantiles": [0.3, 0.6],
        "search_unary_ops": ["square", "sin", "cos", "tanh"],
        "search_nested_unary_patterns": ["sin(square)", "cos(square)"],
        "search_ridge_l2": 0.0001,
    }
    runs["symbolic_stagewise_on"] = _fit_one(
        "symbolic_stagewise",
        stagewise_params,
        train_ds,
        X_test,
        y_test,
        out_root,
        run_tag="symbolic_stagewise_on",
    )

    runs["symbolic_stagewise_auto"] = _fit_one(
        "symbolic_stagewise",
        {
            "artifact_id": "stagewise_known_relation_auto_v1",
            "force_linear_base": "auto",
            "keep_search_trace": True,
            "auto_val_ratio": 0.2,
            "auto_min_val_samples": 64,
            "auto_random_seed": 42,
            "auto_term_penalty": 0.001,
            "auto_depth_penalty": 0.002,
            "auto_grad_penalty": 0.05,
            "search_max_added_terms": 10,
            "search_topk_features": 5,
            "search_max_pair_terms": 8,
            "search_include_hinge": True,
            "search_hinge_quantiles": [0.3, 0.6],
            "search_unary_ops": ["square", "sin", "cos", "tanh"],
            "search_nested_unary_patterns": ["sin(square)", "cos(square)"],
            "search_ridge_l2": 0.0001,
        },
        train_ds,
        X_test,
        y_test,
        out_root,
        run_tag="symbolic_stagewise_auto",
    )

    runs["ridge"] = _fit_one(
        "ridge",
        {"l2": 1.0, "artifact_id": "ridge_known_relation_v1"},
        train_ds,
        X_test,
        y_test,
        out_root,
        run_tag="ridge",
    )

    try:
        runs["xgboost"] = _fit_one(
            "xgboost",
            {
                "artifact_id": "xgboost_known_relation_v1",
                "n_estimators": 420,
                "max_depth": 6,
                "learning_rate": 0.05,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
                "tree_method": "hist",
                "random_seed": 42,
            },
            train_ds,
            X_test,
            y_test,
            out_root,
            run_tag="xgboost",
        )
    except Exception as exc:
        runs["xgboost"] = {
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
        }

    summary = {
        "timestamp": datetime.now().isoformat(),
        "output_root": str(out_root),
        "truth": true_formula,
        "dataset": _jsonable(train_ds.metadata),
        "runs": _jsonable(runs),
    }

    (out_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("KNOWN RELATION DEMO DONE")
    print(f"output_root={out_root}")
    for key, row in runs.items():
        if row.get("status") == "ok":
            m = row["test_metrics"]
            print(
                f"{key:18s} rmse={float(m['rmse']):.6f} mae={float(m['mae']):.6f} r2={float(m['r2']):.6f} t={float(row['duration_sec']):.3f}s"
            )
        else:
            print(f"{key:18s} failed: {row.get('error')}")
    print(f"summary={out_root / 'summary.json'}")


if __name__ == "__main__":
    main()




