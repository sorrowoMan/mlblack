from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.path_defaults import apply_env_defaults, default_outputs_dir, default_reports_dir
from project.scaffold import ScaffoldSpec, load_scaffold_spec


BASE_CENTER = {
    "version": "v2",
    "lower_quantile": 0.0989,
    "upper_quantile": 0.9042,
    "v2_continuous_ops": ["identity", "sin", "cos"],
    "v2_binary_ops": ["identity"],
    "v2_include_interactions": True,
    "v2_max_interactions": 17,
    "v2_topk_features": 7,
    "v2_include_hinge": True,
    "v2_hinge_quantiles": [0.25, 0.5, 0.75],
    "order_penalty": 4.969807170571134,
    "width_penalty": 0.00010368320526873127,
    "epochs": 180,
    "batch_size": 128,
    "lr": 0.0038605201544101336,
    "weight_decay": 0.0001,
    "l1_readout": 0.0001459323297322245,
    "l1_params": 0.0,
    "device": "cpu",
    "conformal_calibration": True,
    "conformal_level": 0.7768,
    "stagewise_warmup_enabled": False,
    "gate_piecewise_enabled": False,
}


def _clamp(v: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, float(v))))


def _log_jitter(v: float, *, scale: float, rng: random.Random) -> float:
    lv = math.log10(max(1e-12, float(v)))
    return float(10 ** (lv + rng.uniform(-scale, scale)))


def _sample_params(center: dict[str, Any], *, trial_seed: int) -> dict[str, Any]:
    rng = random.Random(int(trial_seed))
    p = dict(center)

    lower = _clamp(float(center["lower_quantile"]) + rng.uniform(-0.045, 0.045), 0.06, 0.24)
    gap = _clamp(
        (float(center["upper_quantile"]) - float(center["lower_quantile"])) + rng.uniform(-0.10, 0.10),
        0.55,
        0.84,
    )
    upper = _clamp(lower + gap, lower + 0.55, 0.98)
    p["lower_quantile"] = round(lower, 4)
    p["upper_quantile"] = round(upper, 4)

    p["lr"] = _log_jitter(float(center["lr"]), scale=0.30, rng=rng)
    p["order_penalty"] = _log_jitter(float(center["order_penalty"]), scale=0.35, rng=rng)
    p["width_penalty"] = _log_jitter(float(center["width_penalty"]), scale=0.50, rng=rng)
    p["l1_readout"] = _log_jitter(float(center["l1_readout"]), scale=0.50, rng=rng)
    p["conformal_level"] = round(_clamp(float(center["conformal_level"]) + rng.uniform(-0.08, 0.08), 0.70, 0.93), 4)

    p["v2_max_interactions"] = int(_clamp(int(center["v2_max_interactions"]) + rng.randint(-6, 6), 8, 28))
    p["v2_topk_features"] = int(_clamp(int(center["v2_topk_features"]) + rng.randint(-3, 3), 4, 12))
    p["v2_include_hinge"] = bool(center["v2_include_hinge"] if rng.random() < 0.75 else (not center["v2_include_hinge"]))
    p["epochs"] = int(rng.choice([120, 140, 160, 180, 220]))
    p["random_seed"] = int(trial_seed)
    return p


def _interval_score(y: np.ndarray, l: np.ndarray, u: np.ndarray, alpha: float) -> np.ndarray:
    width = u - l
    under = (2.0 / float(alpha)) * (l - y) * (y < l)
    over = (2.0 / float(alpha)) * (y - u) * (y > u)
    return width + under + over


def _eval_metrics(y_true: np.ndarray, lo: np.ndarray, hi: np.ndarray, *, alpha: float, y_range: float) -> dict[str, float]:
    y = np.asarray(y_true, dtype=float).reshape(-1)
    l = np.asarray(lo, dtype=float).reshape(-1)
    u = np.asarray(hi, dtype=float).reshape(-1)
    mid = 0.5 * (l + u)
    rmse = float(np.sqrt(np.mean((mid - y) ** 2)))
    picp = float(np.mean((y >= l) & (y <= u)))
    width = np.maximum(u - l, 0.0)
    mean_width = float(np.mean(width))
    return {
        "rmse": rmse,
        "picp": picp,
        "mean_width": mean_width,
        "pinaw": float(np.mean(width / max(1e-8, float(y_range)))),
        "interval_score": float(np.mean(_interval_score(y, l, u, alpha))),
    }


def _objective(m: Mapping[str, float], *, target_picp: float = 0.80) -> float:
    rmse = float(m["rmse"])
    cov_pen = max(0.0, float(target_picp) - float(m["picp"]))
    width_pen = max(0.0, float(m["mean_width"]) - 22.0)
    return float(rmse + 30.0 * cov_pen * cov_pen + 0.012 * width_pen)


def _run_one(
    *,
    config_path: str,
    feature_recipe: str,
    params: dict[str, Any],
    output_dir: str,
    run_name: str,
) -> dict[str, Any]:
    # Reduce per-process thread contention for parallel search.
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

    from project.scaffold import ScaffoldSpec, load_scaffold_spec, run_project_scaffold

    spec_base = load_scaffold_spec(config_path)
    data_spec = replace(spec_base.data, feature_recipe=str(feature_recipe))
    train_spec = replace(
        spec_base.train,
        trainer_params=dict(params),
        output_dir=str(output_dir),
        run_name=str(run_name),
    )
    spec = ScaffoldSpec(data=data_spec, train=train_spec)

    t0 = time.perf_counter()
    result = run_project_scaffold(spec)
    duration = float(time.perf_counter() - t0)

    X_test = np.asarray(result.processed.X_test, dtype=float)
    y_test = np.asarray(result.processed.y_test, dtype=float).reshape(-1)
    y_train = np.asarray(result.processed.y_train, dtype=float).reshape(-1)
    y_range = float(np.max(y_train) - np.min(y_train))

    lo, hi = result.artifact.predict_interval(X_test)
    lo = np.asarray(lo, dtype=float).reshape(-1)
    hi = np.asarray(hi, dtype=float).reshape(-1)
    m_common = _eval_metrics(y_test, lo, hi, alpha=0.3, y_range=y_range)

    return {
        "status": "ok",
        "duration_sec": duration,
        "artifact_type": type(result.artifact).__name__,
        "metrics_common": m_common,
        "objective": _objective(m_common),
        "trainer_metrics": {k: dict(v) for k, v in result.metrics.items()},
    }


def main() -> None:
    apply_env_defaults()

    parser = argparse.ArgumentParser(description="Parallel high-budget local search for work_ci symbolic interval.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(ROOT / "examples" / "configs" / "work_ci_symbolic_torch_interval.json"),
    )
    parser.add_argument("--feature-recipe", type=str, default="raw_all_numeric")
    parser.add_argument("--trials", type=int, default=120)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260415)
    parser.add_argument(
        "--out-root",
        type=str,
        default=str(Path(default_outputs_dir()) / "work_ci_symbolic_interval_parallel_max"),
    )
    parser.add_argument(
        "--report-json",
        type=str,
        default=str(Path(default_reports_dir()) / "tune_symbolic_interval_parallel_max_v1.json"),
    )
    args = parser.parse_args()

    rng = random.Random(int(args.seed))
    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    report_path = Path(args.report_json).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)

    tasks: list[dict[str, Any]] = []
    for i in range(int(args.trials)):
        trial_id = f"eval_{i:04d}"
        params = _sample_params(BASE_CENTER, trial_seed=int(args.seed) + i * 31 + 7)
        task = {
            "trial_id": trial_id,
            "params": params,
            "output_dir": str(out_root / "trials" / trial_id),
            "run_name": f"work_ci_parallel_max_{trial_id}",
        }
        tasks.append(task)

    rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None

    print(
        f"PARALLEL SEARCH START trials={len(tasks)} workers={int(args.workers)} "
        f"feature_recipe={args.feature_recipe}"
    )

    with ProcessPoolExecutor(max_workers=int(args.workers)) as ex:
        fut2task = {
            ex.submit(
                _run_one,
                config_path=str(Path(args.config).resolve()),
                feature_recipe=str(args.feature_recipe),
                params=dict(task["params"]),
                output_dir=str(task["output_dir"]),
                run_name=str(task["run_name"]),
            ): task
            for task in tasks
        }

        done = 0
        for fut in as_completed(fut2task):
            task = fut2task[fut]
            row = {
                "trial_id": task["trial_id"],
                "params": task["params"],
                "output_dir": task["output_dir"],
            }
            try:
                res = fut.result()
                row.update(res)
            except Exception as exc:
                row.update(
                    {
                        "status": "failed",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )

            rows.append(row)
            done += 1

            if row.get("status") == "ok":
                if best is None or float(row.get("objective", float("inf"))) < float(best.get("objective", float("inf"))):
                    best = row
                print(
                    f"[{done}/{len(tasks)}] {row['trial_id']} ok rmse={row['metrics_common']['rmse']:.4f} "
                    f"picp={row['metrics_common']['picp']:.4f} obj={row['objective']:.4f} "
                    f"type={row['artifact_type']} t={row['duration_sec']:.1f}s"
                )
            else:
                print(f"[{done}/{len(tasks)}] {row['trial_id']} failed {row.get('error', '')}")

            report_path.write_text(
                json.dumps(
                    {
                        "seed": int(args.seed),
                        "trials": int(args.trials),
                        "workers": int(args.workers),
                        "feature_recipe": str(args.feature_recipe),
                        "completed": int(done),
                        "best": best,
                        "rows": rows,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

    print("PARALLEL SEARCH DONE")
    if best is not None:
        print(
            f"BEST trial={best['trial_id']} rmse={best['metrics_common']['rmse']:.6f} "
            f"picp={best['metrics_common']['picp']:.6f} width={best['metrics_common']['mean_width']:.6f} "
            f"obj={best['objective']:.6f}"
        )
    print(f"report={report_path}")


if __name__ == "__main__":
    main()
