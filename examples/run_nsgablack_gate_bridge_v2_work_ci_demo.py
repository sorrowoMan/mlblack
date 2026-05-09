from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

NSGABLACK_ROOT = ROOT.parent / "nsgablack"
if str(NSGABLACK_ROOT) not in sys.path:
    sys.path.append(str(NSGABLACK_ROOT))

from core.common.contracts import ProcessedDataset
from examples.path_defaults import default_work_ci_csv
from examples.work_ci_reader import WorkCiIntervalReader
from examples.run_nsgablack_gate_bridge_v2_demo import (
    GateTreeV2Problem,
    _decode_choice,
    _fit_global_baselines,
    _fit_tree_predict,
    _jsonable,
)
from nsgablack.core.evolution_solver import EvolutionSolver


def _build_work_dataset(
    *,
    csv_path: str,
    target_col: str,
    test_fold_col: str,
) -> tuple[ProcessedDataset, np.ndarray, np.ndarray]:
    reader = WorkCiIntervalReader(
        csv_path=str(csv_path),
        target_col=str(target_col),
        test_fold_col=str(test_fold_col),
    )
    bundle = reader.read()
    tr = bundle.train
    te = bundle.test
    if te is None:
        raise ValueError("Work CI reader returned no test split.")
    if not isinstance(tr, ProcessedDataset) or not isinstance(te, ProcessedDataset):
        raise TypeError("Work CI reader must return ProcessedDataset splits.")

    ds = ProcessedDataset(
        X_train=np.asarray(tr.X_train, dtype=float),
        y_train=np.asarray(tr.y_train, dtype=float),
        X_test=np.asarray(te.X_train, dtype=float),
        y_test=np.asarray(te.y_train, dtype=float),
        feature_names=tr.feature_names,
        target_names=tr.target_names,
        metadata={
            "dataset": "work_ci_v2_gate_tree",
            "csv_path": str(csv_path),
            "target_col": str(target_col),
            "test_fold_col": str(test_fold_col),
            "n_total": int(tr.X_train.shape[0] + te.X_train.shape[0]),
            "n_train": int(tr.X_train.shape[0]),
            "n_test": int(te.X_train.shape[0]),
        },
    )
    return ds, np.asarray(te.X_train, dtype=float), np.asarray(te.y_train, dtype=float).reshape(-1, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Work-CI V2: soft gate tree with nsgablack + symbolic_stagewise.")
    parser.add_argument(
        "--csv-path",
        type=str,
        default=default_work_ci_csv(),
        help="Work CI csv path.",
    )
    parser.add_argument("--target-col", type=str, default="ci", help="Target column.")
    parser.add_argument("--test-fold-col", type=str, default="test_fold_10", help="Test fold flag column.")
    parser.add_argument("--pop-size", type=int, default=8, help="Outer NSGABLACK population size.")
    parser.add_argument("--generations", type=int, default=4, help="Outer NSGABLACK max generations.")
    parser.add_argument("--rolling-folds", type=int, default=2, help="Rolling validation folds.")
    args = parser.parse_args()

    pop_size = int(max(4, args.pop_size))
    generations = int(max(1, args.generations))
    rolling_folds = int(max(1, args.rolling_folds))

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = ROOT / "examples" / "out" / f"nsgablack_gate_bridge_v2_work_ci_demo_{stamp}"
    out_root.mkdir(parents=True, exist_ok=True)

    ds, X_test, y_test = _build_work_dataset(
        csv_path=args.csv_path,
        target_col=args.target_col,
        test_fold_col=args.test_fold_col,
    )
    X_train = np.asarray(ds.X_train, dtype=float)
    y_train = np.asarray(ds.y_train, dtype=float).reshape(-1, 1)
    feature_names = tuple(str(v) for v in (ds.feature_names or tuple(f"x{i}" for i in range(X_train.shape[1]))))

    baseline = _fit_global_baselines(train_ds=ds, X_test=X_test, y_test=y_test)

    holiday_root_features = (
        "is_holiday_day_or_window",
        "is_holiday_near",
        "is_holiday_mid",
        "is_nonwork_weekend",
    )
    traffic_child_features = (
        "avg_occ",
        "avg_speed",
        "total_flow",
        "aqi",
        "wind",
        "is_bad_weather",
        "weather_dummy",
        "life_impact",
    )

    root_feature_indices = tuple(feature_names.index(nm) for nm in holiday_root_features if nm in feature_names)
    child_feature_indices = tuple(feature_names.index(nm) for nm in traffic_child_features if nm in feature_names)
    if not root_feature_indices:
        # Fallback: if time features are missing, use first small subset.
        d = int(X_train.shape[1])
        root_feature_indices = tuple(range(min(4, d)))
    if not child_feature_indices:
        d = int(X_train.shape[1])
        child_feature_indices = tuple(range(min(8, d)))

    # Keep union for backward-compatible decode helpers.
    gate_feature_indices = tuple(dict.fromkeys(list(root_feature_indices) + list(child_feature_indices)))

    outer_eval_stage_cfg = {
        "force_linear_base": "auto",
        "keep_search_trace": False,
        "auto_val_ratio": 0.2,
        "auto_min_val_samples": 64,
        "auto_random_seed": 42,
        "search_max_added_terms": 3,
        "search_topk_features": min(8, int(X_train.shape[1])),
        "search_max_pair_terms": 4,
        "search_max_candidates_per_iter": 80,
        "search_candidate_keep_top": 6,
        "search_include_hinge": True,
        "search_hinge_quantiles": [0.25, 0.5, 0.75],
        "search_unary_ops": ["square", "sin", "cos"],
        "search_nested_unary_patterns": ["sin(square)"],
        "search_enable_prune": True,
        "search_prune_rmse_tolerance": 1e-6,
        "search_prune_max_removed_per_iter": 1,
        "search_path_memory_enabled": False,
        "search_min_actual_rmse_gain": 0.0,
    }

    min_leaf = max(80, int(round(0.08 * X_train.shape[0])))
    problem = GateTreeV2Problem(
        X_fit=X_train,
        y_fit=y_train,
        feature_names=feature_names,
        gate_feature_indices=gate_feature_indices,
        eval_trainer_key="symbolic_stagewise",
        eval_trainer_params=outer_eval_stage_cfg,
        root_feature_indices=root_feature_indices,
        child_feature_indices=child_feature_indices,
        rolling_folds=rolling_folds,
        rolling_val_ratio=0.18,
        min_leaf=min_leaf,
    )
    solver = EvolutionSolver(problem, pop_size=pop_size, max_generations=generations)
    t0 = time.perf_counter()
    run = solver.run(return_dict=True)
    outer_sec = float(time.perf_counter() - t0)

    pareto_obj = np.asarray(run.get("pareto_objectives"), dtype=float)
    pareto_ind = np.asarray(run.get("pareto_solutions", {}).get("individuals", []), dtype=float)
    if pareto_obj.size == 0 or pareto_ind.size == 0:
        raise RuntimeError("nsgablack returned empty pareto set.")
    if pareto_obj.ndim == 1:
        pareto_obj = pareto_obj.reshape(-1, 1)

    score = pareto_obj[:, 0] + 0.15 * pareto_obj[:, 1]
    best_idx = int(np.argmin(score))
    best_choice = _decode_choice(
        pareto_ind[best_idx],
        gate_feature_indices=gate_feature_indices,
        root_feature_indices=root_feature_indices,
        child_feature_indices=child_feature_indices,
        feature_names=feature_names,
        X_ref=X_train,
    )

    final_local_cfg = {
        "force_linear_base": "auto",
        "keep_search_trace": False,
        "auto_val_ratio": 0.2,
        "auto_min_val_samples": 64,
        "auto_random_seed": 42,
        "search_max_added_terms": 6,
        "search_topk_features": min(8, int(X_train.shape[1])),
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
    }

    pred_test, final_detail = _fit_tree_predict(
        choice=best_choice,
        X_train=X_train,
        y_train=y_train,
        X_eval=X_test,
        trainer_key="symbolic_stagewise",
        trainer_params=final_local_cfg,
        feature_names=feature_names,
        min_leaf=min_leaf,
    )
    err = np.asarray(pred_test, dtype=float) - np.asarray(y_test, dtype=float)
    rmse = float(np.sqrt(np.mean(err**2)))
    mae = float(np.mean(np.abs(err)))
    y_flat = np.asarray(y_test, dtype=float).reshape(-1)
    e_flat = np.asarray(err, dtype=float).reshape(-1)
    ss_tot = float(np.sum((y_flat - np.mean(y_flat)) ** 2))
    r2 = float("nan") if ss_tot <= 1e-12 else float(1.0 - np.sum(e_flat**2) / ss_tot)
    v2_metrics = {"rmse": rmse, "mae": mae, "r2": r2}

    summary = {
        "timestamp": datetime.now().isoformat(),
        "output_root": str(out_root),
        "dataset": _jsonable(ds.metadata),
        "baseline": baseline,
        "nsgablack_search": {
            "duration_sec": outer_sec,
            "pop_size": int(pop_size),
            "max_generations": int(generations),
            "requested_budget": int(pop_size * generations),
            "evaluation_count": int(run.get("evaluation_count", 0)),
            "generation": int(run.get("generation", 0)),
            "rolling_folds": int(rolling_folds),
            "root_feature_pool": [str(feature_names[i]) for i in root_feature_indices],
            "child_feature_pool": [str(feature_names[i]) for i in child_feature_indices],
            "selected_choice": _jsonable(asdict(best_choice)),
            "selected_pareto_obj": _jsonable(pareto_obj[best_idx]),
            "selected_score": float(score[best_idx]),
            "cache": problem.cache_snapshot(),
            "outer_eval": {
                "trainer_key": "symbolic_stagewise",
                "trainer_params": _jsonable(outer_eval_stage_cfg),
                "tree": "soft_root + soft_right_child",
                "validation": "rolling",
                "min_leaf": int(min_leaf),
            },
        },
        "v2_gated_symbolic_tree": {
            "metrics_test": v2_metrics,
            "final_detail": _jsonable(final_detail),
        },
        "delta_vs_global_stagewise_rmse": float(
            v2_metrics["rmse"] - float(baseline["symbolic_stagewise"]["metrics_test"]["rmse"])
        ),
    }
    summary_path = out_root / "summary.json"
    summary_path.write_text(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2), encoding="utf-8")

    print("NSGABLACK_GATE_BRIDGE_V2_WORK_CI_DEMO_DONE")
    print(f"output_root={out_root}")
    print("baseline_rmse:")
    print(f"  ridge={float(baseline['ridge']['metrics_test']['rmse']):.6f}")
    print(f"  xgboost={float(baseline['xgboost']['metrics_test']['rmse']):.6f}")
    print(f"  global_stagewise={float(baseline['symbolic_stagewise']['metrics_test']['rmse']):.6f}")
    print("v2_gated_symbolic_tree_rmse={:.6f}".format(float(v2_metrics["rmse"])))
    print(
        "selected_root={}<= {:.6f} (temp={:.6f}), right={}<= {:.6f} (temp={:.6f})".format(
            str(best_choice.root_feature_name),
            float(best_choice.root_threshold),
            float(best_choice.root_temp),
            str(best_choice.right_feature_name),
            float(best_choice.right_threshold),
            float(best_choice.right_temp),
        )
    )
    print(f"summary={summary_path}")


if __name__ == "__main__":
    main()
