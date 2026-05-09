from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

NSGABLACK_ROOT = ROOT.parent / "nsgablack"
if str(NSGABLACK_ROOT) not in sys.path:
    sys.path.append(str(NSGABLACK_ROOT))

from core.common.contracts import ProcessedDataset
from core.symbolic.symbolic_structure_search import evaluate_genome_with_ridge
from core.trainers.xgboost_trainer import XGBoostSurrogateTrainer, XGBoostTrainerConfig
from examples.path_defaults import default_work_ci_csv
from examples.work_ci_reader import WorkCiIntervalReader
from nsgablack.core.composable_solver import ComposableSolver
from nsgablack.representation import RepresentationPipeline
from nsgablack.representation.continuous import ClipRepair, ContextGaussianMutation, UniformInitializer

from examples.run_nsgablack_symbolic_subset_bridge_work_ci import (
    SymbolicSubsetSelectionProblem,
    _bounds_arrays,
    _build_candidate_pool,
    _build_outer_adapter,
    _jsonable,
    _rmse,
)


def _mask_branches(X: np.ndarray, feature_names: tuple[str, ...]) -> dict[str, np.ndarray]:
    f2i = {str(v): i for i, v in enumerate(feature_names)}
    h = int(f2i.get("is_holiday_day_or_window", -1))
    w = int(f2i.get("is_nonwork_weekend", -1))
    if h < 0 or w < 0:
        n = int(np.asarray(X).shape[0])
        return {
            "holiday": np.zeros((n,), dtype=bool),
            "weekend": np.zeros((n,), dtype=bool),
            "regular": np.ones((n,), dtype=bool),
        }
    x = np.asarray(X, dtype=float)
    is_holiday = x[:, h] > 0.5
    is_weekend = x[:, w] > 0.5
    holiday = np.asarray(is_holiday, dtype=bool)
    weekend = np.asarray((~holiday) & is_weekend, dtype=bool)
    regular = np.asarray((~holiday) & (~weekend), dtype=bool)
    return {"holiday": holiday, "weekend": weekend, "regular": regular}


def _train_branch(
    *,
    branch_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: tuple[str, ...],
    pop_size: int,
    generations: int,
    outer_strategy: str,
    portfolio_phases: str,
    portfolio_phase_weights: str,
    moead_neighborhood_size: int,
    moead_delta: float,
    moead_nr: int,
    vns_k_max: int,
    vns_batch_size: int,
    max_terms: int,
    ridge_l2: float,
    rolling_folds: int,
    rolling_val_ratio: float,
    seed: int,
) -> dict[str, Any]:
    candidates = _build_candidate_pool(X_train, y_train, feature_names=feature_names, topk_for_pairs=6)
    problem = SymbolicSubsetSelectionProblem(
        X_fit=X_train,
        y_fit=y_train,
        candidates=candidates,
        max_terms=int(max_terms),
        ridge_l2=float(max(0.0, ridge_l2)),
        rolling_folds=int(max(1, rolling_folds)),
        rolling_val_ratio=float(np.clip(rolling_val_ratio, 0.05, 0.45)),
        min_train=max(64, int(round(0.4 * X_train.shape[0]))),
        regime_branch_mode=False,
        regime_gate_idx=None,
        regime_min_branch_train=64,
        regime_branch_parallel_workers=1,
        inner_opt_enabled=False,
        inner_opt_adam_steps=0,
        inner_opt_adam_lr=1e-2,
        inner_opt_lbfgs_steps=0,
        inner_opt_lbfgs_lr=0.8,
        inner_opt_accept_rmse_tol=0.0,
        inner_opt_alt_freeze_readout=True,
        inner_opt_grad_clip_norm=1.0,
        inner_opt_residual_clip_q=0.98,
    )
    outer_adapter, outer_meta = _build_outer_adapter(
        strategy=str(outer_strategy),
        pop_size=int(max(4, pop_size)),
        generations=int(max(1, generations)),
        portfolio_phases_csv=str(portfolio_phases),
        portfolio_weights_csv=str(portfolio_phase_weights),
        moead_neighborhood_size=int(max(2, moead_neighborhood_size)),
        moead_delta=float(moead_delta),
        moead_nr=int(max(1, moead_nr)),
        vns_k_max=int(max(1, vns_k_max)),
        vns_batch_size=int(max(4, vns_batch_size)),
    )
    low, high = _bounds_arrays(problem)
    rep = RepresentationPipeline(
        initializer=UniformInitializer(low=low, high=high),
        mutator=ContextGaussianMutation(base_sigma=0.18, sigma_key="mutation_sigma", low=low, high=high),
        repair=ClipRepair(low=low, high=high),
    )
    solver = ComposableSolver(problem=problem, adapter=outer_adapter, representation_pipeline=rep)
    solver.max_steps = int(max(1, outer_meta.get("max_generations", generations)))
    solver.set_random_seed(int(seed))
    t0 = time.perf_counter()
    run = solver.run()
    sec = float(time.perf_counter() - t0)
    top_cache = problem.cache_top(topn=30)
    if not top_cache:
        return {"branch": branch_name, "ok": False, "reason": "empty_cache", "duration_sec": sec}
    best = dict(top_cache[0])
    subset_idx = [int(v) for v in best.get("subset_idx", [])]
    if not subset_idx:
        return {"branch": branch_name, "ok": False, "reason": "empty_subset", "duration_sec": sec}
    genome = [{"name": candidates[i].name, "expr": dict(candidates[i].expr)} for i in subset_idx]
    tuned_l2 = float(best.get("tuned_l2", max(0.0, ridge_l2)))
    return {
        "branch": branch_name,
        "ok": True,
        "duration_sec": sec,
        "run_result": _jsonable(run),
        "n_cache": int(len(problem._cache)),
        "subset_size": int(len(subset_idx)),
        "subset_idx": subset_idx,
        "subset_names": [candidates[i].name for i in subset_idx],
        "genome": genome,
        "tuned_l2": tuned_l2,
        "top_cache": top_cache[:10],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Piecewise-gated symbolic bridge on Work-CI.")
    parser.add_argument("--csv-path", type=str, default=default_work_ci_csv())
    parser.add_argument("--target-col", type=str, default="ci")
    parser.add_argument("--test-fold-col", type=str, default="test_fold_10")
    parser.add_argument("--pop-size", type=int, default=24)
    parser.add_argument("--generations", type=int, default=15)
    parser.add_argument("--rolling-folds", type=int, default=3)
    parser.add_argument("--rolling-val-ratio", type=float, default=0.18)
    parser.add_argument("--max-terms", type=int, default=12)
    parser.add_argument("--ridge-l2", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outer-strategy", type=str, default="portfolio", choices=["nsga2", "moead", "vns", "portfolio"])
    parser.add_argument("--portfolio-phases", type=str, default="nsga2,moead,vns")
    parser.add_argument("--portfolio-phase-weights", type=str, default="2,1,1")
    parser.add_argument("--moead-neighborhood-size", type=int, default=12)
    parser.add_argument("--moead-delta", type=float, default=0.9)
    parser.add_argument("--moead-nr", type=int, default=2)
    parser.add_argument("--vns-k-max", type=int, default=5)
    parser.add_argument("--vns-batch-size", type=int, default=32)
    parser.add_argument("--min-branch-train", type=int, default=200)
    args = parser.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = ROOT / "examples" / "out" / f"nsgablack_piecewise_symbolic_bridge_work_ci_{stamp}"
    out_root.mkdir(parents=True, exist_ok=True)

    reader = WorkCiIntervalReader(csv_path=str(args.csv_path), target_col=str(args.target_col), test_fold_col=str(args.test_fold_col))
    bundle = reader.read()
    tr = bundle.train
    te = bundle.test
    if te is None:
        raise ValueError("no test split in reader output")

    X_train = np.asarray(tr.X_train, dtype=float)
    y_train = np.asarray(tr.y_train, dtype=float).reshape(-1, 1)
    X_test = np.asarray(te.X_train, dtype=float)
    y_test = np.asarray(te.y_train, dtype=float).reshape(-1, 1)
    feature_names = tuple(str(v) for v in tr.feature_names)

    masks_tr = _mask_branches(X_train, feature_names)
    masks_te = _mask_branches(X_test, feature_names)

    branch_models: dict[str, dict[str, Any]] = {}
    for bn in ("holiday", "weekend", "regular"):
        mtr = np.asarray(masks_tr[bn], dtype=bool)
        if int(np.sum(mtr)) < int(max(40, args.min_branch_train)):
            branch_models[bn] = {"branch": bn, "ok": False, "reason": "too_few_train", "n_train": int(np.sum(mtr))}
            continue
        br = _train_branch(
            branch_name=bn,
            X_train=X_train[mtr],
            y_train=y_train[mtr],
            feature_names=feature_names,
            pop_size=int(args.pop_size),
            generations=int(args.generations),
            outer_strategy=str(args.outer_strategy),
            portfolio_phases=str(args.portfolio_phases),
            portfolio_phase_weights=str(args.portfolio_phase_weights),
            moead_neighborhood_size=int(args.moead_neighborhood_size),
            moead_delta=float(args.moead_delta),
            moead_nr=int(args.moead_nr),
            vns_k_max=int(args.vns_k_max),
            vns_batch_size=int(args.vns_batch_size),
            max_terms=int(args.max_terms),
            ridge_l2=float(args.ridge_l2),
            rolling_folds=int(args.rolling_folds),
            rolling_val_ratio=float(args.rolling_val_ratio),
            seed=int(args.seed),
        )
        br["n_train"] = int(np.sum(mtr))
        branch_models[bn] = br

    # Fallback global model when branch missing.
    global_candidates = _build_candidate_pool(X_train, y_train, feature_names=feature_names, topk_for_pairs=6)
    global_genome = [{"name": c.name, "expr": dict(c.expr)} for c in global_candidates[: min(10, len(global_candidates))]]
    fit_global = evaluate_genome_with_ridge(
        global_genome,
        X_train=X_train,
        y_train=y_train,
        X_eval=X_test,
        y_eval=y_test,
        l2=float(max(0.0, args.ridge_l2)),
    )
    pred_piece = np.asarray(fit_global.get("pred_eval"), dtype=float).reshape(-1, 1)

    for bn in ("holiday", "weekend", "regular"):
        mte = np.asarray(masks_te[bn], dtype=bool)
        if int(np.sum(mte)) <= 0:
            continue
        bm = dict(branch_models.get(bn, {}))
        if not bool(bm.get("ok", False)):
            continue
        fit_b = evaluate_genome_with_ridge(
            bm["genome"],
            X_train=X_train[np.asarray(masks_tr[bn], dtype=bool)],
            y_train=y_train[np.asarray(masks_tr[bn], dtype=bool)],
            X_eval=X_test[mte],
            y_eval=y_test[mte],
            l2=float(max(0.0, bm.get("tuned_l2", args.ridge_l2))),
        )
        pred_piece[mte] = np.asarray(fit_b.get("pred_eval"), dtype=float).reshape(-1, 1)

    piece_rmse = _rmse(y_test, pred_piece)

    xgb = XGBoostSurrogateTrainer(
        config=XGBoostTrainerConfig(
            artifact_id="piecewise_bridge_xgb_baseline",
            n_estimators=360,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            tree_method="hist",
            random_seed=42,
        )
    )
    xgb_art = xgb.fit(ProcessedDataset(X_train=X_train, y_train=y_train, feature_names=feature_names, target_names=(str(args.target_col),)))
    xgb_pred = np.asarray(xgb_art.predict(X_test), dtype=float).reshape(-1, 1)
    xgb_rmse = _rmse(y_test, xgb_pred)

    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "out_root": str(out_root),
        "config": {
            "pop_size": int(args.pop_size),
            "generations": int(args.generations),
            "rolling_folds": int(args.rolling_folds),
            "rolling_val_ratio": float(args.rolling_val_ratio),
            "max_terms": int(args.max_terms),
            "ridge_l2": float(args.ridge_l2),
            "outer_strategy": str(args.outer_strategy),
            "portfolio_phases": str(args.portfolio_phases),
            "portfolio_phase_weights": str(args.portfolio_phase_weights),
            "seed": int(args.seed),
            "piecewise_mode": "holiday|weekend|regular",
            "min_branch_train": int(args.min_branch_train),
        },
        "dataset": {
            "n_train": int(X_train.shape[0]),
            "n_test": int(X_test.shape[0]),
            "n_features": int(X_train.shape[1]),
            "feature_names": list(feature_names),
            "branch_train_size": {k: int(np.sum(v)) for k, v in masks_tr.items()},
            "branch_test_size": {k: int(np.sum(v)) for k, v in masks_te.items()},
        },
        "branches": _jsonable(branch_models),
        "test_compare": {
            "piecewise_symbolic_rmse": float(piece_rmse),
            "xgboost_rmse": float(xgb_rmse),
            "delta_piecewise_minus_xgb": float(piece_rmse - xgb_rmse),
        },
    }
    report_path = out_root / "summary.json"
    report_path.write_text(json.dumps(_jsonable(report), ensure_ascii=False, indent=2), encoding="utf-8")

    print("NSGABLACK_PIECEWISE_SYMBOLIC_BRIDGE_DONE")
    print(f"summary={report_path}")
    print(
        "rmse: "
        f"piecewise_symbolic={float(piece_rmse):.6f}, "
        f"xgboost={float(xgb_rmse):.6f}, "
        f"delta={float(piece_rmse - xgb_rmse):.6f}"
    )


if __name__ == "__main__":
    main()
