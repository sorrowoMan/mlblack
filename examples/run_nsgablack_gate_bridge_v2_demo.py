from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

NSGABLACK_ROOT = ROOT.parent / "nsgablack"
if str(NSGABLACK_ROOT) not in sys.path:
    sys.path.append(str(NSGABLACK_ROOT))

from config import TrainerAssemblySpec, build_trainer
from core.common.contracts import ProcessedDataset
from examples.run_new_problem_regime_lag_test import _build_problem
from nsgablack.core.base import BlackBoxProblem
from nsgablack.core.evolution_solver import EvolutionSolver


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


def _fit_artifact(
    *,
    trainer_key: str,
    trainer_params: Mapping[str, Any],
    X: np.ndarray,
    y: np.ndarray,
    feature_names: tuple[str, ...],
    target_name: str = "target",
):
    ds = ProcessedDataset(
        X_train=np.asarray(X, dtype=float),
        y_train=np.asarray(y, dtype=float).reshape(-1, 1),
        feature_names=feature_names,
        target_names=(target_name,),
        metadata={"dataset": "nsgablack_gate_bridge_v2_local_fit"},
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


def _sigmoid(z: np.ndarray) -> np.ndarray:
    zz = np.clip(np.asarray(z, dtype=float), -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-zz))


def _soft_gate(x_col: np.ndarray, thr: float, temp: float) -> np.ndarray:
    t = max(float(temp), 1e-6)
    return _sigmoid((np.asarray(x_col, dtype=float) - float(thr)) / t)


def _quantile_and_temp(x_col: np.ndarray, q: float, tau_q: float) -> tuple[float, float]:
    x = np.asarray(x_col, dtype=float).reshape(-1)
    qq = float(np.clip(q, 0.15, 0.85))
    tq = float(np.clip(tau_q, 0.02, 0.20))
    thr = float(np.quantile(x, qq))
    lo = float(np.quantile(x, max(0.0, qq - tq)))
    hi = float(np.quantile(x, min(1.0, qq + tq)))
    temp = max(abs(hi - lo), 1e-4)
    return thr, temp


@dataclass(frozen=True)
class GateTreeChoice:
    root_feature_index: int
    root_feature_name: str
    root_quantile: float
    root_threshold: float
    root_temp: float
    right_feature_index: int
    right_feature_name: str
    right_quantile: float
    right_threshold: float
    right_temp: float


def _decode_choice(
    x: np.ndarray,
    *,
    gate_feature_indices: tuple[int, ...] = (),
    root_feature_indices: tuple[int, ...] | None = None,
    child_feature_indices: tuple[int, ...] | None = None,
    feature_names: tuple[str, ...],
    X_ref: np.ndarray,
) -> GateTreeChoice:
    xv = np.asarray(x, dtype=float).reshape(-1)
    if xv.shape[0] < 6:
        raise ValueError("GateTreeChoice requires dimension >= 6")

    root_pool = tuple(int(v) for v in (root_feature_indices if root_feature_indices is not None else gate_feature_indices))
    child_pool = tuple(int(v) for v in (child_feature_indices if child_feature_indices is not None else gate_feature_indices))
    if not root_pool or not child_pool:
        raise ValueError("root/child feature pools must be non-empty")

    ridx_local = int(np.clip(int(round(float(xv[0]))), 0, len(root_pool) - 1))
    rfeat = int(root_pool[ridx_local])
    rq = float(np.clip(float(xv[1]), 0.15, 0.85))
    rtq = float(np.clip(float(xv[2]), 0.02, 0.20))
    rthr, rtemp = _quantile_and_temp(np.asarray(X_ref[:, rfeat], dtype=float), rq, rtq)

    cidx_local = int(np.clip(int(round(float(xv[3]))), 0, len(child_pool) - 1))
    cfeat = int(child_pool[cidx_local])
    cq = float(np.clip(float(xv[4]), 0.15, 0.85))
    ctq = float(np.clip(float(xv[5]), 0.02, 0.20))
    cthr, ctemp = _quantile_and_temp(np.asarray(X_ref[:, cfeat], dtype=float), cq, ctq)

    return GateTreeChoice(
        root_feature_index=rfeat,
        root_feature_name=str(feature_names[rfeat]),
        root_quantile=rq,
        root_threshold=rthr,
        root_temp=rtemp,
        right_feature_index=cfeat,
        right_feature_name=str(feature_names[cfeat]),
        right_quantile=cq,
        right_threshold=cthr,
        right_temp=ctemp,
    )


def _fit_tree_predict(
    *,
    choice: GateTreeChoice,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    trainer_key: str,
    trainer_params: Mapping[str, Any],
    feature_names: tuple[str, ...],
    min_leaf: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    xtr = np.asarray(X_train, dtype=float)
    ytr = np.asarray(y_train, dtype=float).reshape(-1, 1)
    xev = np.asarray(X_eval, dtype=float)

    rj = int(choice.root_feature_index)
    cj = int(choice.right_feature_index)
    rthr = float(choice.root_threshold)
    cthr = float(choice.right_threshold)
    rt = float(choice.root_temp)
    ct = float(choice.right_temp)

    # soft-overlap training masks (band around threshold).
    mask_left = np.asarray(xtr[:, rj] <= rthr + rt, dtype=bool)
    mask_rbase = np.asarray(xtr[:, rj] > rthr - rt, dtype=bool)
    mask_rl = mask_rbase & np.asarray(xtr[:, cj] <= cthr + ct, dtype=bool)
    mask_rr = mask_rbase & np.asarray(xtr[:, cj] > cthr - ct, dtype=bool)

    n_left = int(np.sum(mask_left))
    n_rl = int(np.sum(mask_rl))
    n_rr = int(np.sum(mask_rr))

    if min(n_left, n_rl, n_rr) < int(min_leaf):
        raise ValueError(
            f"leaf too small for soft tree: left={n_left}, right_left={n_rl}, right_right={n_rr}, min_leaf={min_leaf}"
        )

    left_art = _fit_artifact(
        trainer_key=trainer_key,
        trainer_params={**dict(trainer_params), "artifact_id": "v2_leaf_left_v1"},
        X=xtr[mask_left],
        y=ytr[mask_left],
        feature_names=feature_names,
    )
    rl_art = _fit_artifact(
        trainer_key=trainer_key,
        trainer_params={**dict(trainer_params), "artifact_id": "v2_leaf_rl_v1"},
        X=xtr[mask_rl],
        y=ytr[mask_rl],
        feature_names=feature_names,
    )
    rr_art = _fit_artifact(
        trainer_key=trainer_key,
        trainer_params={**dict(trainer_params), "artifact_id": "v2_leaf_rr_v1"},
        X=xtr[mask_rr],
        y=ytr[mask_rr],
        feature_names=feature_names,
    )

    g1 = _soft_gate(xev[:, rj], rthr, rt).reshape(-1, 1)
    g2 = _soft_gate(xev[:, cj], cthr, ct).reshape(-1, 1)

    pred_left = np.asarray(left_art.predict(xev), dtype=float).reshape(-1, 1)
    pred_rl = np.asarray(rl_art.predict(xev), dtype=float).reshape(-1, 1)
    pred_rr = np.asarray(rr_art.predict(xev), dtype=float).reshape(-1, 1)

    pred = (1.0 - g1) * pred_left + g1 * ((1.0 - g2) * pred_rl + g2 * pred_rr)
    detail = {
        "n_leaf_left": n_left,
        "n_leaf_right_left": n_rl,
        "n_leaf_right_right": n_rr,
    }
    return np.asarray(pred, dtype=float), detail


def _rolling_splits(n: int, *, folds: int, val_ratio: float, min_train: int) -> list[tuple[np.ndarray, np.ndarray]]:
    nn = int(n)
    ff = int(max(1, folds))
    val_size = max(64, int(round(float(val_ratio) * nn)))
    val_size = min(val_size, max(64, nn // 3))

    start_min = max(int(min_train), val_size + 64)
    start_max = nn - val_size
    if start_max <= start_min:
        split = int(round(nn * 0.75))
        split = max(64, min(split, nn - 64))
        tr = np.arange(0, split, dtype=int)
        va = np.arange(split, nn, dtype=int)
        return [(tr, va)]

    anchors = np.linspace(start_min, start_max, num=ff, dtype=int)
    out: list[tuple[np.ndarray, np.ndarray]] = []
    for s in anchors:
        start = int(s)
        end = min(nn, start + val_size)
        if end - start < 32:
            continue
        tr = np.arange(0, start, dtype=int)
        va = np.arange(start, end, dtype=int)
        if tr.size >= 64 and va.size >= 32:
            out.append((tr, va))
    if not out:
        split = int(round(nn * 0.75))
        split = max(64, min(split, nn - 64))
        out.append((np.arange(0, split, dtype=int), np.arange(split, nn, dtype=int)))
    return out


class GateTreeV2Problem(BlackBoxProblem):
    """V2: soft-gated two-layer symbolic tree with rolling validation objective."""

    def __init__(
        self,
        *,
        X_fit: np.ndarray,
        y_fit: np.ndarray,
        feature_names: tuple[str, ...],
        gate_feature_indices: tuple[int, ...],
        eval_trainer_key: str,
        eval_trainer_params: Mapping[str, Any],
        root_feature_indices: tuple[int, ...] | None = None,
        child_feature_indices: tuple[int, ...] | None = None,
        rolling_folds: int = 2,
        rolling_val_ratio: float = 0.18,
        min_leaf: int = 120,
    ) -> None:
        root_pool = tuple(int(i) for i in (root_feature_indices if root_feature_indices is not None else gate_feature_indices))
        child_pool = tuple(int(i) for i in (child_feature_indices if child_feature_indices is not None else gate_feature_indices))
        if not root_pool or not child_pool:
            raise ValueError("GateTreeV2Problem requires non-empty root/child feature pools")

        super().__init__(
            dimension=6,
            objectives=["minimize", "minimize"],
            bounds=[
                (0.0, float(max(0, len(root_pool) - 1))),
                (0.15, 0.85),
                (0.02, 0.20),
                (0.0, float(max(0, len(child_pool) - 1))),
                (0.15, 0.85),
                (0.02, 0.20),
            ],
        )
        self.name = "GateTreeV2Problem"
        self.X_fit = np.asarray(X_fit, dtype=float)
        self.y_fit = np.asarray(y_fit, dtype=float).reshape(-1, 1)
        self.feature_names = tuple(str(n) for n in feature_names)
        self.gate_feature_indices = tuple(int(i) for i in gate_feature_indices)
        self.root_feature_indices = root_pool
        self.child_feature_indices = child_pool
        self.eval_trainer_key = str(eval_trainer_key)
        self.eval_trainer_params = dict(eval_trainer_params)
        self.min_leaf = int(min_leaf)
        self.splits = _rolling_splits(
            int(self.X_fit.shape[0]),
            folds=int(rolling_folds),
            val_ratio=float(rolling_val_ratio),
            min_train=max(256, int(min_leaf) * 2),
        )
        self._cache: dict[tuple[float, ...], tuple[np.ndarray, dict[str, Any]]] = {}

    def _eval_choice(self, choice: GateTreeChoice) -> tuple[np.ndarray, dict[str, Any]]:
        fold_rmse: list[float] = []
        leaf_stats: list[dict[str, Any]] = []

        for tr_idx, va_idx in self.splits:
            xtr = self.X_fit[tr_idx]
            ytr = self.y_fit[tr_idx]
            xva = self.X_fit[va_idx]
            yva = self.y_fit[va_idx]
            pred, detail = _fit_tree_predict(
                choice=choice,
                X_train=xtr,
                y_train=ytr,
                X_eval=xva,
                trainer_key=self.eval_trainer_key,
                trainer_params=self.eval_trainer_params,
                feature_names=self.feature_names,
                min_leaf=self.min_leaf,
            )
            rmse = float(np.sqrt(np.mean((pred - yva) ** 2)))
            fold_rmse.append(rmse)
            leaf_stats.append(detail)

        rmse_mean = float(np.mean(fold_rmse))
        rmse_std = float(np.std(fold_rmse))
        obj_rmse = float(rmse_mean + 0.35 * rmse_std)

        rj = int(choice.root_feature_index)
        cj = int(choice.right_feature_index)
        hard_left = np.asarray(self.X_fit[:, rj] <= choice.root_threshold, dtype=bool)
        hard_rl = (~hard_left) & np.asarray(self.X_fit[:, cj] <= choice.right_threshold, dtype=bool)
        hard_rr = (~hard_left) & np.asarray(self.X_fit[:, cj] > choice.right_threshold, dtype=bool)
        ratios = np.array(
            [
                float(np.mean(hard_left.astype(float))),
                float(np.mean(hard_rl.astype(float))),
                float(np.mean(hard_rr.astype(float))),
            ],
            dtype=float,
        )
        imbalance = float(np.std(ratios))
        softness = float((choice.root_temp + choice.right_temp) * 0.5)
        obj_complexity = float(imbalance + 0.02 * softness)

        obj = np.array([obj_rmse, obj_complexity], dtype=float)
        detail = {
            "invalid_split": False,
            "fold_rmse": [float(v) for v in fold_rmse],
            "rmse_mean": rmse_mean,
            "rmse_std": rmse_std,
            "obj_rmse": obj_rmse,
            "obj_complexity": obj_complexity,
            "imbalance": imbalance,
            "softness": softness,
            "hard_leaf_ratios": ratios.tolist(),
            "leaf_stats": _jsonable(leaf_stats),
            "eval_trainer_key": self.eval_trainer_key,
            "n_splits": int(len(self.splits)),
        }
        return obj, detail

    def evaluate(self, x):
        xr = tuple(round(float(v), 4) for v in np.asarray(x, dtype=float).reshape(-1))
        hit = self._cache.get(xr)
        if hit is not None:
            return hit[0]

        try:
            choice = _decode_choice(
                np.asarray(x, dtype=float),
                gate_feature_indices=self.gate_feature_indices,
                root_feature_indices=self.root_feature_indices,
                child_feature_indices=self.child_feature_indices,
                feature_names=self.feature_names,
                X_ref=self.X_fit,
            )
            obj, detail = self._eval_choice(choice)
            store = {"choice": _jsonable(asdict(choice)), **detail}
            self._cache[xr] = (obj, store)
            return obj
        except Exception as exc:
            bad = np.array([1e6, 1e3], dtype=float)
            self._cache[xr] = (
                bad,
                {
                    "invalid_split": True,
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )
            return bad

    def cache_snapshot(self) -> dict[str, Any]:
        out = []
        for _, (obj, detail) in self._cache.items():
            out.append(
                {
                    "obj_rmse": float(obj[0]),
                    "obj_complexity": float(obj[1]),
                    **_jsonable(detail),
                }
            )
        out.sort(key=lambda r: (float(r["obj_rmse"]), float(r["obj_complexity"])))
        return {"n_cache": len(out), "entries_top20": out[:20]}


def _fit_global_baselines(
    *,
    train_ds: ProcessedDataset,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    candidates = [
        ("ridge", {"artifact_id": "v2_global_ridge_v1", "l2": 1.0}),
        (
            "xgboost",
            {
                "artifact_id": "v2_global_xgboost_v1",
                "n_estimators": 320,
                "max_depth": 6,
                "learning_rate": 0.05,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
                "tree_method": "hist",
                "random_seed": 42,
            },
        ),
        (
            "symbolic_stagewise",
            {
                "artifact_id": "v2_global_stagewise_v1",
                "force_linear_base": "auto",
                "keep_search_trace": False,
                "auto_val_ratio": 0.2,
                "auto_min_val_samples": 64,
                "auto_random_seed": 42,
                "search_max_added_terms": 8,
                "search_topk_features": 8,
                "search_max_pair_terms": 10,
                "search_max_candidates_per_iter": 220,
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
        ),
    ]
    for key, params in candidates:
        t0 = time.perf_counter()
        spec = TrainerAssemblySpec(
            trainer_key=key,
            trainer_params=dict(params),
            pipeline_key="identity",
            pipeline_params={},
            biases=(),
        )
        trainer = build_trainer(spec)
        art = trainer.fit(train_ds)
        pred = np.asarray(art.predict(X_test), dtype=float).reshape(-1, 1)
        out[key] = {
            "duration_sec": float(time.perf_counter() - t0),
            "metrics_test": _metrics(y_test, pred),
            "artifact_id": str(getattr(art, "artifact_id", "unknown")),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="V2 bridge: soft-gate two-layer tree + rolling objective.")
    parser.add_argument("--pop-size", type=int, default=10, help="Outer NSGABLACK population size.")
    parser.add_argument("--generations", type=int, default=6, help="Outer NSGABLACK max generations.")
    parser.add_argument("--rolling-folds", type=int, default=2, help="Number of rolling validation folds.")
    args = parser.parse_args()

    pop_size = int(max(4, args.pop_size))
    generations = int(max(1, args.generations))
    rolling_folds = int(max(1, args.rolling_folds))

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = ROOT / "examples" / "out" / f"nsgablack_gate_bridge_v2_demo_{stamp}"
    out_root.mkdir(parents=True, exist_ok=True)

    ds, X_test, y_test = _build_problem()
    X_train = np.asarray(ds.X_train, dtype=float)
    y_train = np.asarray(ds.y_train, dtype=float).reshape(-1, 1)
    feature_names = tuple(str(v) for v in (ds.feature_names or tuple(f"x{i}" for i in range(X_train.shape[1]))))

    baseline = _fit_global_baselines(train_ds=ds, X_test=X_test, y_test=y_test)

    gate_feature_names = ("flow", "humidity", "lag1", "lag24", "regime", "hour_sin")
    gate_feature_indices = tuple(feature_names.index(nm) for nm in gate_feature_names if nm in feature_names)
    if not gate_feature_indices:
        raise ValueError("No gate features found in dataset")

    outer_eval_stage_cfg = {
        "force_linear_base": "auto",
        "keep_search_trace": False,
        "auto_val_ratio": 0.2,
        "auto_min_val_samples": 64,
        "auto_random_seed": 42,
        "search_max_added_terms": 3,
        "search_topk_features": 6,
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

    problem = GateTreeV2Problem(
        X_fit=X_train,
        y_fit=y_train,
        feature_names=feature_names,
        gate_feature_indices=gate_feature_indices,
        eval_trainer_key="symbolic_stagewise",
        eval_trainer_params=outer_eval_stage_cfg,
        rolling_folds=rolling_folds,
        rolling_val_ratio=0.18,
        min_leaf=140,
    )
    solver = EvolutionSolver(problem, pop_size=pop_size, max_generations=generations)
    t0 = time.perf_counter()
    run = solver.run(return_dict=True)
    outer_sec = float(time.perf_counter() - t0)

    pareto_obj = np.asarray(run.get("pareto_objectives"), dtype=float)
    pareto_ind = np.asarray(run.get("pareto_solutions", {}).get("individuals", []), dtype=float)
    if pareto_obj.size == 0 or pareto_ind.size == 0:
        raise RuntimeError("nsgablack returned empty pareto set")
    if pareto_obj.ndim == 1:
        pareto_obj = pareto_obj.reshape(-1, 1)

    selection_score = pareto_obj[:, 0] + 0.15 * pareto_obj[:, 1]
    best_idx = int(np.argmin(selection_score))
    best_x = pareto_ind[best_idx]
    best_choice = _decode_choice(
        best_x,
        gate_feature_indices=gate_feature_indices,
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
        "search_topk_features": 8,
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
        min_leaf=140,
    )
    v2_metrics = _metrics(y_test, pred_test)

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
            "selected_choice": _jsonable(asdict(best_choice)),
            "selected_pareto_obj": _jsonable(pareto_obj[best_idx]),
            "selected_score": float(selection_score[best_idx]),
            "cache": problem.cache_snapshot(),
            "outer_eval": {
                "trainer_key": "symbolic_stagewise",
                "trainer_params": _jsonable(outer_eval_stage_cfg),
                "tree": "soft_root + soft_right_child",
                "validation": "rolling",
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

    print("NSGABLACK_GATE_BRIDGE_V2_DEMO_DONE")
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
