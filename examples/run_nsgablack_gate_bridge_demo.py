from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

NSGABLACK_ROOT = ROOT.parent / "nsgablack"
if str(NSGABLACK_ROOT) not in sys.path:
    # Keep mlblack imports higher priority to avoid module-name collisions (e.g., bias).
    sys.path.append(str(NSGABLACK_ROOT))

from config import TrainerAssemblySpec, build_trainer
from core.common.contracts import ProcessedDataset, SurrogateArtifact
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
) -> SurrogateArtifact:
    ds = ProcessedDataset(
        X_train=np.asarray(X, dtype=float),
        y_train=np.asarray(y, dtype=float).reshape(-1, 1),
        feature_names=feature_names,
        target_names=(target_name,),
        metadata={"dataset": "nsgablack_gate_bridge_local_fit"},
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


@dataclass(frozen=True)
class GateChoice:
    feature_index: int
    feature_name: str
    quantile: float
    threshold: float


def _decode_gate(
    x: np.ndarray,
    *,
    gate_feature_indices: tuple[int, ...],
    feature_names: tuple[str, ...],
    X_ref: np.ndarray,
    q_min: float = 0.15,
    q_max: float = 0.85,
) -> GateChoice:
    g_raw = float(np.asarray(x, dtype=float).reshape(-1)[0])
    q_raw = float(np.asarray(x, dtype=float).reshape(-1)[1])
    idx_local = int(np.clip(int(round(g_raw)), 0, len(gate_feature_indices) - 1))
    f_idx = int(gate_feature_indices[idx_local])
    q = float(np.clip(q_raw, q_min, q_max))
    thr = float(np.quantile(np.asarray(X_ref[:, f_idx], dtype=float), q))
    return GateChoice(
        feature_index=f_idx,
        feature_name=str(feature_names[f_idx]),
        quantile=q,
        threshold=thr,
    )


class GateThresholdProblem(BlackBoxProblem):
    """nsgablack 外层问题：搜索 gate(feature, threshold quantile)."""

    def __init__(
        self,
        *,
        X_fit: np.ndarray,
        y_fit: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        feature_names: tuple[str, ...],
        gate_feature_indices: tuple[int, ...],
        eval_trainer_key: str = "symbolic_stagewise",
        eval_trainer_params: Mapping[str, Any] | None = None,
        min_leaf: int = 120,
    ) -> None:
        super().__init__(
            dimension=2,
            objectives=["minimize", "minimize"],
            bounds=[
                (0.0, float(max(0, len(gate_feature_indices) - 1))),
                (0.15, 0.85),
            ],
        )
        self.name = "GateThresholdProblem"
        self.X_fit = np.asarray(X_fit, dtype=float)
        self.y_fit = np.asarray(y_fit, dtype=float).reshape(-1, 1)
        self.X_val = np.asarray(X_val, dtype=float)
        self.y_val = np.asarray(y_val, dtype=float).reshape(-1, 1)
        self.feature_names = tuple(str(n) for n in feature_names)
        self.gate_feature_indices = tuple(int(i) for i in gate_feature_indices)
        self.eval_trainer_key = str(eval_trainer_key)
        self.eval_trainer_params = dict(eval_trainer_params or {})
        self.min_leaf = int(min_leaf)

        self._cache: dict[tuple[int, float], tuple[np.ndarray, dict[str, Any]]] = {}

    def _eval_gate(self, choice: GateChoice) -> tuple[np.ndarray, dict[str, Any]]:
        f_idx = int(choice.feature_index)
        thr = float(choice.threshold)

        fit_mask_left = np.asarray(self.X_fit[:, f_idx] <= thr, dtype=bool)
        fit_mask_right = ~fit_mask_left
        val_mask_left = np.asarray(self.X_val[:, f_idx] <= thr, dtype=bool)
        val_mask_right = ~val_mask_left

        n_fit_left = int(np.sum(fit_mask_left))
        n_fit_right = int(np.sum(fit_mask_right))
        n_val_left = int(np.sum(val_mask_left))
        n_val_right = int(np.sum(val_mask_right))

        if (
            n_fit_left < self.min_leaf
            or n_fit_right < self.min_leaf
            or n_val_left < max(20, self.min_leaf // 5)
            or n_val_right < max(20, self.min_leaf // 5)
        ):
            bad = np.array([1e6, 1e3], dtype=float)
            detail = {
                "invalid_split": True,
                "n_fit_left": n_fit_left,
                "n_fit_right": n_fit_right,
                "n_val_left": n_val_left,
                "n_val_right": n_val_right,
            }
            return bad, detail

        # 全量评估：外层每个候选 gate 都直接调用 mlblack trainer（默认 symbolic_stagewise）。
        left_art = _fit_artifact(
            trainer_key=self.eval_trainer_key,
            trainer_params={**self.eval_trainer_params, "artifact_id": "gate_left_outer_eval_v1"},
            X=self.X_fit[fit_mask_left],
            y=self.y_fit[fit_mask_left],
            feature_names=self.feature_names,
        )
        right_art = _fit_artifact(
            trainer_key=self.eval_trainer_key,
            trainer_params={**self.eval_trainer_params, "artifact_id": "gate_right_outer_eval_v1"},
            X=self.X_fit[fit_mask_right],
            y=self.y_fit[fit_mask_right],
            feature_names=self.feature_names,
        )

        pred = np.zeros_like(self.y_val, dtype=float)
        pred[val_mask_left] = np.asarray(left_art.predict(self.X_val[val_mask_left]), dtype=float).reshape(-1, 1)
        pred[val_mask_right] = np.asarray(right_art.predict(self.X_val[val_mask_right]), dtype=float).reshape(-1, 1)

        rmse = float(np.sqrt(np.mean((pred - self.y_val) ** 2)))
        imbalance = abs((n_fit_left / max(1, (n_fit_left + n_fit_right))) - 0.5)
        complexity = float(imbalance)

        obj = np.array([rmse, complexity], dtype=float)
        detail = {
            "invalid_split": False,
            "rmse_outer_eval": rmse,
            "eval_trainer_key": self.eval_trainer_key,
            "imbalance": float(imbalance),
            "n_fit_left": n_fit_left,
            "n_fit_right": n_fit_right,
            "n_val_left": n_val_left,
            "n_val_right": n_val_right,
        }
        return obj, detail

    def evaluate(self, x):
        choice = _decode_gate(
            np.asarray(x, dtype=float),
            gate_feature_indices=self.gate_feature_indices,
            feature_names=self.feature_names,
            X_ref=self.X_fit,
        )
        key = (int(choice.feature_index), round(float(choice.quantile), 4))
        hit = self._cache.get(key)
        if hit is not None:
            return hit[0]
        obj, detail = self._eval_gate(choice)
        self._cache[key] = (obj, {"choice": _jsonable(choice), **detail})
        return obj

    def cache_snapshot(self) -> dict[str, Any]:
        out = []
        for _, (obj, detail) in self._cache.items():
            out.append(
                {
                    "obj_rmse_proxy": float(obj[0]),
                    "obj_complexity": float(obj[1]),
                    **_jsonable(detail),
                }
            )
        out.sort(key=lambda r: (float(r["obj_rmse_proxy"]), float(r["obj_complexity"])))
        return {"n_cache": len(out), "entries_top20": out[:20]}


def _fit_global_baselines(
    *,
    train_ds: ProcessedDataset,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, params in [
        ("ridge", {"artifact_id": "global_ridge_v1", "l2": 1.0}),
        (
            "xgboost",
            {
                "artifact_id": "global_xgboost_v1",
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
                "artifact_id": "global_stagewise_v1",
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
    ]:
        t0 = time.perf_counter()
        spec = TrainerAssemblySpec(
            trainer_key=name,
            trainer_params=dict(params),
            pipeline_key="identity",
            pipeline_params={},
            biases=(),
        )
        trainer = build_trainer(spec)
        art = trainer.fit(train_ds)
        pred = np.asarray(art.predict(X_test), dtype=float).reshape(-1, 1)
        out[name] = {
            "duration_sec": float(time.perf_counter() - t0),
            "metrics_test": _metrics(y_test, pred),
            "artifact_id": str(getattr(art, "artifact_id", "unknown")),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="NSGABLACK gate search + mlblack local symbolic bridge demo.")
    parser.add_argument("--pop-size", type=int, default=12, help="Outer NSGABLACK population size.")
    parser.add_argument("--generations", type=int, default=6, help="Outer NSGABLACK max generations.")
    args = parser.parse_args()

    pop_size = int(max(4, args.pop_size))
    generations = int(max(1, args.generations))

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = ROOT / "examples" / "out" / f"nsgablack_gate_bridge_demo_{stamp}"
    out_root.mkdir(parents=True, exist_ok=True)

    ds, X_test, y_test = _build_problem()
    X_train = np.asarray(ds.X_train, dtype=float)
    y_train = np.asarray(ds.y_train, dtype=float).reshape(-1, 1)
    feature_names = tuple(str(v) for v in (ds.feature_names or tuple(f"x{i}" for i in range(X_train.shape[1]))))

    # 外层搜索用 train 内部分验证，最终评估看 test。
    n = int(X_train.shape[0])
    cut = int(round(0.8 * n))
    cut = max(200, min(cut, n - 200))
    X_fit, y_fit = X_train[:cut], y_train[:cut]
    X_val, y_val = X_train[cut:], y_train[cut:]

    gate_feature_names = ("flow", "humidity", "lag1", "lag24", "regime", "hour_sin")
    gate_feature_indices = tuple(feature_names.index(nm) for nm in gate_feature_names if nm in feature_names)
    if not gate_feature_indices:
        raise ValueError("No gate features found in dataset")

    baseline = _fit_global_baselines(train_ds=ds, X_test=X_test, y_test=y_test)

    outer_eval_stage_cfg = {
        "force_linear_base": "auto",
        "keep_search_trace": False,
        "auto_val_ratio": 0.2,
        "auto_min_val_samples": 64,
        "auto_random_seed": 42,
        "search_max_added_terms": 6,
        "search_topk_features": 8,
        "search_max_pair_terms": 8,
        "search_max_candidates_per_iter": 140,
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

    problem = GateThresholdProblem(
        X_fit=X_fit,
        y_fit=y_fit,
        X_val=X_val,
        y_val=y_val,
        feature_names=feature_names,
        gate_feature_indices=gate_feature_indices,
        eval_trainer_key="symbolic_stagewise",
        eval_trainer_params=outer_eval_stage_cfg,
        min_leaf=120,
    )
    solver = EvolutionSolver(problem, pop_size=pop_size, max_generations=generations)
    t0 = time.perf_counter()
    run = solver.run(return_dict=True)
    gate_search_sec = float(time.perf_counter() - t0)

    pareto_obj = np.asarray(run.get("pareto_objectives"), dtype=float)
    pareto_ind = np.asarray(run.get("pareto_solutions", {}).get("individuals", []), dtype=float)
    if pareto_obj.size == 0 or pareto_ind.size == 0:
        raise RuntimeError("nsgablack returned empty pareto set")
    if pareto_obj.ndim == 1:
        pareto_obj = pareto_obj.reshape(-1, 1)

    score = pareto_obj[:, 0] + 0.10 * pareto_obj[:, 1]
    best_idx = int(np.argmin(score))
    best_x = pareto_ind[best_idx]
    best_choice = _decode_gate(
        best_x,
        gate_feature_indices=gate_feature_indices,
        feature_names=feature_names,
        X_ref=X_train,
    )

    # 最终局部模型：mlblack symbolic_stagewise。
    f_idx = int(best_choice.feature_index)
    thr = float(best_choice.threshold)

    tr_left = np.asarray(X_train[:, f_idx] <= thr, dtype=bool)
    tr_right = ~tr_left
    te_left = np.asarray(X_test[:, f_idx] <= thr, dtype=bool)
    te_right = ~te_left

    local_stage_cfg = {
        "artifact_id": "local_symbolic_stagewise_v1",
        "force_linear_base": "auto",
        "keep_search_trace": False,
        "auto_val_ratio": 0.2,
        "auto_min_val_samples": 64,
        "auto_random_seed": 42,
        "search_max_added_terms": 6,
        "search_topk_features": 8,
        "search_max_pair_terms": 8,
        "search_max_candidates_per_iter": 160,
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
    left_art = _fit_artifact(
        trainer_key="symbolic_stagewise",
        trainer_params=local_stage_cfg,
        X=X_train[tr_left],
        y=y_train[tr_left],
        feature_names=feature_names,
    )
    right_art = _fit_artifact(
        trainer_key="symbolic_stagewise",
        trainer_params=local_stage_cfg,
        X=X_train[tr_right],
        y=y_train[tr_right],
        feature_names=feature_names,
    )

    pred_test = np.zeros((X_test.shape[0], 1), dtype=float)
    pred_test[te_left] = np.asarray(left_art.predict(X_test[te_left]), dtype=float).reshape(-1, 1)
    pred_test[te_right] = np.asarray(right_art.predict(X_test[te_right]), dtype=float).reshape(-1, 1)
    gated_metrics = _metrics(y_test, pred_test)

    summary = {
        "timestamp": datetime.now().isoformat(),
        "output_root": str(out_root),
        "dataset": _jsonable(ds.metadata),
        "baseline": baseline,
        "nsgablack_search": {
            "duration_sec": gate_search_sec,
            "pop_size": int(pop_size),
            "max_generations": int(generations),
            "requested_budget": int(pop_size * generations),
            "evaluation_count": int(run.get("evaluation_count", 0)),
            "generation": int(run.get("generation", 0)),
            "selected_gate": {
                "feature_index": int(best_choice.feature_index),
                "feature_name": str(best_choice.feature_name),
                "quantile": float(best_choice.quantile),
                "threshold": float(best_choice.threshold),
                "train_left_ratio": float(np.mean(tr_left.astype(float))),
                "test_left_ratio": float(np.mean(te_left.astype(float))),
                "pareto_objective": _jsonable(pareto_obj[best_idx]),
                "selection_score": float(score[best_idx]),
            },
            "cache": problem.cache_snapshot(),
            "outer_evaluation": {
                "trainer_key": "symbolic_stagewise",
                "trainer_params": _jsonable(outer_eval_stage_cfg),
                "mode": "full",
            },
        },
        "gated_symbolic_local": {
            "metrics_test": gated_metrics,
            "n_train_left": int(np.sum(tr_left)),
            "n_train_right": int(np.sum(tr_right)),
            "n_test_left": int(np.sum(te_left)),
            "n_test_right": int(np.sum(te_right)),
        },
        "delta_vs_global_stagewise_rmse": float(
            gated_metrics["rmse"] - float(baseline["symbolic_stagewise"]["metrics_test"]["rmse"])
        ),
    }

    (out_root / "summary.json").write_text(
        json.dumps(_jsonable(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("NSGABLACK_GATE_BRIDGE_DEMO_DONE")
    print(f"output_root={out_root}")
    print("baseline_rmse:")
    print(f"  ridge={float(baseline['ridge']['metrics_test']['rmse']):.6f}")
    print(f"  xgboost={float(baseline['xgboost']['metrics_test']['rmse']):.6f}")
    print(f"  global_stagewise={float(baseline['symbolic_stagewise']['metrics_test']['rmse']):.6f}")
    print("gated_symbolic_local_rmse={:.6f}".format(float(gated_metrics["rmse"])))
    print(
        "selected_gate={} <= {:.6f} (q={:.3f})".format(
            str(best_choice.feature_name),
            float(best_choice.threshold),
            float(best_choice.quantile),
        )
    )
    print(f"summary={out_root / 'summary.json'}")


if __name__ == "__main__":
    main()
