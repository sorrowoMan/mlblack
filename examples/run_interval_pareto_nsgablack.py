from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

MLBLACK_ROOT = Path(__file__).resolve().parents[1]
if str(MLBLACK_ROOT) not in sys.path:
    sys.path.insert(0, str(MLBLACK_ROOT))

from project.scaffold import ScaffoldSpec, load_scaffold_spec, run_project_scaffold
from examples.path_defaults import apply_env_defaults, default_nsgablack_root
from nowcasting_work_ci.mlblack_side.problem.domain_router import WORK_CI_STRICT4_GATE_NAMES

DEFAULT_GATE_FEATURES = WORK_CI_STRICT4_GATE_NAMES


def _to_jsonable(v: Any) -> Any:
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, dict):
        return {str(k): _to_jsonable(val) for k, val in v.items()}
    if isinstance(v, (list, tuple, set)):
        return [_to_jsonable(x) for x in v]
    return str(v)


def _log10_interp(z: float, low: float, high: float) -> float:
    if low <= 0.0 or high <= 0.0 or low >= high:
        raise ValueError("require 0 < low < high for log-scale mapping")
    a = math.log10(low)
    b = math.log10(high)
    return float(10 ** (a + float(z) * (b - a)))


def _interval_metrics(artifact: Any, X_test: np.ndarray | None, y_test: np.ndarray | None) -> tuple[float, float]:
    if X_test is None or y_test is None:
        raise ValueError("test split missing")
    if not hasattr(artifact, "predict_interval"):
        raise TypeError("artifact has no predict_interval()")

    lo, hi = artifact.predict_interval(X_test)
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    yt = np.asarray(y_test, dtype=float)

    if yt.ndim == 1:
        yt = yt.reshape(-1, 1)
    if lo.ndim == 1:
        lo = lo.reshape(-1, 1)
    if hi.ndim == 1:
        hi = hi.reshape(-1, 1)

    inside = (yt >= lo) & (yt <= hi)
    width = np.maximum(hi - lo, 0.0)
    return float(np.mean(inside)), float(np.mean(width))


def _vector_key(x: np.ndarray, ndigits: int = 8) -> str:
    arr = np.asarray(x, dtype=float).reshape(-1)
    rounded = [round(float(v), ndigits) for v in arr]
    return json.dumps(rounded, ensure_ascii=False)


class MLBlackIntervalOuterProblem:  # dynamic mixin target for BlackBoxProblem
    pass


def build_problem_class(BlackBoxProblem):
    class IntervalTrainerOuterProblem(BlackBoxProblem):
        """Outer multi-objective problem: nsgablack evaluates hparams, mlblack trains inner interval model."""

        def __init__(
            self,
            *,
            base_spec: ScaffoldSpec,
            output_root: Path,
            target_coverage: float,
            max_good_width: float,
            inner_epochs: int,
            default_seed: int,
            force_advanced: bool,
            force_stagewise_warmup: bool,
            force_gate_piecewise: bool,
            gate_blend_kappa: float,
            path_memory_db_path: str,
            path_memory_namespace: str,
        ) -> None:
            self.base_spec = base_spec
            self.output_root = output_root
            self.target_coverage = float(target_coverage)
            self.max_good_width = float(max_good_width)
            self.inner_epochs = int(inner_epochs)
            self.default_seed = int(default_seed)
            self.force_advanced = bool(force_advanced)
            self.force_stagewise_warmup = bool(force_stagewise_warmup)
            self.force_gate_piecewise = bool(force_gate_piecewise)
            self.gate_blend_kappa = float(max(1e-8, float(gate_blend_kappa)))
            self.path_memory_db_path = str(path_memory_db_path).strip()
            self.path_memory_namespace = str(path_memory_namespace).strip() or "interval_outer"

            self.records: list[dict[str, Any]] = []
            self._cache: dict[str, np.ndarray] = {}
            self._record_by_key: dict[str, dict[str, Any]] = {}
            self._trial_counter = 0

            bounds = [
                (0.05, 0.20),
                (0.78, 0.97),
                (0.00, 1.00),
                (0.00, 1.00),
                (0.00, 1.00),
                (8.0, 24.0),
                (4.0, 10.0),
                (0.75, 0.95),
                (0.00, 1.00),
                (0.0, 1.0),
            ]

            super().__init__(
                name="mlblack_interval_outer_nsgablack",
                dimension=len(bounds),
                objectives=["minimize", "minimize", "minimize"],
                bounds=bounds,
            )

        def _decode_params(self, x: np.ndarray) -> dict[str, Any]:
            arr = np.asarray(x, dtype=float).reshape(-1)
            if arr.size != 10:
                raise ValueError(f"expected 10 vars, got {arr.size}")

            ql = float(arr[0])
            qh = float(arr[1])
            if qh < ql + 0.5:
                qh = min(0.97, ql + 0.5)

            max_interactions = int(round(float(arr[5])))
            topk = int(round(float(arr[6])))

            hinge_sets = (
                [0.25, 0.5, 0.75],
                [0.2, 0.5, 0.8],
                [0.33, 0.66],
            )
            hinge_set = hinge_sets[max(0, min(len(hinge_sets) - 1, int(round(float(arr[6])) % len(hinge_sets))))]

            params = dict(self.base_spec.train.trainer_params)
            params.update(
                {
                    "version": "v2",
                    "lower_quantile": float(round(ql, 4)),
                    "upper_quantile": float(round(qh, 4)),
                    "v2_continuous_ops": ["identity", "sin", "cos"],
                    "v2_binary_ops": ["identity"],
                    "v2_include_interactions": True,
                    "v2_max_interactions": int(max(6, min(30, max_interactions))),
                    "v2_topk_features": int(max(3, min(12, topk))),
                    "v2_include_hinge": bool(float(arr[9]) >= 0.5),
                    "v2_hinge_quantiles": [float(v) for v in hinge_set],
                    "order_penalty": float(_log10_interp(float(arr[2]), 1.5, 40.0)),
                    "width_penalty": float(_log10_interp(float(arr[3]), 1e-5, 5e-2)),
                    "lr": float(_log10_interp(float(arr[4]), 3e-4, 3e-3)),
                    "weight_decay": float(params.get("weight_decay", 1e-4)),
                    "l1_readout": float(_log10_interp(float(arr[8]), 1e-6, 2e-3)),
                    "l1_params": 0.0,
                    "conformal_calibration": True,
                    "conformal_level": float(round(float(arr[7]), 4)),
                    "epochs": int(self.inner_epochs),
                    "batch_size": int(params.get("batch_size", 128)),
                    "device": "cpu",
                    "random_seed": int(self.default_seed + self._trial_counter),
                }
            )
            if bool(self.force_advanced):
                if bool(self.force_stagewise_warmup):
                    params["stagewise_warmup_enabled"] = True
                    warm = dict(params.get("stagewise_warmup_params", {}))
                    warm.setdefault("search_max_added_terms", 4)
                    warm.setdefault("search_topk_features", 8)
                    warm.setdefault("search_max_pair_terms", 8)
                    warm.setdefault("search_max_candidates_per_iter", 120)
                    warm.setdefault("search_candidate_keep_top", 8)
                    warm.setdefault("search_include_hinge", True)
                    warm.setdefault("search_hinge_quantiles", [0.25, 0.5, 0.75])
                    warm.setdefault("search_unary_ops", ["square", "sin", "cos"])
                    warm.setdefault("search_enable_prune", True)
                    warm["search_path_memory_enabled"] = True
                    warm.setdefault("search_path_memory_namespace", str(self.path_memory_namespace))
                    if str(self.path_memory_db_path):
                        warm["search_path_memory_db_path"] = str(self.path_memory_db_path)
                    params["stagewise_warmup_params"] = warm

                if bool(self.force_gate_piecewise):
                    params["gate_piecewise_enabled"] = True
                    params.setdefault("gate_feature_names", list(DEFAULT_GATE_FEATURES))
                    params["gate_blend_kappa"] = float(self.gate_blend_kappa)
            return params

        def evaluate_constraints(self, x):
            arr = np.asarray(x, dtype=float).reshape(-1)
            ql, qh = float(arr[0]), float(arr[1])
            g = ql + 0.5 - qh
            return np.asarray([float(g)], dtype=float)

        def evaluate(self, x):
            arr = np.asarray(x, dtype=float).reshape(-1)
            key = _vector_key(arr)
            cached = self._cache.get(key)
            if cached is not None:
                return cached

            params = self._decode_params(arr)
            trial_id = f"eval_{self._trial_counter:04d}"
            self._trial_counter += 1
            trial_dir = self.output_root / "trials" / trial_id
            trial_dir.mkdir(parents=True, exist_ok=True)

            train_spec = replace(
                self.base_spec.train,
                trainer_params=params,
                output_dir=str(trial_dir),
                run_name=f"{self.base_spec.train.run_name}_{trial_id}",
            )
            trial_spec = ScaffoldSpec(data=self.base_spec.data, train=train_spec)

            t0 = time.perf_counter()
            row: dict[str, Any] = {
                "trial_id": trial_id,
                "x": arr.tolist(),
                "status": "failed",
                "params": params,
                "output_dir": str(trial_dir),
                "obj_coverage_error": float("inf"),
                "obj_mean_width_excess": float("inf"),
                "obj_mean_width": float("inf"),
                "obj_rmse": float("inf"),
                "test_coverage": float("nan"),
                "test_mean_width": float("nan"),
                "test_rmse": float("nan"),
                "test_mae": float("nan"),
                "test_r2": float("nan"),
                "duration_sec": float("nan"),
                "error": "",
            }

            try:
                result = run_project_scaffold(trial_spec)
                test_metrics = result.metrics.get("test", {})
                rmse = float(test_metrics.get("rmse", float("nan")))
                mae = float(test_metrics.get("mae", float("nan")))
                r2 = float(test_metrics.get("r2", float("nan")))

                cov, width = _interval_metrics(result.artifact, result.processed.X_test, result.processed.y_test)
                width_excess = float(max(0.0, width - self.max_good_width))

                o1 = float(abs(cov - self.target_coverage))
                o2 = float(width_excess)
                o3 = float(rmse)

                row.update(
                    {
                        "status": "ok",
                        "obj_coverage_error": o1,
                        "obj_mean_width_excess": o2,
                        "obj_mean_width": float(width),
                        "obj_rmse": o3,
                        "test_coverage": float(cov),
                        "test_mean_width": float(width),
                        "test_rmse": float(rmse),
                        "test_mae": float(mae),
                        "test_r2": float(r2),
                    }
                )
                obj = np.asarray([o1, o2, o3], dtype=float)
            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"
                obj = np.asarray([1e6, 1e6, 1e6], dtype=float)

            row["duration_sec"] = float(time.perf_counter() - t0)
            self.records.append(row)
            self._record_by_key[key] = row
            self._cache[key] = obj

            if row["status"] == "ok":
                print(
                    f"[{trial_id}] ok  rmse={row['test_rmse']:.4f} coverage={row['test_coverage']:.4f} "
                    f"width={row['test_mean_width']:.4f} obj=({row['obj_coverage_error']:.4f},"
                    f"{row['obj_mean_width_excess']:.4f},{row['obj_rmse']:.4f}) t={row['duration_sec']:.1f}s"
                )
            else:
                print(f"[{trial_id}] failed {row['error']} t={row['duration_sec']:.1f}s")

            return obj

        def lookup_record(self, x: np.ndarray) -> dict[str, Any] | None:
            return self._record_by_key.get(_vector_key(x))

    return IntervalTrainerOuterProblem


def _pareto_from_result(problem: Any, result: dict[str, Any]) -> list[dict[str, Any]]:
    ps = result.get("pareto_solutions") or {}
    xs = np.asarray(ps.get("individuals", []), dtype=float)
    fs = np.asarray(ps.get("objectives", []), dtype=float)

    if xs.ndim == 1 and xs.size > 0:
        xs = xs.reshape(1, -1)
    if fs.ndim == 1 and fs.size > 0:
        fs = fs.reshape(1, -1)

    rows: list[dict[str, Any]] = []
    n = int(min(xs.shape[0] if xs.ndim == 2 else 0, fs.shape[0] if fs.ndim == 2 else 0))
    for i in range(n):
        x = xs[i]
        f = fs[i]
        rec = problem.lookup_record(x)
        row = {
            "pareto_index": int(i),
            "x": x.tolist(),
            "objectives": f.tolist(),
            "obj_coverage_error": float(f[0]) if f.size >= 1 else float("nan"),
            "obj_mean_width_excess": float(f[1]) if f.size >= 2 else float("nan"),
            "obj_rmse": float(f[2]) if f.size >= 3 else float("nan"),
            "record": rec,
        }
        rows.append(row)
    return rows


def _pick_compromise(pareto_rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not pareto_rows:
        return None
    c = np.asarray([float(r.get("obj_coverage_error", np.inf)) for r in pareto_rows], dtype=float)
    w = np.asarray([float(r.get("obj_mean_width_excess", np.inf)) for r in pareto_rows], dtype=float)
    r = np.asarray([float(r.get("obj_rmse", np.inf)) for r in pareto_rows], dtype=float)

    def norm(a: np.ndarray) -> np.ndarray:
        lo = float(np.nanmin(a))
        hi = float(np.nanmax(a))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo <= 1e-12:
            return np.zeros_like(a)
        return (a - lo) / (hi - lo)

    score = 0.4 * norm(c) + 0.35 * norm(w) + 0.25 * norm(r)
    idx = int(np.nanargmin(score))
    out = dict(pareto_rows[idx])
    out["compromise_score"] = float(score[idx])
    return out


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_to_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "trial_id",
        "status",
        "obj_coverage_error",
        "obj_mean_width_excess",
        "obj_mean_width",
        "obj_rmse",
        "test_coverage",
        "test_mean_width",
        "test_rmse",
        "test_mae",
        "test_r2",
        "duration_sec",
        "output_dir",
        "params_json",
        "error",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        for row in rows:
            out = {k: row.get(k, "") for k in headers}
            out["params_json"] = json.dumps(row.get("params", {}), ensure_ascii=False, sort_keys=True)
            w.writerow(out)


def main() -> None:
    apply_env_defaults()

    parser = argparse.ArgumentParser(description="Use nsgablack as outer Pareto optimizer for mlblack interval trainer.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(MLBLACK_ROOT / "examples" / "configs" / "work_ci_symbolic_torch_interval_no_flow_speed_occ.json"),
        help="Base mlblack scaffold config (must be symbolic_torch_interval).",
    )
    parser.add_argument(
        "--nsgablack-root",
        type=str,
        default=default_nsgablack_root(),
        help="Path to nsgablack repo root (contains nsgablack/__init__.py).",
    )
    parser.add_argument(
        "--base-params-json",
        type=str,
        default="",
        help="Optional JSON to update base inner trainer params before outer search.",
    )
    parser.add_argument("--pop-size", type=int, default=5, help="Outer population size.")
    parser.add_argument("--generations", type=int, default=1, help="Outer generations.")
    parser.add_argument("--seed", type=int, default=42, help="Outer random seed.")
    parser.add_argument("--inner-epochs", type=int, default=180, help="Inner trainer epochs per evaluation.")
    parser.add_argument("--force-advanced", type=int, default=1, help="1=force advanced inner features, 0=keep config.")
    parser.add_argument("--force-stagewise-warmup", type=int, default=1, help="1=force stagewise warmup on.")
    parser.add_argument("--force-gate-piecewise", type=int, default=1, help="1=force holiday gate piecewise on.")
    parser.add_argument("--gate-blend-kappa", type=float, default=768.0, help="gate_piecewise blend kappa.")
    parser.add_argument("--path-memory-db-path", type=str, default="", help="Optional sqlite path for stagewise path memory.")
    parser.add_argument("--path-memory-namespace", type=str, default="interval_outer", help="Path-memory namespace.")
    parser.add_argument(
        "--max-good-width",
        type=float,
        default=20.0,
        help="Good interval width threshold. objective_2 = max(mean_width - threshold, 0).",
    )
    parser.add_argument(
        "--target-coverage",
        type=float,
        default=None,
        help="Coverage target in objective |coverage-target|. Default: conformal_level or upper-lower in config.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="",
        help="Output root. Default: sibling folder near base output dir.",
    )
    args = parser.parse_args()

    ns_root = Path(args.nsgablack_root).resolve()
    if not (ns_root / "nsgablack" / "__init__.py").exists():
        raise FileNotFoundError(f"invalid nsgablack root: {ns_root}")
    if str(ns_root) not in sys.path:
        sys.path.insert(0, str(ns_root))

    from nsgablack.core.base import BlackBoxProblem
    from nsgablack.core.evolution_solver import EvolutionSolver

    base_spec = load_scaffold_spec(args.config)
    if str(base_spec.train.trainer_key).strip().lower() != "symbolic_torch_interval":
        raise ValueError("config.train.trainer_key must be 'symbolic_torch_interval'")

    if str(args.base_params_json).strip():
        payload = json.loads(Path(args.base_params_json).read_text(encoding="utf-8-sig"))
        if not isinstance(payload, dict):
            raise ValueError("base_params_json must be dict or {'params': {...}}")
        if isinstance(payload.get("params"), dict):
            merged_params = dict(base_spec.train.trainer_params)
            merged_params.update(dict(payload["params"]))
        else:
            merged_params = dict(base_spec.train.trainer_params)
            merged_params.update(payload)
        base_spec = ScaffoldSpec(
            data=base_spec.data,
            train=replace(base_spec.train, trainer_params=merged_params),
        )

    base_params = dict(base_spec.train.trainer_params)
    if args.target_coverage is None:
        if "conformal_level" in base_params:
            target_coverage = float(base_params["conformal_level"])
        else:
            target_coverage = float(base_params.get("upper_quantile", 0.9)) - float(base_params.get("lower_quantile", 0.1))
    else:
        target_coverage = float(args.target_coverage)
    target_coverage = float(np.clip(target_coverage, 0.0, 1.0))

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_out = Path(base_spec.train.output_dir)
    if str(args.output_root).strip():
        out_root = Path(args.output_root).resolve()
    else:
        out_root = (base_out.parent / f"{base_spec.train.run_name}_nsgablack_pareto_{stamp}").resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    if str(args.path_memory_db_path).strip():
        Path(args.path_memory_db_path).resolve().parent.mkdir(parents=True, exist_ok=True)

    ProblemCls = build_problem_class(BlackBoxProblem)
    problem = ProblemCls(
        base_spec=base_spec,
        output_root=out_root,
        target_coverage=target_coverage,
        max_good_width=float(args.max_good_width),
        inner_epochs=int(args.inner_epochs),
        default_seed=int(args.seed),
        force_advanced=bool(int(args.force_advanced)),
        force_stagewise_warmup=bool(int(args.force_stagewise_warmup)),
        force_gate_piecewise=bool(int(args.force_gate_piecewise)),
        gate_blend_kappa=float(args.gate_blend_kappa),
        path_memory_db_path=str(args.path_memory_db_path),
        path_memory_namespace=str(args.path_memory_namespace),
    )

    solver = EvolutionSolver(
        problem,
        pop_size=max(2, int(args.pop_size)),
        max_generations=max(1, int(args.generations)),
        mutation_rate=0.2,
        crossover_rate=0.85,
        random_seed=int(args.seed),
        enable_progress_log=True,
        report_interval=1,
        max_pareto_solutions=300,
    )

    print("NSGABLACK OUTER SEARCH START")
    print(f"config={args.config}")
    print(f"pop_size={int(args.pop_size)} generations={int(args.generations)} seed={int(args.seed)}")
    print(
        f"inner_epochs={int(args.inner_epochs)} target_coverage={target_coverage:.4f} "
        f"max_good_width={float(args.max_good_width):.4f}"
    )
    print(
        f"force_advanced={bool(int(args.force_advanced))} "
        f"stagewise={bool(int(args.force_stagewise_warmup))} "
        f"gate_piecewise={bool(int(args.force_gate_piecewise))} "
        f"gate_blend_kappa={float(args.gate_blend_kappa):.1f}"
    )
    if str(args.base_params_json).strip():
        print(f"base_params_json={str(args.base_params_json)}")
    if str(args.path_memory_db_path).strip():
        print(
            f"path_memory_db={str(args.path_memory_db_path)} "
            f"namespace={str(args.path_memory_namespace)}"
        )
    print(f"output_root={out_root}")

    t0 = time.perf_counter()
    result = solver.run(return_dict=True)
    elapsed = float(time.perf_counter() - t0)

    all_rows = list(problem.records)
    ok_rows = [r for r in all_rows if r.get("status") == "ok"]

    pareto_rows = _pareto_from_result(problem, result)
    best = _pick_compromise(pareto_rows)

    summary = {
        "timestamp": datetime.now().isoformat(),
        "elapsed_sec": elapsed,
        "target_coverage": target_coverage,
        "max_good_width": float(args.max_good_width),
        "outer": {
            "algorithm": "nsgablack_evolution_solver_nsga2",
            "pop_size": int(args.pop_size),
            "generations": int(args.generations),
            "seed": int(args.seed),
        },
        "inner": {
            "trainer_key": base_spec.train.trainer_key,
            "inner_epochs": int(args.inner_epochs),
            "base_params_json": str(args.base_params_json),
            "force_advanced": bool(int(args.force_advanced)),
            "force_stagewise_warmup": bool(int(args.force_stagewise_warmup)),
            "force_gate_piecewise": bool(int(args.force_gate_piecewise)),
            "gate_blend_kappa": float(args.gate_blend_kappa),
            "path_memory_db_path": str(args.path_memory_db_path),
            "path_memory_namespace": str(args.path_memory_namespace),
        },
        "counts": {
            "eval_total": int(len(all_rows)),
            "eval_ok": int(len(ok_rows)),
            "eval_failed": int(len(all_rows) - len(ok_rows)),
            "eval_good_width": int(sum(1 for r in ok_rows if float(r.get("test_mean_width", np.inf)) <= float(args.max_good_width))),
            "pareto_count": int(len(pareto_rows)),
        },
        "best_compromise": best,
    }

    _write_json(out_root / "summary.json", summary)
    _write_json(out_root / "all_evals.json", all_rows)
    _write_json(out_root / "pareto_front.json", pareto_rows)
    _write_json(out_root / "nsgablack_result.json", result)
    _write_csv(out_root / "all_evals.csv", all_rows)

    if best is not None and isinstance(best.get("record"), dict):
        best_rec = dict(best["record"])
        best_cfg = {
            "data": _to_jsonable(base_spec.data.__dict__),
            "train": {
                **_to_jsonable(base_spec.train.__dict__),
                "trainer_params": _to_jsonable(best_rec.get("params", {})),
                "output_dir": str(best_rec.get("output_dir", "")),
                "run_name": str(best_rec.get("trial_id", "best_trial")),
            },
        }
        _write_json(out_root / "best_config.json", best_cfg)

    print("NSGABLACK OUTER SEARCH DONE")
    print(f"elapsed={elapsed:.1f}s  eval_total={len(all_rows)}  eval_ok={len(ok_rows)}  pareto={len(pareto_rows)}")
    if best is not None and isinstance(best.get("record"), dict):
        r = best["record"]
        print(
            "best_compromise "
            f"trial={r.get('trial_id')}  rmse={float(r.get('test_rmse', float('nan'))):.4f}  "
            f"coverage={float(r.get('test_coverage', float('nan'))):.4f}  "
            f"width={float(r.get('test_mean_width', float('nan'))):.4f}"
        )
    print(f"results_dir={out_root}")


if __name__ == "__main__":
    main()
