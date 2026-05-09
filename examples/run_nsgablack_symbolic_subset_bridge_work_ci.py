from __future__ import annotations

import argparse
import json
import sys
import time
import concurrent.futures
import types
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

from core.common.contracts import ProcessedDataset
from core.models.symbolic_torch_model import SymbolicTorchRegressor
from core.symbolic.expression_graph_cache import ExpressionGraphCache
from core.symbolic.gradient_parser import GradientParser
from core.symbolic.symbolic_dsl import evaluate_genome_numpy
from core.symbolic.symbolic_structure_search import evaluate_genome_with_ridge
from core.trainers.xgboost_trainer import XGBoostSurrogateTrainer, XGBoostTrainerConfig
from examples.path_defaults import default_work_ci_csv
from examples.work_ci_reader import WorkCiIntervalReader
from nsgablack.adapters import (
    MOEADAdapter,
    MOEADConfig,
    NSGA2Adapter,
    NSGA2Config,
    SerialPhaseSpec,
    SerialStrategyConfig,
    StrategyChainAdapter,
    VNSAdapter,
    VNSConfig,
)
from nsgablack.core.base import BlackBoxProblem
from nsgablack.core.composable_solver import ComposableSolver
from nsgablack.representation import RepresentationPipeline
from nsgablack.representation.continuous import ClipRepair, ContextGaussianMutation, UniformInitializer


def _jsonable(v: Any) -> Any:
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, Mapping):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, set)):
        return [_jsonable(x) for x in v]
    return str(v)


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt = np.asarray(y_true, dtype=float).reshape(-1)
    yp = np.asarray(y_pred, dtype=float).reshape(-1)
    return float(np.sqrt(np.mean((yp - yt) ** 2)))


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt = np.asarray(y_true, dtype=float).reshape(-1)
    yp = np.asarray(y_pred, dtype=float).reshape(-1)
    return float(np.mean(np.abs(yp - yt)))


def _as_2d(v: np.ndarray) -> np.ndarray:
    arr = np.asarray(v, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError("array must be 2D")
    return arr


def _build_symmetric_interval(
    *,
    y_train: np.ndarray,
    pred_train: np.ndarray,
    pred_eval: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    yt = _as_2d(np.asarray(y_train, dtype=float)).reshape(-1)
    pt = _as_2d(np.asarray(pred_train, dtype=float)).reshape(-1)
    pe = _as_2d(np.asarray(pred_eval, dtype=float)).reshape(-1)
    a = float(np.clip(alpha, 1e-6, 0.99))
    q = float(np.quantile(np.abs(yt - pt), 1.0 - a))
    lower = pe - q
    upper = pe + q
    return lower, upper, q


def _interval_metrics(
    *,
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    alpha: float,
) -> dict[str, float]:
    y = _as_2d(np.asarray(y_true, dtype=float)).reshape(-1)
    lo = _as_2d(np.asarray(lower, dtype=float)).reshape(-1)
    up = _as_2d(np.asarray(upper, dtype=float)).reshape(-1)
    a = float(np.clip(alpha, 1e-6, 0.99))
    inside = np.logical_and(y >= lo, y <= up)
    picp = float(np.mean(inside))
    width = np.asarray(up - lo, dtype=float)
    y_range = float(np.max(y) - np.min(y))
    pinaw = float(np.mean(width) / max(1e-8, y_range))
    below = np.asarray(lo - y, dtype=float)
    above = np.asarray(y - up, dtype=float)
    interval_score = float(
        np.mean(
            width
            + (2.0 / a) * np.maximum(0.0, below)
            + (2.0 / a) * np.maximum(0.0, above)
        )
    )
    return {
        "picp": float(picp),
        "pinaw": float(pinaw),
        "interval_score": float(interval_score),
        "mean_width": float(np.mean(width)),
        "coverage_target": float(1.0 - a),
    }


def _three_layer_fit_predict(
    *,
    genome: Sequence[Mapping[str, Any]],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    y_eval: np.ndarray | None,
    l2: float,
    inner_opt_enabled: bool,
    inner_opt_adam_steps: int,
    inner_opt_adam_lr: float,
    inner_opt_lbfgs_steps: int,
    inner_opt_lbfgs_lr: float,
    inner_opt_accept_rmse_tol: float,
    inner_opt_accept_rel_tol: float,
    inner_opt_guard_patience: int,
    inner_opt_guard_check_interval: int,
    inner_opt_alt_freeze_readout: bool,
    inner_opt_grad_clip_norm: float,
    inner_opt_residual_clip_q: float,
) -> dict[str, Any]:
    xtr = _as_2d(np.asarray(X_train, dtype=float))
    ytr = _as_2d(np.asarray(y_train, dtype=float))
    xev = _as_2d(np.asarray(X_eval, dtype=float))
    yev = None if y_eval is None else _as_2d(np.asarray(y_eval, dtype=float))

    # Inner-level seed: fast closed-form readout fit (Ridge).
    seed = evaluate_genome_with_ridge(
        genome,
        X_train=xtr,
        y_train=ytr,
        X_eval=xev,
        y_eval=yev,
        l2=float(max(0.0, l2)),
    )
    pred_eval_seed = _as_2d(np.asarray(seed.get("pred_eval"), dtype=float))
    pred_train_seed = _as_2d(np.asarray(seed.get("pred_train"), dtype=float))
    rmse_train_seed = _rmse(ytr, pred_train_seed)

    info: dict[str, Any] = {
        "inner_opt_enabled": bool(inner_opt_enabled),
        "inner_opt_applied": False,
        "status": "ridge_only",
        "rmse_train_before": float(rmse_train_seed),
        "rmse_train_after": float(rmse_train_seed),
        "accept_abs_tol": float(max(0.0, inner_opt_accept_rmse_tol)),
        "accept_rel_tol": float(max(0.0, inner_opt_accept_rel_tol)),
        "guard_patience": int(max(1, inner_opt_guard_patience)),
        "guard_check_interval": int(max(1, inner_opt_guard_check_interval)),
    }
    if not bool(inner_opt_enabled):
        out = dict(seed)
        out["inner_opt_info"] = info
        return out

    try:
        import torch
    except Exception as exc:
        info["status"] = "no_torch"
        info["error"] = f"{type(exc).__name__}: {exc}"
        out = dict(seed)
        out["inner_opt_info"] = info
        return out

    try:
        model = SymbolicTorchRegressor(
            input_dim=int(xtr.shape[1]),
            output_dim=int(ytr.shape[1]),
            genome=genome,
            epsilon=1e-6,
        )
        xtr_t = torch.as_tensor(xtr, dtype=torch.float32)
        ytr_t = torch.as_tensor(ytr, dtype=torch.float32)
        xev_t = torch.as_tensor(xev, dtype=torch.float32)
        l2v = float(max(0.0, l2))

        # Pre-condition: normalize basis on train split to reduce ill-conditioning.
        with torch.no_grad():
            phi_seed = model.basis(xtr_t)
            phi_mu = torch.mean(phi_seed, dim=0, keepdim=True)
            phi_std = torch.std(phi_seed, dim=0, keepdim=True).clamp_min(1e-6)
            w_seed_raw = torch.as_tensor(_as_2d(np.asarray(seed.get("weight"), dtype=float)), dtype=torch.float32)
            b_seed_raw = torch.as_tensor(_as_2d(np.asarray(seed.get("bias"), dtype=float)).reshape(-1), dtype=torch.float32)
            w_norm0 = phi_std.T * w_seed_raw
            b_norm0 = b_seed_raw + torch.sum(phi_mu * w_seed_raw.T, dim=1)

        w_norm = torch.nn.Parameter(w_norm0.clone())
        b_norm = torch.nn.Parameter(b_norm0.clone())

        tol_abs = float(max(0.0, inner_opt_accept_rmse_tol))
        tol_rel = float(max(0.0, inner_opt_accept_rel_tol))
        guard_patience = int(max(1, inner_opt_guard_patience))
        guard_check_interval = int(max(1, inner_opt_guard_check_interval))
        soft_upper = float(rmse_train_seed + tol_abs + tol_rel * max(1e-8, rmse_train_seed))

        def _capture_state() -> dict[str, Any]:
            return {
                "param_table": {k: v.detach().cpu().clone() for k, v in model.param_table.state_dict().items()},
                "w_norm": w_norm.detach().cpu().clone(),
                "b_norm": b_norm.detach().cpu().clone(),
            }

        def _restore_state(state: Mapping[str, Any]) -> None:
            with torch.no_grad():
                model.param_table.load_state_dict(dict(state["param_table"]), strict=True)
                w_norm.copy_(state["w_norm"].to(dtype=w_norm.dtype, device=w_norm.device))
                b_norm.copy_(state["b_norm"].to(dtype=b_norm.dtype, device=b_norm.device))

        best_rmse = float(rmse_train_seed)
        best_state = _capture_state()
        guard_bad_streak = 0
        guard_checks = 0
        guard_rollback_triggered = False

        def _guard_check(*, force: bool = False) -> bool:
            nonlocal best_rmse, best_state, guard_bad_streak, guard_checks, guard_rollback_triggered
            if (not force) and guard_check_interval > 1:
                return True
            with torch.no_grad():
                pred_now = _as_2d(np.asarray(predict_with_norm(xtr_t).cpu().numpy(), dtype=float))
            rm_now = float(_rmse(ytr, pred_now))
            guard_checks += 1
            if rm_now < best_rmse:
                best_rmse = float(rm_now)
                best_state = _capture_state()
                guard_bad_streak = 0
                return True
            if rm_now > soft_upper:
                guard_bad_streak += 1
            else:
                guard_bad_streak = 0
            if guard_bad_streak >= guard_patience:
                guard_rollback_triggered = True
                _restore_state(best_state)
                return False
            return True

        abs_res_seed = np.abs(np.asarray(ytr - pred_train_seed, dtype=float).reshape(-1))
        q = float(np.clip(inner_opt_residual_clip_q, 0.70, 0.999))
        res_clip = float(np.quantile(abs_res_seed, q)) if abs_res_seed.size > 0 else 1.0
        res_clip = float(max(1e-6, res_clip))

        def predict_with_norm(Xt: Any) -> Any:
            phi = model.basis(Xt)
            phi_n = (phi - phi_mu) / phi_std
            return phi_n @ w_norm + b_norm.reshape(1, -1)

        def objective() -> Any:
            pred_t = predict_with_norm(xtr_t)
            res = pred_t - ytr_t
            res = torch.clamp(res, min=-res_clip, max=res_clip)
            mse = torch.mean(res * res)
            reg = l2v * torch.mean((w_norm / phi_std.T) ** 2)
            return mse + reg

        grad_clip = float(max(0.0, inner_opt_grad_clip_norm))
        alt_mode = bool(inner_opt_alt_freeze_readout)

        # Phase A: optimize internal symbolic parameters only, freeze readout.
        if alt_mode:
            w_norm.requires_grad_(False)
            b_norm.requires_grad_(False)
            for p in model.readout.parameters():
                p.requires_grad_(False)
            inner_params = [p for p in model.param_table.parameters() if p.requires_grad]
            adam_steps = int(max(0, inner_opt_adam_steps))
            if inner_params and adam_steps > 0:
                opt_adam_inner = torch.optim.Adam(inner_params, lr=float(max(1e-6, inner_opt_adam_lr)), weight_decay=0.0)
                for step in range(adam_steps):
                    opt_adam_inner.zero_grad(set_to_none=True)
                    loss_t = objective()
                    loss_t.backward()
                    if grad_clip > 0.0:
                        torch.nn.utils.clip_grad_norm_(inner_params, max_norm=grad_clip)
                    opt_adam_inner.step()
                    if ((step + 1) % guard_check_interval == 0 or (step + 1) == adam_steps) and not _guard_check(force=True):
                        break
            w_norm.requires_grad_(True)
            b_norm.requires_grad_(True)
        else:
            all_params = list(model.parameters()) + [w_norm, b_norm]
            adam_steps = int(max(0, inner_opt_adam_steps))
            if adam_steps > 0:
                opt_adam = torch.optim.Adam(all_params, lr=float(max(1e-6, inner_opt_adam_lr)), weight_decay=0.0)
                for step in range(adam_steps):
                    opt_adam.zero_grad(set_to_none=True)
                    loss_t = objective()
                    loss_t.backward()
                    if grad_clip > 0.0:
                        torch.nn.utils.clip_grad_norm_(all_params, max_norm=grad_clip)
                    opt_adam.step()
                    if ((step + 1) % guard_check_interval == 0 or (step + 1) == adam_steps) and not _guard_check(force=True):
                        break

        # Phase B: optimize readout in normalized coordinate (keep internals fixed in alt mode).
        lbfgs_steps = int(max(0, inner_opt_lbfgs_steps))
        if lbfgs_steps > 0 and not guard_rollback_triggered:
            if alt_mode:
                for p in model.param_table.parameters():
                    p.requires_grad_(False)
                params_lbfgs = [w_norm, b_norm]
            else:
                params_lbfgs = [p for p in (list(model.param_table.parameters()) + [w_norm, b_norm]) if p.requires_grad]

            opt_lbfgs = torch.optim.LBFGS(
                params_lbfgs,
                lr=float(max(1e-6, inner_opt_lbfgs_lr)),
                max_iter=1,
                history_size=20,
                line_search_fn="strong_wolfe",
            )

            def closure() -> Any:
                opt_lbfgs.zero_grad(set_to_none=True)
                loss_t = objective()
                loss_t.backward()
                if grad_clip > 0.0:
                    torch.nn.utils.clip_grad_norm_(params_lbfgs, max_norm=grad_clip)
                return loss_t

            for step in range(lbfgs_steps):
                opt_lbfgs.step(closure)
                if ((step + 1) % guard_check_interval == 0 or (step + 1) == lbfgs_steps) and not _guard_check(force=True):
                    break

        with torch.no_grad():
            pred_train_opt = _as_2d(np.asarray(predict_with_norm(xtr_t).cpu().numpy(), dtype=float))
            pred_eval_opt = _as_2d(np.asarray(predict_with_norm(xev_t).cpu().numpy(), dtype=float))
            w_raw = np.asarray((w_norm / phi_std.T).detach().cpu().numpy(), dtype=float)
            b_raw = np.asarray((b_norm - torch.sum(phi_mu * (w_norm / phi_std.T).T, dim=1)).detach().cpu().numpy(), dtype=float)

        rmse_train_opt = _rmse(ytr, pred_train_opt)
        strict_upper = float(rmse_train_seed + tol_abs)
        accepted = bool(rmse_train_opt <= soft_upper)

        if not accepted:
            info["status"] = "rejected_by_train_guard"
            info["rmse_train_after"] = float(rmse_train_opt)
            info["guard_bad_streak"] = int(guard_bad_streak)
            info["guard_checks"] = int(guard_checks)
            info["guard_rollback_triggered"] = bool(guard_rollback_triggered)
            info["strict_upper"] = float(strict_upper)
            info["soft_upper"] = float(soft_upper)
            info["best_seen_rmse"] = float(best_rmse)
            out = dict(seed)
            out["inner_opt_info"] = info
            return out

        accept_tier = "strict" if rmse_train_opt <= strict_upper else "soft"
        status = "applied" if accept_tier == "strict" else "applied_soft_degrade"
        if guard_rollback_triggered and accept_tier == "soft":
            status = "applied_soft_after_guard_rollback"
        elif guard_rollback_triggered and accept_tier == "strict":
            status = "applied_after_guard_rollback"

        out = {
            "weight": np.asarray(w_raw, dtype=float),
            "bias": _as_2d(np.asarray(b_raw, dtype=float)).reshape(-1),
            "pred_train": np.asarray(pred_train_opt, dtype=float),
            "pred_eval": np.asarray(pred_eval_opt, dtype=float),
            "metrics_train": {"rmse": float(rmse_train_opt)},
            "metrics_eval": {"rmse": float(_rmse(yev, pred_eval_opt)) if yev is not None else float("nan")},
            "inner_opt_info": {
                **info,
                "status": status,
                "inner_opt_applied": True,
                "rmse_train_after": float(rmse_train_opt),
                "accept_tier": str(accept_tier),
                "strict_upper": float(strict_upper),
                "soft_upper": float(soft_upper),
                "guard_bad_streak": int(guard_bad_streak),
                "guard_checks": int(guard_checks),
                "guard_rollback_triggered": bool(guard_rollback_triggered),
                "best_seen_rmse": float(best_rmse),
                "precondition": {"residual_clip_q": float(q), "residual_clip": float(res_clip)},
                "alt_mode_freeze_readout": bool(alt_mode),
                "grad_clip_norm": float(grad_clip),
                "parameter_values": _jsonable(model.export_parameter_values()),
            },
        }
        return out
    except Exception as exc:
        info["status"] = "inner_opt_failed"
        info["error"] = f"{type(exc).__name__}: {exc}"
        out = dict(seed)
        out["inner_opt_info"] = info
        return out


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
        return [(np.arange(0, split, dtype=int), np.arange(split, nn, dtype=int))]

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


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=float).reshape(-1)
    y = np.asarray(b, dtype=float).reshape(-1)
    xc = x - float(np.mean(x))
    yc = y - float(np.mean(y))
    denom = float(np.sqrt(np.dot(xc, xc) * np.dot(yc, yc))) + 1e-12
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(xc, yc) / denom)


def _is_novel_enough(
    z: np.ndarray,
    accepted_z: Sequence[np.ndarray],
    *,
    max_abs_corr: float,
) -> bool:
    zz = np.asarray(z, dtype=float).reshape(-1)
    th = float(np.clip(max_abs_corr, 0.0, 0.999999))
    for prev in accepted_z:
        c = abs(_safe_corr(zz, np.asarray(prev, dtype=float).reshape(-1)))
        if c >= th:
            return False
    return True


def _counterfactual_sensitivity(
    *,
    expr: Mapping[str, Any],
    X: np.ndarray,
    feats: Sequence[int],
    z_ref: np.ndarray,
    noise_scale: float,
) -> float:
    x = np.asarray(X, dtype=float)
    z0 = np.asarray(z_ref, dtype=float).reshape(-1)
    if x.ndim != 2 or z0.size != int(x.shape[0]):
        return float("inf")
    idx = sorted({int(v) for v in feats if 0 <= int(v) < int(x.shape[1])})
    if not idx:
        return 0.0
    rng = np.random.default_rng(12345)
    xp = np.asarray(x, dtype=float).copy()
    scale = float(max(1e-6, noise_scale))
    for j in idx:
        col = np.asarray(x[:, j], dtype=float)
        sigma = float(np.std(col))
        if not np.isfinite(sigma) or sigma <= 1e-12:
            continue
        xp[:, j] = col + rng.normal(0.0, sigma * scale, size=col.shape[0])
    try:
        z1 = np.asarray(evaluate_genome_numpy([{"name": "tmp", "expr": dict(expr)}], xp), dtype=float).reshape(-1)
    except Exception:
        return float("inf")
    denom = float(np.std(z0)) + 1e-8
    sens = float(np.mean(np.abs(z1 - z0)) / denom)
    if not np.isfinite(sens):
        return float("inf")
    return sens


def _parse_csv_list(raw: str) -> list[str]:
    return [str(v).strip() for v in str(raw).split(",") if str(v).strip()]


def _parse_csv_floats(raw: str) -> list[float]:
    out: list[float] = []
    for tok in _parse_csv_list(raw):
        try:
            out.append(float(tok))
        except Exception:
            continue
    return out


def _allocate_phase_steps(total_steps: int, n_phases: int, weights: Sequence[float]) -> list[int]:
    total = int(max(1, total_steps))
    n = int(max(1, n_phases))
    w = np.asarray([max(0.0, float(v)) for v in list(weights)[:n]], dtype=float)
    if w.size < n:
        w = np.concatenate([w, np.ones(n - int(w.size), dtype=float)], axis=0)
    if float(np.sum(w)) <= 0:
        w = np.ones(n, dtype=float)
    w = w / float(np.sum(w))

    steps = [max(1, int(round(total * float(wi)))) for wi in w]
    while sum(steps) > total and any(v > 1 for v in steps):
        idx = int(np.argmax(np.asarray(steps, dtype=int)))
        if steps[idx] > 1:
            steps[idx] -= 1
        else:
            break
    while sum(steps) < total:
        idx = int(np.argmin(np.asarray(steps, dtype=int)))
        steps[idx] += 1
    return [int(v) for v in steps]


def _build_outer_adapter(
    *,
    strategy: str,
    pop_size: int,
    generations: int,
    portfolio_phases_csv: str,
    portfolio_weights_csv: str,
    moead_neighborhood_size: int,
    moead_delta: float,
    moead_nr: int,
    vns_k_max: int,
    vns_batch_size: int,
) -> tuple[Any, dict[str, Any]]:
    mode = str(strategy).strip().lower()
    pop = int(max(4, pop_size))
    gens = int(max(1, generations))

    if mode == "nsga2":
        adapter = NSGA2Adapter(
            config=NSGA2Config(
                population_size=int(pop),
                offspring_size=int(pop),
                crossover_rate=0.90,
                objective_aggregation="sum",
            )
        )
        return adapter, {"strategy": "nsga2", "max_generations": int(gens)}

    if mode == "moead":
        adapter = MOEADAdapter(
            config=MOEADConfig(
                population_size=int(pop),
                neighborhood_size=int(max(2, moead_neighborhood_size)),
                batch_size=int(pop),
                delta=float(np.clip(moead_delta, 0.05, 1.0)),
                nr=int(max(1, moead_nr)),
                decomposition="tchebycheff",
            )
        )
        return adapter, {"strategy": "moead", "max_generations": int(gens)}

    if mode == "vns":
        adapter = VNSAdapter(
            config=VNSConfig(
                batch_size=int(max(4, vns_batch_size)),
                k_max=int(max(1, vns_k_max)),
                base_sigma=0.15,
                scale=1.45,
                max_sigma=2.0,
                objective_aggregation="sum",
            )
        )
        return adapter, {"strategy": "vns", "max_generations": int(gens)}

    phase_names = [v.lower() for v in _parse_csv_list(portfolio_phases_csv)]
    if not phase_names:
        phase_names = ["nsga2", "moead", "vns"]
    phase_weights = _parse_csv_floats(portfolio_weights_csv)
    if not phase_weights:
        phase_weights = [2.0, 1.0, 1.0]
    phase_steps = _allocate_phase_steps(gens, len(phase_names), phase_weights)

    phases: list[SerialPhaseSpec] = []
    phase_meta: list[dict[str, Any]] = []
    for i, (nm, st) in enumerate(zip(phase_names, phase_steps)):
        name = str(nm).strip().lower()
        if name == "nsga2":
            adapter_i = NSGA2Adapter(
                config=NSGA2Config(
                    population_size=int(pop),
                    offspring_size=int(pop),
                    crossover_rate=0.90,
                    objective_aggregation="sum",
                ),
                name=f"nsga2_phase_{i}",
            )
        elif name == "moead":
            adapter_i = MOEADAdapter(
                config=MOEADConfig(
                    population_size=int(pop),
                    neighborhood_size=int(max(2, moead_neighborhood_size)),
                    batch_size=int(pop),
                    delta=float(np.clip(moead_delta, 0.05, 1.0)),
                    nr=int(max(1, moead_nr)),
                    decomposition="tchebycheff",
                ),
                name=f"moead_phase_{i}",
            )
        elif name == "vns":
            adapter_i = VNSAdapter(
                config=VNSConfig(
                    batch_size=int(max(4, vns_batch_size)),
                    k_max=int(max(1, vns_k_max)),
                    base_sigma=0.15,
                    scale=1.45,
                    max_sigma=2.0,
                    objective_aggregation="sum",
                ),
                name=f"vns_phase_{i}",
            )
        else:
            continue
        phases.append(SerialPhaseSpec(name=name, adapter=adapter_i, steps=int(max(1, st))))
        phase_meta.append({"name": str(name), "steps": int(max(1, st))})

    if not phases:
        phases = [
            SerialPhaseSpec(
                name="nsga2",
                adapter=NSGA2Adapter(
                    config=NSGA2Config(
                        population_size=int(pop),
                        offspring_size=int(pop),
                        crossover_rate=0.90,
                        objective_aggregation="sum",
                    )
                ),
                steps=int(gens),
            )
        ]
        phase_meta = [{"name": "nsga2", "steps": int(gens)}]

    adapter = StrategyChainAdapter(
        phases=phases,
        config=SerialStrategyConfig(repeat_last=False),
        name="portfolio_serial_chain",
    )
    return adapter, {
        "strategy": "portfolio",
        "max_generations": int(sum(int(v["steps"]) for v in phase_meta)),
        "portfolio_phases": phase_meta,
    }


def _bounds_arrays(problem: BlackBoxProblem) -> tuple[np.ndarray, np.ndarray]:
    b = getattr(problem, "bounds", None)
    d = int(getattr(problem, "dimension", 0))
    if isinstance(b, dict):
        keys = list(getattr(problem, "variables", []))
        if len(keys) != d or any(k not in b for k in keys):
            keys = list(b.keys())
        pairs = [b[k] for k in keys]
    else:
        pairs = list(b or [])
    if len(pairs) != d:
        low = np.full(d, -1.0, dtype=float)
        high = np.full(d, 1.0, dtype=float)
        return low, high
    low = np.asarray([float(p[0]) for p in pairs], dtype=float)
    high = np.asarray([float(p[1]) for p in pairs], dtype=float)
    lo = np.minimum(low, high)
    hi = np.maximum(low, high)
    return lo, hi


STRICT4_REGIME_ORDER: tuple[tuple[int, int, int, int], ...] = (
    (1, 1, 0, 0),  # holiday_near
    (1, 0, 1, 0),  # holiday_mid
    (0, 0, 0, 1),  # weekend
    (0, 0, 0, 0),  # regular
)


def _normalize_fixed4_key(raw_key: tuple[int, ...]) -> tuple[int, int, int, int]:
    if len(raw_key) >= 4:
        return tuple(int(v > 0) for v in raw_key[:4])  # type: ignore[return-value]
    padded = list(int(v > 0) for v in raw_key)
    while len(padded) < 4:
        padded.append(0)
    return tuple(padded[:4])  # type: ignore[return-value]


def _map_to_strict4_regime(raw_key: tuple[int, ...]) -> tuple[int, int, int, int]:
    k = _normalize_fixed4_key(raw_key)
    if k in STRICT4_REGIME_ORDER:
        return k
    if k[0] > 0 and k[1] > 0:
        return (1, 1, 0, 0)
    if k[0] > 0 and k[2] > 0:
        return (1, 0, 1, 0)
    if k[3] > 0:
        return (0, 0, 0, 1)
    if k[0] == 0 and k[1] == 0 and k[2] == 0:
        return (0, 0, 0, 0)
    return (0, 0, 0, 0)


def _strict4_keys_from_X(X: np.ndarray, gate_idx: tuple[int, int, int, int]) -> tuple[tuple[int, int, int, int], ...]:
    x = np.asarray(X, dtype=float)
    i0, i1, i2, i3 = gate_idx
    out: list[tuple[int, int, int, int]] = []
    for r in range(int(x.shape[0])):
        raw = (
            int(x[r, i0] > 0.5),
            int(x[r, i1] > 0.5),
            int(x[r, i2] > 0.5),
            int(x[r, i3] > 0.5),
        )
        out.append(_map_to_strict4_regime(raw))
    return tuple(out)


@dataclass(frozen=True)
class CandidateTerm:
    name: str
    expr: dict[str, Any]
    complexity: float
    family: str
    features: tuple[int, ...]
    prior_corr: float


def _feature_expr(j: int) -> dict[str, Any]:
    return {"type": "feature", "index": int(j)}


def _const_expr(v: float) -> dict[str, Any]:
    return {"type": "const", "value": float(v)}


def _unary_expr(op: str, arg: Mapping[str, Any]) -> dict[str, Any]:
    return {"type": "unary", "op": str(op), "arg": dict(arg)}


def _binary_expr(op: str, left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    return {"type": "binary", "op": str(op), "left": dict(left), "right": dict(right)}


def _relu_expr(arg: Mapping[str, Any]) -> dict[str, Any]:
    # relu(z) = 0.5 * (z + abs(z)) using existing DSL ops.
    z = dict(arg)
    return _binary_expr("mul", _const_expr(0.5), _binary_expr("add", z, _unary_expr("abs", z)))


def _soft_step_expr(feature_idx: int, threshold: float, steepness: float) -> dict[str, Any]:
    # soft-step(x>c) ~= 0.5 * (1 + tanh(k*(x-c)))
    z = _binary_expr("sub", _feature_expr(feature_idx), _const_expr(float(threshold)))
    kz = _binary_expr("mul", _const_expr(float(steepness)), z)
    t = _unary_expr("tanh", kz)
    return _binary_expr("mul", _const_expr(0.5), _binary_expr("add", _const_expr(1.0), t))


def _estimate_gradient_change_points(
    x_col: np.ndarray,
    residual: np.ndarray,
    *,
    min_bin: int,
    topk: int = 2,
) -> list[tuple[float, float]]:
    x = np.asarray(x_col, dtype=float).reshape(-1)
    r = np.asarray(residual, dtype=float).reshape(-1)
    n = int(x.size)
    if n < int(max(8, 2 * min_bin)):
        return []
    order = np.argsort(x)
    xs = x[order]
    rs = r[order]
    ps = np.concatenate(([0.0], np.cumsum(rs)))
    candidate: list[tuple[float, int]] = []
    step = max(1, n // 128)
    for t in range(int(min_bin), int(n - min_bin), int(step)):
        ln = float(t)
        rn = float(n - t)
        lm = float((ps[t] - ps[0]) / max(1.0, ln))
        rm = float((ps[n] - ps[t]) / max(1.0, rn))
        # weighted jump score on residual means
        score = float(abs(lm - rm) * np.sqrt((ln * rn) / max(1.0, float(n))))
        candidate.append((score, t))
    if not candidate:
        return []
    candidate.sort(key=lambda kv: kv[0], reverse=True)
    out: list[tuple[float, float]] = []
    used_pos: list[int] = []
    min_sep = max(4, n // 20)
    for score, t in candidate:
        if len(out) >= int(max(1, topk)):
            break
        if any(abs(int(t) - int(u)) < int(min_sep) for u in used_pos):
            continue
        if t <= 0 or t >= n:
            continue
        c = float(0.5 * (xs[t - 1] + xs[t]))
        if not np.isfinite(c):
            continue
        out.append((c, float(score)))
        used_pos.append(int(t))
    return out


def _build_candidate_pool(
    X: np.ndarray,
    y: np.ndarray,
    *,
    feature_names: Sequence[str],
    topk_for_pairs: int = 6,
    include_pair_interactions: bool = True,
    include_gradient_enrich: bool = True,
) -> list[CandidateTerm]:
    x = np.asarray(X, dtype=float)
    yt = np.asarray(y, dtype=float).reshape(-1)
    d = int(x.shape[1])

    pool: list[CandidateTerm] = []
    for j in range(d):
        base = _feature_expr(j)
        z0 = np.asarray(x[:, j], dtype=float)
        pool.append(
            CandidateTerm(
                name=f"x{j}:{feature_names[j]}",
                expr=base,
                complexity=1.0,
                family="linear",
                features=(int(j),),
                prior_corr=float(abs(_safe_corr(z0, yt))),
            )
        )
        for op in ("square", "sin", "cos", "tanh"):
            if op == "square":
                z = np.asarray(z0 * z0, dtype=float)
            elif op == "sin":
                z = np.asarray(np.sin(z0), dtype=float)
            elif op == "cos":
                z = np.asarray(np.cos(z0), dtype=float)
            else:
                z = np.asarray(np.tanh(z0), dtype=float)
            pool.append(
                CandidateTerm(
                    name=f"{op}(x{j})",
                    expr=_unary_expr(op, base),
                    complexity=2.0,
                    family=f"unary_{op}",
                    features=(int(j),),
                    prior_corr=float(abs(_safe_corr(z, yt))),
                )
            )

    corr = np.asarray([abs(_safe_corr(x[:, j], yt)) for j in range(d)], dtype=float)
    top_idx = list(np.argsort(-corr)[: min(int(max(2, topk_for_pairs)), d)])
    if bool(include_pair_interactions):
        for i in range(len(top_idx)):
            for j in range(i + 1, len(top_idx)):
                a = int(top_idx[i])
                b = int(top_idx[j])
                z = np.asarray(x[:, a] * x[:, b], dtype=float)
                pool.append(
                    CandidateTerm(
                        name=f"x{a}*x{b}",
                        expr=_binary_expr("mul", _feature_expr(a), _feature_expr(b)),
                        complexity=3.0,
                        family="interaction",
                        features=(int(a), int(b)),
                        prior_corr=float(abs(_safe_corr(z, yt))),
                    )
                )

    # Gradient-guided interaction enrichment (function-space guidance, not constant tuning).
    if bool(include_gradient_enrich):
        try:
            seed_genome = [{"name": f"x{j}", "expr": _feature_expr(j)} for j in range(d)]
            fit_seed = evaluate_genome_with_ridge(
                seed_genome,
                X_train=x,
                y_train=yt.reshape(-1, 1),
                X_eval=x,
                y_eval=yt.reshape(-1, 1),
                l2=1e-5,
            )
            gs = GradientParser.build_signal(
                genome=seed_genome,
                weight=np.asarray(fit_seed.get("weight"), dtype=float),
                X=x,
                y=yt.reshape(-1, 1),
                slope_mode="binned_median",
                slope_bins=24,
                slope_min_bin_samples=12,
            )
            cross = np.asarray(getattr(gs, "cross_feature_priority", np.zeros((d, d), dtype=float)), dtype=float)
            p = np.asarray(getattr(gs, "feature_priority", np.zeros((d,), dtype=float)), dtype=float).reshape(-1)
            if p.size == d and cross.shape == (d, d):
                focus_idx = [int(i) for i in np.argsort(-p)[: min(4, d)]]
                existing_keys = {json.dumps(c.expr, sort_keys=True) for c in pool}
                for i in focus_idx:
                    partner_idx = [int(j) for j in np.argsort(-cross[i, :]).tolist() if int(j) != int(i)]
                    for j in partner_idx[:3]:
                        ex1 = _binary_expr("mul", _feature_expr(i), _feature_expr(j))
                        k1 = json.dumps(ex1, sort_keys=True)
                        if k1 not in existing_keys:
                            z1 = np.asarray(x[:, i] * x[:, j], dtype=float)
                            pool.append(
                                CandidateTerm(
                                    name=f"grad:x{i}*x{j}",
                                    expr=ex1,
                                    complexity=3.0,
                                    family="interaction_grad",
                                    features=(int(i), int(j)),
                                    prior_corr=float(abs(_safe_corr(z1, yt))),
                                )
                            )
                            existing_keys.add(k1)
                        ex2 = _binary_expr("mul", _unary_expr("tanh", _feature_expr(i)), _feature_expr(j))
                        k2 = json.dumps(ex2, sort_keys=True)
                        if k2 not in existing_keys:
                            z2 = np.asarray(np.tanh(x[:, i]) * x[:, j], dtype=float)
                            pool.append(
                                CandidateTerm(
                                    name=f"grad:tanh(x{i})*x{j}",
                                    expr=ex2,
                                    complexity=4.0,
                                    family="interaction_grad",
                                    features=(int(i), int(j)),
                                    prior_corr=float(abs(_safe_corr(z2, yt))),
                                )
                            )
                            existing_keys.add(k2)
        except Exception:
            pass
    return pool


def _is_gate_feature_name(name: str) -> bool:
    nm = str(name).lower()
    return nm.startswith("is_") or ("holiday" in nm) or ("weekend" in nm) or ("bad_weather" in nm) or ("aqi_high" in nm)


def _expand_candidate_pool_from_residual(
    *,
    X: np.ndarray,
    y_residual: np.ndarray,
    feature_names: Sequence[str],
    base_genome: Sequence[Mapping[str, Any]],
    base_weight: np.ndarray,
    existing: Sequence[CandidateTerm],
    max_new_terms: int,
    focus_top_features: int,
    partner_topk: int,
    enable_piecewise: bool = True,
    enable_dynamic_interactions: bool = True,
    enable_gate_interactions: bool = True,
    info_gain_enabled: bool = True,
    info_gain_min_abs_corr: float = 0.02,
    novelty_enabled: bool = True,
    novelty_max_abs_corr: float = 0.985,
    counterfactual_enabled: bool = True,
    counterfactual_noise_scale: float = 0.05,
    counterfactual_max_sensitivity: float = 0.50,
) -> list[CandidateTerm]:
    x = np.asarray(X, dtype=float)
    r = np.asarray(y_residual, dtype=float).reshape(-1)
    d = int(x.shape[1])
    new_terms: list[CandidateTerm] = []
    existing_keys = {json.dumps(c.expr, sort_keys=True) for c in existing}

    gate_idx = [int(i) for i, nm in enumerate(feature_names) if _is_gate_feature_name(str(nm))]
    gate_set = set(gate_idx)
    name_to_idx = {str(nm): int(i) for i, nm in enumerate(feature_names)}

    try:
        gs = GradientParser.build_signal(
            genome=base_genome,
            weight=np.asarray(base_weight, dtype=float),
            X=x,
            y=r.reshape(-1, 1),
            slope_mode="binned_median",
            slope_bins=24,
            slope_min_bin_samples=12,
        )
        cross = np.asarray(getattr(gs, "cross_feature_priority", np.zeros((d, d), dtype=float)), dtype=float)
        p = np.asarray(getattr(gs, "feature_priority", np.zeros((d,), dtype=float)), dtype=float).reshape(-1)
    except Exception:
        cross = np.zeros((d, d), dtype=float)
        p = np.asarray([abs(_safe_corr(x[:, j], r)) for j in range(d)], dtype=float)

    if p.size != d:
        p = np.asarray([abs(_safe_corr(x[:, j], r)) for j in range(d)], dtype=float)
    if cross.shape != (d, d):
        cross = np.zeros((d, d), dtype=float)

    focus_idx = [int(i) for i in np.argsort(-np.abs(p))[: max(2, min(int(focus_top_features), d))]]
    budget = int(max(1, max_new_terms))
    accepted_z: list[np.ndarray] = []
    gain_min = float(max(0.0, info_gain_min_abs_corr))
    nov_max = float(np.clip(novelty_max_abs_corr, 0.0, 0.999999))
    cf_noise = float(max(1e-6, counterfactual_noise_scale))
    cf_sens_max = float(max(0.0, counterfactual_max_sensitivity))

    def _try_add(name: str, expr: Mapping[str, Any], complexity: float, family: str, feats: Sequence[int], z: np.ndarray) -> None:
        nonlocal budget
        if budget <= 0:
            return
        z_arr = np.asarray(z, dtype=float).reshape(-1)
        if bool(info_gain_enabled):
            gain = float(abs(_safe_corr(z_arr, r)))
            if gain < gain_min:
                return
        if bool(novelty_enabled):
            if not _is_novel_enough(z_arr, accepted_z, max_abs_corr=nov_max):
                return
        if bool(counterfactual_enabled):
            sens = _counterfactual_sensitivity(
                expr=expr,
                X=x,
                feats=feats,
                z_ref=z_arr,
                noise_scale=cf_noise,
            )
            if sens > cf_sens_max:
                return
        key = json.dumps(expr, sort_keys=True)
        if key in existing_keys:
            return
        existing_keys.add(key)
        accepted_z.append(z_arr)
        budget -= 1
        new_terms.append(
            CandidateTerm(
                name=str(name),
                expr=dict(expr),
                complexity=float(complexity),
                family=str(family),
                features=tuple(int(v) for v in feats),
                prior_corr=float(abs(_safe_corr(z_arr, r))),
            )
        )

    # 0) gradient change-point driven hinge / gate atoms on continuous features
    cont_focus_idx = [int(i) for i in focus_idx if int(i) not in gate_set]
    if bool(enable_piecewise):
        for i in cont_focus_idx:
            cps = _estimate_gradient_change_points(
                x_col=x[:, i],
                residual=r,
                min_bin=max(24, int(0.06 * x.shape[0])),
                topk=2,
            )
            if not cps:
                continue
            xi = np.asarray(x[:, i], dtype=float)
            spread = float(np.quantile(xi, 0.9) - np.quantile(xi, 0.1))
            k = float(4.0 / max(1e-6, spread))
            for c, _score in cps:
                z_shift = np.asarray(xi - float(c), dtype=float)
                ex_h_pos = _relu_expr(_binary_expr("sub", _feature_expr(i), _const_expr(float(c))))
                z_h_pos = np.asarray(np.maximum(0.0, z_shift), dtype=float)
                _try_add(f"hinge+:x{i}-({c:.4g})", ex_h_pos, 3.5, "gate_hinge_grad", (i,), z_h_pos)
                if budget <= 0:
                    break

                ex_h_neg = _relu_expr(_binary_expr("sub", _const_expr(float(c)), _feature_expr(i)))
                z_h_neg = np.asarray(np.maximum(0.0, -z_shift), dtype=float)
                _try_add(f"hinge-:({c:.4g})-x{i}", ex_h_neg, 3.5, "gate_hinge_grad", (i,), z_h_neg)
                if budget <= 0:
                    break

                ex_h_pos_self = _binary_expr("mul", ex_h_pos, _feature_expr(i))
                z_h_pos_self = np.asarray(z_h_pos * xi, dtype=float)
                _try_add(f"hinge+:x{i}-({c:.4g})*x{i}", ex_h_pos_self, 4.5, "interaction_hinge_grad", (i, i), z_h_pos_self)
                if budget <= 0:
                    break

                ex_h_neg_self = _binary_expr("mul", ex_h_neg, _feature_expr(i))
                z_h_neg_self = np.asarray(z_h_neg * xi, dtype=float)
                _try_add(f"hinge-:({c:.4g})-x{i}*x{i}", ex_h_neg_self, 4.5, "interaction_hinge_grad", (i, i), z_h_neg_self)
                if budget <= 0:
                    break

                ex_step = _soft_step_expr(feature_idx=i, threshold=float(c), steepness=float(k))
                z_step = np.asarray(0.5 * (1.0 + np.tanh(k * z_shift)), dtype=float)
                _try_add(f"soft_step:x{i}>{c:.4g}", ex_step, 4.0, "gate_step_grad", (i,), z_step)
                if budget <= 0:
                    break

                if not bool(enable_gate_interactions):
                    continue
                partner = [int(j) for j in np.argsort(-np.abs(cross[i, :])).tolist() if int(j) != int(i)]
                fi_name = str(feature_names[i]) if int(i) < len(feature_names) else ""
                preferred: list[int] = []
                if fi_name.startswith("ci_lag"):
                    lag_suffix = fi_name.replace("ci_", "")
                    for base in ("avg_speed_", "avg_occ_", "total_flow_"):
                        pj = name_to_idx.get(f"{base}{lag_suffix}")
                        if pj is not None and int(pj) != int(i):
                            preferred.append(int(pj))
                elif "_lag" in fi_name:
                    lag_suffix = fi_name.split("_")[-1]
                    for base in ("ci", "avg_speed", "avg_occ", "total_flow"):
                        key = f"{base}_{lag_suffix}" if base != "ci" else f"ci_{lag_suffix}"
                        pj = name_to_idx.get(key)
                        if pj is not None and int(pj) != int(i):
                            preferred.append(int(pj))
                if preferred:
                    seen = set()
                    merged = []
                    for j in preferred + partner:
                        if int(j) not in seen:
                            seen.add(int(j))
                            merged.append(int(j))
                    partner = merged
                if partner:
                    j = int(partner[0])
                    ex_h_pos_mul = _binary_expr("mul", ex_h_pos, _feature_expr(j))
                    z_h_pos_mul = np.asarray(z_h_pos * x[:, j], dtype=float)
                    _try_add(f"hinge+(x{i}>{c:.4g})*x{j}", ex_h_pos_mul, 5.0, "interaction_hinge_grad", (i, j), z_h_pos_mul)
                    if budget <= 0:
                        break
                    ex_h_neg_mul = _binary_expr("mul", ex_h_neg, _feature_expr(j))
                    z_h_neg_mul = np.asarray(z_h_neg * x[:, j], dtype=float)
                    _try_add(f"hinge-(x{i}<{c:.4g})*x{j}", ex_h_neg_mul, 5.0, "interaction_hinge_grad", (i, j), z_h_neg_mul)
                    if budget <= 0:
                        break
                    ex_gate_mul = _binary_expr("mul", ex_step, _feature_expr(j))
                    z_gate_mul = np.asarray(z_step * x[:, j], dtype=float)
                    fam = "interaction_gate_step" if (j in gate_set) else "interaction_step"
                    _try_add(f"soft_step(x{i}>{c:.4g})*x{j}", ex_gate_mul, 5.0, fam, (i, j), z_gate_mul)
                    if budget <= 0:
                        break
            if budget <= 0:
                break

    # 1) residual-guided interaction expansions
    if bool(enable_dynamic_interactions):
        for i in focus_idx:
            partner_idx = [int(j) for j in np.argsort(-np.abs(cross[i, :])).tolist() if int(j) != int(i)]
            if not partner_idx:
                partner_idx = [int(j) for j in np.argsort(-np.abs(p)).tolist() if int(j) != int(i)]
            for j in partner_idx[: max(2, int(partner_topk))]:
                ex1 = _binary_expr("mul", _feature_expr(i), _feature_expr(j))
                z1 = np.asarray(x[:, i] * x[:, j], dtype=float)
                fam = "interaction_gate" if (int(i) in gate_set or int(j) in gate_set) else "interaction_dynamic"
                _try_add(f"dyn:x{i}*x{j}", ex1, 3.0, fam, (i, j), z1)
                if budget <= 0:
                    break
                ex2 = _binary_expr("mul", _unary_expr("tanh", _feature_expr(i)), _feature_expr(j))
                z2 = np.asarray(np.tanh(x[:, i]) * x[:, j], dtype=float)
                _try_add(f"dyn:tanh(x{i})*x{j}", ex2, 4.0, fam, (i, j), z2)
                if budget <= 0:
                    break
                ex3 = _unary_expr("sin", ex1)
                z3 = np.asarray(np.sin(z1), dtype=float)
                _try_add(f"dyn:sin(x{i}*x{j})", ex3, 4.0, "interaction_dynamic", (i, j), z3)
                if budget <= 0:
                    break
            if budget <= 0:
                break

    # 2) gate-feature injections to mimic piecewise behavior
    if budget > 0 and gate_idx and bool(enable_gate_interactions):
        cont_idx = [int(i) for i in focus_idx if int(i) not in gate_set]
        for i in cont_idx:
            for g in gate_idx:
                exg = _binary_expr("mul", _feature_expr(i), _feature_expr(g))
                zg = np.asarray(x[:, i] * x[:, g], dtype=float)
                _try_add(f"gate:x{i}*x{g}", exg, 3.0, "interaction_gate", (i, g), zg)
                if budget <= 0:
                    break
                ext = _binary_expr("mul", _unary_expr("tanh", _feature_expr(i)), _feature_expr(g))
                zt = np.asarray(np.tanh(x[:, i]) * x[:, g], dtype=float)
                _try_add(f"gate:tanh(x{i})*x{g}", ext, 4.0, "interaction_gate", (i, g), zt)
                if budget <= 0:
                    break
            if budget <= 0:
                break

    return new_terms


def _prune_candidate_pool(
    *,
    candidates: Sequence[CandidateTerm],
    keep_expr_keys: set[str],
    feature_names: Sequence[str],
    max_pool_size: int,
) -> list[CandidateTerm]:
    pool = list(candidates)
    cap = int(max(16, max_pool_size))
    if len(pool) <= cap:
        return pool

    # Always keep foundational terms.
    anchor_keys: set[str] = set()
    for c in pool:
        if str(c.family) == "linear":
            anchor_keys.add(json.dumps(c.expr, sort_keys=True))
        if str(c.family).startswith("unary_") and len(c.features) == 1 and int(c.features[0]) < int(len(feature_names)):
            anchor_keys.add(json.dumps(c.expr, sort_keys=True))

    keep_all = set(keep_expr_keys) | anchor_keys
    keep_terms = [c for c in pool if json.dumps(c.expr, sort_keys=True) in keep_all]
    rest = [c for c in pool if json.dumps(c.expr, sort_keys=True) not in keep_all]
    rest.sort(key=lambda c: float(c.prior_corr), reverse=True)
    budget = max(0, cap - len(keep_terms))
    return keep_terms + rest[:budget]


class SymbolicSubsetSelectionProblem(BlackBoxProblem):
    def __init__(
        self,
        *,
        X_fit: np.ndarray,
        y_fit: np.ndarray,
        candidates: Sequence[CandidateTerm],
        max_terms: int,
        ridge_l2: float,
        rolling_folds: int,
        rolling_val_ratio: float,
        min_train: int,
        strict4_branch_mode: bool,
        strict4_gate_idx: tuple[int, int, int, int] | None,
        strict4_min_branch_train: int,
        strict4_branch_parallel_workers: int,
        inner_opt_enabled: bool,
        inner_opt_adam_steps: int,
        inner_opt_adam_lr: float,
        inner_opt_lbfgs_steps: int,
        inner_opt_lbfgs_lr: float,
        inner_opt_accept_rmse_tol: float,
        inner_opt_accept_rel_tol: float,
        inner_opt_guard_patience: int,
        inner_opt_guard_check_interval: int,
        inner_opt_alt_freeze_readout: bool,
        inner_opt_grad_clip_norm: float,
        inner_opt_residual_clip_q: float,
        graph_cache: ExpressionGraphCache | None = None,
    ) -> None:
        self.X_fit = np.asarray(X_fit, dtype=float)
        self.y_fit = np.asarray(y_fit, dtype=float).reshape(-1, 1)
        self.candidates = list(candidates)
        self.families = tuple(sorted({str(c.family) for c in self.candidates}))
        self.family_to_idx = {str(v): int(i) for i, v in enumerate(self.families)}
        self.max_terms = int(max(2, max_terms))
        self.base_ridge_l2 = float(max(0.0, ridge_l2))
        self.strict4_branch_mode = bool(strict4_branch_mode)
        self.strict4_gate_idx = strict4_gate_idx
        self.base_strict4_min_branch_train = int(max(8, strict4_min_branch_train))
        self.strict4_branch_parallel_workers = int(max(1, strict4_branch_parallel_workers))
        self.inner_opt_enabled = bool(inner_opt_enabled)
        self.inner_opt_adam_steps = int(max(0, inner_opt_adam_steps))
        self.inner_opt_adam_lr = float(max(1e-8, inner_opt_adam_lr))
        self.inner_opt_lbfgs_steps = int(max(0, inner_opt_lbfgs_steps))
        self.inner_opt_lbfgs_lr = float(max(1e-8, inner_opt_lbfgs_lr))
        self.inner_opt_accept_rmse_tol = float(max(0.0, inner_opt_accept_rmse_tol))
        self.inner_opt_accept_rel_tol = float(max(0.0, inner_opt_accept_rel_tol))
        self.inner_opt_guard_patience = int(max(1, inner_opt_guard_patience))
        self.inner_opt_guard_check_interval = int(max(1, inner_opt_guard_check_interval))
        self.inner_opt_alt_freeze_readout = bool(inner_opt_alt_freeze_readout)
        self.inner_opt_grad_clip_norm = float(max(0.0, inner_opt_grad_clip_norm))
        self.inner_opt_residual_clip_q = float(np.clip(inner_opt_residual_clip_q, 0.70, 0.999))
        self.graph_cache = graph_cache
        self.splits = _rolling_splits(
            int(self.X_fit.shape[0]),
            folds=int(max(1, rolling_folds)),
            val_ratio=float(rolling_val_ratio),
            min_train=int(max(128, min_train)),
        )
        self._cache: dict[tuple[tuple[int, ...], str], tuple[np.ndarray, dict[str, Any]]] = {}

        n_terms = int(len(self.candidates))
        n_fam = int(len(self.families))
        self.n_hyper_genes = 11
        bounds = (
            [(-1.0, 1.0) for _ in range(n_terms)]
            + [(-0.8, 0.8) for _ in range(n_fam)]
            + [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]
            + [(0.0, 1.0) for _ in range(self.n_hyper_genes)]
        )
        super().__init__(
            name="SymbolicSubsetSelectionProblem",
            dimension=int(n_terms + n_fam + 3 + self.n_hyper_genes),
            bounds=bounds,
            objectives=["minimize", "minimize", "minimize"],
        )

    def _decode(self, x: np.ndarray) -> tuple[list[int], int, dict[str, Any]]:
        z = np.asarray(x, dtype=float).reshape(-1)
        n_terms = int(len(self.candidates))
        n_fam = int(len(self.families))
        raw_scores = np.asarray(z[:n_terms], dtype=float)
        family_bias = np.asarray(z[n_terms : n_terms + n_fam], dtype=float)
        k_gene = float(np.clip(z[n_terms + n_fam], 0.0, 1.0))
        thresh_gene = float(np.clip(z[n_terms + n_fam + 1], 0.0, 1.0))
        inter_gene = float(np.clip(z[n_terms + n_fam + 2], 0.0, 1.0))
        hyper = np.asarray(z[n_terms + n_fam + 3 :], dtype=float)
        if hyper.size < self.n_hyper_genes:
            hyper = np.pad(hyper, (0, int(self.n_hyper_genes - hyper.size)), constant_values=0.5)

        prior_corr_w = float(0.05 + 0.85 * float(np.clip(hyper[0], 0.0, 1.0)))
        family_bias_scale = float(0.10 + 1.40 * float(np.clip(hyper[1], 0.0, 1.0)))
        tuned_l2 = float(10.0 ** (-8.0 + 6.0 * float(np.clip(hyper[2], 0.0, 1.0))))
        complexity_scale = float(0.30 + 1.50 * float(np.clip(hyper[3], 0.0, 1.0)))
        family_penalty_scale = float(0.20 + 1.20 * float(np.clip(hyper[4], 0.0, 1.0)))
        feature_penalty_scale = float(0.20 + 1.20 * float(np.clip(hyper[5], 0.0, 1.0)))
        drift_weight = float(0.05 + 0.40 * float(np.clip(hyper[6], 0.0, 1.0)))
        strict4_min_train_ratio = float(0.02 + 0.18 * float(np.clip(hyper[7], 0.0, 1.0)))
        q_low = float(0.15 + 0.45 * float(np.clip(hyper[8], 0.0, 1.0)))
        q_span = float(0.20 + 0.70 * float(np.clip(hyper[9], 0.0, 1.0)))
        inter_floor_ratio = float(0.50 * float(np.clip(hyper[10], 0.0, 1.0)))

        k = int(round(2 + k_gene * (self.max_terms - 2)))
        k = int(np.clip(k, 2, self.max_terms))

        adj = np.asarray(raw_scores, dtype=float).copy()
        for i, cand in enumerate(self.candidates):
            fam_idx = int(self.family_to_idx.get(str(cand.family), 0))
            adj[i] = float(
                adj[i]
                + prior_corr_w * float(cand.prior_corr)
                + family_bias_scale * float(family_bias[fam_idx])
            )

        q = float(np.clip(q_low + q_span * thresh_gene, 0.05, 0.98))
        cut = float(np.quantile(adj, q))
        active = [int(i) for i in range(adj.size) if float(adj[i]) >= cut]
        if len(active) < 2:
            active = list(range(int(adj.size)))
        order = sorted(active, key=lambda i: float(adj[i]), reverse=True)

        inter_cap = int(max(1, round((inter_floor_ratio + (1.0 - inter_floor_ratio) * inter_gene) * k)))
        inter_count = 0
        picked: list[int] = []
        for i in order:
            fam = str(self.candidates[i].family)
            if fam == "interaction" and inter_count >= inter_cap:
                continue
            picked.append(int(i))
            if fam == "interaction":
                inter_count += 1
            if len(picked) >= k:
                break

        if len(picked) < 2:
            order_all = list(np.argsort(-adj))
            picked = [int(i) for i in order_all[: max(2, k)]]

        # Encourage at least one linear anchor when possible.
        if not any(str(self.candidates[i].family) == "linear" for i in picked):
            linear_idx = [i for i in range(len(self.candidates)) if str(self.candidates[i].family) == "linear"]
            if linear_idx:
                best_linear = int(max(linear_idx, key=lambda i: float(adj[i])))
                picked = [best_linear] + [i for i in picked if i != best_linear]
                picked = picked[:k]

        meta = {
            "k": int(k),
            "threshold_q": float(q),
            "threshold_cut": float(cut),
            "interaction_cap": int(inter_cap),
            "interaction_count": int(sum(1 for i in picked if str(self.candidates[i].family) == "interaction")),
            "tuned_l2": float(tuned_l2),
            "complexity_scale": float(complexity_scale),
            "family_penalty_scale": float(family_penalty_scale),
            "feature_penalty_scale": float(feature_penalty_scale),
            "drift_weight": float(drift_weight),
            "strict4_min_train_ratio": float(strict4_min_train_ratio),
            "prior_corr_w": float(prior_corr_w),
            "family_bias_scale": float(family_bias_scale),
        }
        if len(picked) < 2:
            picked = [int(i) for i in list(np.argsort(-adj))[:2]]
        return picked, k, meta

    @staticmethod
    def _cache_sig(meta: Mapping[str, Any]) -> str:
        keys = (
            "tuned_l2",
            "complexity_scale",
            "family_penalty_scale",
            "feature_penalty_scale",
            "drift_weight",
            "strict4_min_train_ratio",
            "prior_corr_w",
            "family_bias_scale",
            "threshold_q",
            "interaction_cap",
            "k",
        )
        out: dict[str, Any] = {}
        for k in keys:
            v = meta.get(k)
            if isinstance(v, float):
                out[str(k)] = round(float(v), 7)
            else:
                out[str(k)] = v
        return json.dumps(out, sort_keys=True, separators=(",", ":"))

    def _eval_fold_global(
        self,
        genome: Sequence[Mapping[str, Any]],
        tr_idx: np.ndarray,
        va_idx: np.ndarray,
        *,
        l2: float,
    ) -> dict[str, Any]:
        fit = _three_layer_fit_predict(
            genome=genome,
            X_train=self.X_fit[tr_idx],
            y_train=self.y_fit[tr_idx],
            X_eval=self.X_fit[va_idx],
            y_eval=self.y_fit[va_idx],
            l2=float(max(0.0, l2)),
            inner_opt_enabled=bool(self.inner_opt_enabled),
            inner_opt_adam_steps=int(self.inner_opt_adam_steps),
            inner_opt_adam_lr=float(self.inner_opt_adam_lr),
            inner_opt_lbfgs_steps=int(self.inner_opt_lbfgs_steps),
            inner_opt_lbfgs_lr=float(self.inner_opt_lbfgs_lr),
            inner_opt_accept_rmse_tol=float(self.inner_opt_accept_rmse_tol),
            inner_opt_accept_rel_tol=float(self.inner_opt_accept_rel_tol),
            inner_opt_guard_patience=int(self.inner_opt_guard_patience),
            inner_opt_guard_check_interval=int(self.inner_opt_guard_check_interval),
            inner_opt_alt_freeze_readout=bool(self.inner_opt_alt_freeze_readout),
            inner_opt_grad_clip_norm=float(self.inner_opt_grad_clip_norm),
            inner_opt_residual_clip_q=float(self.inner_opt_residual_clip_q),
        )
        m_eval = dict(fit.get("metrics_eval", {}))
        return {
            "rmse": float(m_eval.get("rmse", float("inf"))),
            "mode": "global",
            "branch_detail": {"inner_opt_info": _jsonable(fit.get("inner_opt_info", {}))},
        }

    def _eval_fold_strict4(
        self,
        genome: Sequence[Mapping[str, Any]],
        tr_idx: np.ndarray,
        va_idx: np.ndarray,
        *,
        l2: float,
        strict4_min_branch_train: int,
    ) -> dict[str, Any]:
        if not self.strict4_branch_mode or self.strict4_gate_idx is None:
            return self._eval_fold_global(genome, tr_idx, va_idx, l2=l2)

        Xtr = np.asarray(self.X_fit[tr_idx], dtype=float)
        ytr = np.asarray(self.y_fit[tr_idx], dtype=float)
        Xva = np.asarray(self.X_fit[va_idx], dtype=float)
        yva = np.asarray(self.y_fit[va_idx], dtype=float)

        keys_tr = _strict4_keys_from_X(Xtr, self.strict4_gate_idx)
        keys_va = _strict4_keys_from_X(Xva, self.strict4_gate_idx)

        idx_tr_by_key: dict[tuple[int, int, int, int], np.ndarray] = {}
        idx_va_by_key: dict[tuple[int, int, int, int], np.ndarray] = {}
        for k in STRICT4_REGIME_ORDER:
            idx_tr_by_key[k] = np.asarray([i for i, kk in enumerate(keys_tr) if kk == k], dtype=int)
            idx_va_by_key[k] = np.asarray([i for i, kk in enumerate(keys_va) if kk == k], dtype=int)

        fit_global = _three_layer_fit_predict(
            genome=genome,
            X_train=Xtr,
            y_train=ytr,
            X_eval=Xva,
            y_eval=yva,
            l2=float(max(0.0, l2)),
            inner_opt_enabled=bool(self.inner_opt_enabled),
            inner_opt_adam_steps=int(self.inner_opt_adam_steps),
            inner_opt_adam_lr=float(self.inner_opt_adam_lr),
            inner_opt_lbfgs_steps=int(self.inner_opt_lbfgs_steps),
            inner_opt_lbfgs_lr=float(self.inner_opt_lbfgs_lr),
            inner_opt_accept_rmse_tol=float(self.inner_opt_accept_rmse_tol),
            inner_opt_accept_rel_tol=float(self.inner_opt_accept_rel_tol),
            inner_opt_guard_patience=int(self.inner_opt_guard_patience),
            inner_opt_guard_check_interval=int(self.inner_opt_guard_check_interval),
            inner_opt_alt_freeze_readout=bool(self.inner_opt_alt_freeze_readout),
            inner_opt_grad_clip_norm=float(self.inner_opt_grad_clip_norm),
            inner_opt_residual_clip_q=float(self.inner_opt_residual_clip_q),
        )
        pred_global = np.asarray(fit_global.get("pred_eval"), dtype=float).reshape(-1, 1)

        pred_va = np.asarray(pred_global, dtype=float).copy()
        branch_rmse: dict[str, float] = {}
        branch_used_train: dict[str, int] = {}
        branch_used_fallback: dict[str, bool] = {}

        def _fit_branch(k: tuple[int, int, int, int]) -> tuple[tuple[int, int, int, int], np.ndarray | None, bool]:
            tr_local = np.asarray(idx_tr_by_key[k], dtype=int)
            va_local = np.asarray(idx_va_by_key[k], dtype=int)
            if int(va_local.size) <= 0:
                return k, None, True
            if int(tr_local.size) < int(strict4_min_branch_train):
                return k, None, True
            fit = _three_layer_fit_predict(
                genome=genome,
                X_train=Xtr[tr_local],
                y_train=ytr[tr_local],
                X_eval=Xva[va_local],
                y_eval=yva[va_local],
                l2=float(max(0.0, l2)),
                inner_opt_enabled=bool(self.inner_opt_enabled),
                inner_opt_adam_steps=int(self.inner_opt_adam_steps),
                inner_opt_adam_lr=float(self.inner_opt_adam_lr),
                inner_opt_lbfgs_steps=int(self.inner_opt_lbfgs_steps),
                inner_opt_lbfgs_lr=float(self.inner_opt_lbfgs_lr),
                inner_opt_accept_rmse_tol=float(self.inner_opt_accept_rmse_tol),
                inner_opt_accept_rel_tol=float(self.inner_opt_accept_rel_tol),
                inner_opt_guard_patience=int(self.inner_opt_guard_patience),
                inner_opt_guard_check_interval=int(self.inner_opt_guard_check_interval),
                inner_opt_alt_freeze_readout=bool(self.inner_opt_alt_freeze_readout),
                inner_opt_grad_clip_norm=float(self.inner_opt_grad_clip_norm),
                inner_opt_residual_clip_q=float(self.inner_opt_residual_clip_q),
            )
            pred = np.asarray(fit.get("pred_eval"), dtype=float).reshape(-1, 1)
            return k, pred, False

        n_workers = int(max(1, min(self.strict4_branch_parallel_workers, len(STRICT4_REGIME_ORDER))))
        if n_workers <= 1:
            branch_results = [_fit_branch(k) for k in STRICT4_REGIME_ORDER]
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as ex:
                fut = [ex.submit(_fit_branch, k) for k in STRICT4_REGIME_ORDER]
                branch_results = [f.result() for f in fut]

        for k, pred_k, fallback in branch_results:
            va_local = np.asarray(idx_va_by_key[k], dtype=int)
            tr_local = np.asarray(idx_tr_by_key[k], dtype=int)
            if int(va_local.size) <= 0:
                continue
            if pred_k is not None and not bool(fallback):
                pred_va[va_local] = np.asarray(pred_k, dtype=float)
            yk = np.asarray(yva[va_local], dtype=float).reshape(-1)
            pk = np.asarray(pred_va[va_local], dtype=float).reshape(-1)
            branch_rmse[str(k)] = float(_rmse(yk, pk))
            branch_used_train[str(k)] = int(tr_local.size)
            branch_used_fallback[str(k)] = bool(fallback)

        rmse = float(_rmse(yva, pred_va))
        return {
            "rmse": float(rmse),
            "mode": "strict4_branch",
            "branch_detail": {
                "branch_rmse": dict(branch_rmse),
                "branch_train_size": dict(branch_used_train),
                "branch_fallback": dict(branch_used_fallback),
            },
        }

    def _evaluate_subset(self, subset_idx: Sequence[int], meta: Mapping[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
        key_subset = tuple(sorted(int(i) for i in subset_idx))
        key = (key_subset, self._cache_sig(meta))
        hit = self._cache.get(key)
        if hit is not None:
            return hit

        genome = [{"name": self.candidates[i].name, "expr": dict(self.candidates[i].expr)} for i in key_subset]
        tuned_l2 = float(max(0.0, meta.get("tuned_l2", self.base_ridge_l2)))
        strict4_ratio = float(np.clip(meta.get("strict4_min_train_ratio", 0.08), 0.01, 0.30))
        complexity_scale = float(max(0.05, meta.get("complexity_scale", 1.0)))
        family_penalty_scale = float(max(0.05, meta.get("family_penalty_scale", 1.0)))
        feature_penalty_scale = float(max(0.05, meta.get("feature_penalty_scale", 1.0)))
        drift_weight = float(max(0.0, meta.get("drift_weight", 0.15)))

        fold_rmse: list[float] = []
        fold_branch: list[dict[str, Any]] = []
        for fold_id, (tr_idx, va_idx) in enumerate(self.splits):
            tr_arr = np.asarray(tr_idx, dtype=int)
            va_arr = np.asarray(va_idx, dtype=int)
            strict4_min_train = int(
                max(
                    self.base_strict4_min_branch_train,
                    round(float(strict4_ratio) * float(tr_arr.size)),
                )
            )
            fold_res = self._eval_fold_strict4(
                genome,
                tr_idx=tr_arr,
                va_idx=va_arr,
                l2=tuned_l2,
                strict4_min_branch_train=strict4_min_train,
            )
            fold_rmse.append(float(fold_res["rmse"]))
            fold_branch.append(dict(fold_res.get("branch_detail", {})))

        rmse_mean = float(np.mean(fold_rmse))
        rmse_std = float(np.std(fold_rmse))
        rmse_drift = float(np.mean(np.abs(np.diff(np.asarray(fold_rmse, dtype=float))))) if len(fold_rmse) >= 2 else 0.0
        complexity = float(sum(float(self.candidates[i].complexity) for i in key_subset))

        fam_counts: dict[str, int] = {}
        feat_counts: dict[int, int] = {}
        for i in key_subset:
            c = self.candidates[i]
            fam_counts[str(c.family)] = int(fam_counts.get(str(c.family), 0) + 1)
            for f in c.features:
                feat_counts[int(f)] = int(feat_counts.get(int(f), 0) + 1)
        fam_share = np.asarray([float(v) for v in fam_counts.values()], dtype=float)
        if fam_share.size > 0:
            fam_share = fam_share / float(np.sum(fam_share))
        feat_share = np.asarray([float(v) for v in feat_counts.values()], dtype=float)
        if feat_share.size > 0:
            feat_share = feat_share / float(np.sum(feat_share))
        fam_concentration = float(np.sum(fam_share**2)) if fam_share.size > 0 else 1.0
        feat_concentration = float(np.sum(feat_share**2)) if feat_share.size > 0 else 1.0

        obj_accuracy = float(rmse_mean)
        obj_stability = float(rmse_std + drift_weight * rmse_drift)
        obj_complexity = float(
            complexity_scale * (complexity / max(1.0, float(self.max_terms)))
            + family_penalty_scale * fam_concentration
            + feature_penalty_scale * feat_concentration
        )
        out_obj = np.asarray([obj_accuracy, obj_stability, obj_complexity], dtype=float)
        detail = {
            "subset_size": int(len(key_subset)),
            "subset_idx": [int(i) for i in key_subset],
            "subset_names": [self.candidates[i].name for i in key_subset],
            "subset_families": [self.candidates[i].family for i in key_subset],
            "fold_rmse": [float(v) for v in fold_rmse],
            "fold_branch_detail": _jsonable(fold_branch),
            "rmse_mean": float(rmse_mean),
            "rmse_std": float(rmse_std),
            "rmse_drift": float(rmse_drift),
            "complexity_raw": float(complexity),
            "family_concentration": float(fam_concentration),
            "feature_concentration": float(feat_concentration),
            "tuned_l2": float(tuned_l2),
            "strict4_min_train_ratio": float(strict4_ratio),
            "complexity_scale": float(complexity_scale),
            "family_penalty_scale": float(family_penalty_scale),
            "feature_penalty_scale": float(feature_penalty_scale),
            "drift_weight": float(drift_weight),
            "decode_meta": _jsonable(dict(meta)),
        }
        self._cache[key] = (out_obj, detail)
        return out_obj, detail

    @staticmethod
    def _design_matrix_for_genome(
        genome: Sequence[Mapping[str, Any]],
        X: np.ndarray,
        *,
        graph_cache: ExpressionGraphCache | None = None,
        batch_key: str | None = None,
    ) -> np.ndarray:
        x = np.asarray(X, dtype=float)
        if x.ndim != 2:
            raise ValueError("X must be 2D")
        if len(genome) <= 0:
            return np.zeros((int(x.shape[0]), 0), dtype=float)
        if graph_cache is None:
            phi = evaluate_genome_numpy(genome, x)
            return np.asarray(phi, dtype=float)
        cols: list[np.ndarray] = []
        for term in genome:
            expr = term.get("expr", term)
            z = graph_cache.evaluate_expression(
                expr,
                x,
                param_values=None,
                eps=1e-6,
                batch_key=batch_key,
            )
            cols.append(np.asarray(z, dtype=float).reshape(-1, 1))
        if not cols:
            return np.zeros((int(x.shape[0]), 0), dtype=float)
        return np.concatenate(cols, axis=1)

    @classmethod
    def _batched_ridge_predict(
        cls,
        *,
        genomes: Sequence[Sequence[Mapping[str, Any]]],
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_eval: np.ndarray,
        l2_values: Sequence[float],
        graph_cache: ExpressionGraphCache | None = None,
        batch_key_train: str | None = None,
        batch_key_eval: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        xtr = np.asarray(X_train, dtype=float)
        ytr = _as_2d(np.asarray(y_train, dtype=float))
        xev = np.asarray(X_eval, dtype=float)
        B = int(len(genomes))
        if B <= 0:
            return np.zeros((0, int(xev.shape[0]), int(ytr.shape[1])), dtype=float), np.zeros((0, int(xtr.shape[0]), int(ytr.shape[1])), dtype=float)

        groups: dict[int, list[int]] = {}
        for i, g in enumerate(genomes):
            groups.setdefault(int(len(g)), []).append(int(i))

        pred_eval = np.zeros((B, int(xev.shape[0]), int(ytr.shape[1])), dtype=float)
        pred_train = np.zeros((B, int(xtr.shape[0]), int(ytr.shape[1])), dtype=float)

        try:
            import torch
        except Exception:
            # fallback to per-candidate ridge
            for i, g in enumerate(genomes):
                fit = evaluate_genome_with_ridge(
                    g,
                    X_train=xtr,
                    y_train=ytr,
                    X_eval=xev,
                    y_eval=None,
                    l2=float(max(0.0, l2_values[i])),
                )
                pred_eval[i] = _as_2d(np.asarray(fit.get("pred_eval"), dtype=float))
                pred_train[i] = _as_2d(np.asarray(fit.get("pred_train"), dtype=float))
            return pred_eval, pred_train

        ytr_t = torch.as_tensor(ytr, dtype=torch.float64)

        for k, idxs in groups.items():
            if int(k) <= 0:
                # intercept-only model
                b = np.mean(ytr, axis=0, keepdims=True)
                for i in idxs:
                    pred_train[i] = np.repeat(b, repeats=int(xtr.shape[0]), axis=0)
                    pred_eval[i] = np.repeat(b, repeats=int(xev.shape[0]), axis=0)
                continue

            phis_tr = []
            phis_ev = []
            reg_vals = []
            for i in idxs:
                g = genomes[int(i)]
                phis_tr.append(
                    cls._design_matrix_for_genome(
                        g,
                        xtr,
                        graph_cache=graph_cache,
                        batch_key=batch_key_train,
                    )
                )
                phis_ev.append(
                    cls._design_matrix_for_genome(
                        g,
                        xev,
                        graph_cache=graph_cache,
                        batch_key=batch_key_eval,
                    )
                )
                reg_vals.append(float(max(0.0, l2_values[int(i)])))

            A_tr = np.asarray(np.stack(phis_tr, axis=0), dtype=float)  # [Bg, n, k]
            A_ev = np.asarray(np.stack(phis_ev, axis=0), dtype=float)
            Bg = int(A_tr.shape[0])
            ones_tr = np.ones((Bg, int(A_tr.shape[1]), 1), dtype=float)
            ones_ev = np.ones((Bg, int(A_ev.shape[1]), 1), dtype=float)
            Atr = np.concatenate([A_tr, ones_tr], axis=2)  # [Bg,n,k+1]
            Aev = np.concatenate([A_ev, ones_ev], axis=2)

            Atr_t = torch.as_tensor(Atr, dtype=torch.float64)
            Aev_t = torch.as_tensor(Aev, dtype=torch.float64)
            yb_t = ytr_t.unsqueeze(0).expand(Bg, -1, -1)  # [Bg,n,m]

            At = Atr_t.transpose(1, 2)  # [Bg,k+1,n]
            lhs = torch.bmm(At, Atr_t)  # [Bg,k+1,k+1]
            rhs = torch.bmm(At, yb_t)  # [Bg,k+1,m]

            reg = torch.eye(int(k + 1), dtype=torch.float64).unsqueeze(0).repeat(Bg, 1, 1)
            reg[:, -1, -1] = 0.0
            lam = torch.as_tensor(np.asarray(reg_vals, dtype=float), dtype=torch.float64).reshape(Bg, 1, 1)
            lhs = lhs + lam * reg

            try:
                W = torch.linalg.solve(lhs, rhs)  # [Bg,k+1,m]
            except Exception:
                W = torch.matmul(torch.linalg.pinv(lhs), rhs)

            pred_tr_g = torch.bmm(Atr_t, W).cpu().numpy()
            pred_ev_g = torch.bmm(Aev_t, W).cpu().numpy()
            for loc, i in enumerate(idxs):
                pred_train[int(i)] = np.asarray(pred_tr_g[int(loc)], dtype=float)
                pred_eval[int(i)] = np.asarray(pred_ev_g[int(loc)], dtype=float)

        return pred_eval, pred_train

    def evaluate_population_batch(self, population: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pop = np.asarray(population, dtype=float)
        if pop.ndim == 1:
            pop = pop.reshape(1, -1)
        n = int(pop.shape[0])
        out_obj = np.zeros((n, 3), dtype=float)
        out_vio = np.zeros((n,), dtype=float)
        if n <= 0:
            return out_obj, out_vio

        decoded: list[tuple[list[int], int, dict[str, Any]]] = []
        cache_keys: list[tuple[tuple[int, ...], str]] = []
        need_eval_idx: list[int] = []
        for i in range(n):
            subset_idx, _, meta = self._decode(np.asarray(pop[i], dtype=float))
            key_subset = tuple(sorted(int(v) for v in subset_idx))
            key = (key_subset, self._cache_sig(meta))
            decoded.append((subset_idx, int(len(subset_idx)), meta))
            cache_keys.append(key)
            if key in self._cache:
                obj, _detail = self._cache[key]
                out_obj[i] = np.asarray(obj, dtype=float)
            else:
                need_eval_idx.append(int(i))

        if not need_eval_idx:
            return out_obj, out_vio

        genomes: list[list[Mapping[str, Any]]] = []
        metas: list[Mapping[str, Any]] = []
        for i in need_eval_idx:
            subset_idx, _k, meta = decoded[i]
            g = [{"name": self.candidates[j].name, "expr": dict(self.candidates[j].expr)} for j in sorted(int(v) for v in subset_idx)]
            genomes.append(g)
            metas.append(meta)

        B = int(len(genomes))
        fold_rmse = [[] for _ in range(B)]
        fold_branch = [[] for _ in range(B)]

        for fold_id, (tr_idx, va_idx) in enumerate(self.splits):
            tr_arr = np.asarray(tr_idx, dtype=int)
            va_arr = np.asarray(va_idx, dtype=int)
            Xtr = np.asarray(self.X_fit[tr_arr], dtype=float)
            ytr = np.asarray(self.y_fit[tr_arr], dtype=float)
            Xva = np.asarray(self.X_fit[va_arr], dtype=float)
            yva = np.asarray(self.y_fit[va_arr], dtype=float)

            l2s = [float(max(0.0, m.get("tuned_l2", self.base_ridge_l2))) for m in metas]
            pred_global, _pred_train = self._batched_ridge_predict(
                genomes=genomes,
                X_train=Xtr,
                y_train=ytr,
                X_eval=Xva,
                l2_values=l2s,
                graph_cache=self.graph_cache,
                batch_key_train=f"fold{int(fold_id)}|global|tr",
                batch_key_eval=f"fold{int(fold_id)}|global|va",
            )  # [B,nva,m]

            if not self.strict4_branch_mode or self.strict4_gate_idx is None:
                for bi in range(B):
                    rm = float(_rmse(yva, pred_global[bi]))
                    fold_rmse[bi].append(rm)
                    fold_branch[bi].append({})
                continue

            keys_tr = _strict4_keys_from_X(Xtr, self.strict4_gate_idx)
            keys_va = _strict4_keys_from_X(Xva, self.strict4_gate_idx)
            idx_tr_by_key = {k: np.asarray([ii for ii, kk in enumerate(keys_tr) if kk == k], dtype=int) for k in STRICT4_REGIME_ORDER}
            idx_va_by_key = {k: np.asarray([ii for ii, kk in enumerate(keys_va) if kk == k], dtype=int) for k in STRICT4_REGIME_ORDER}

            pred_va = np.asarray(pred_global, dtype=float).copy()  # [B,nva,m]
            branch_detail_all: list[dict[str, Any]] = [{"branch_rmse": {}, "branch_train_size": {}, "branch_fallback": {}} for _ in range(B)]

            for regime in STRICT4_REGIME_ORDER:
                tr_local = np.asarray(idx_tr_by_key[regime], dtype=int)
                va_local = np.asarray(idx_va_by_key[regime], dtype=int)
                if int(va_local.size) <= 0:
                    continue

                active_local: list[int] = []
                for bi, meta in enumerate(metas):
                    strict4_ratio = float(np.clip(meta.get("strict4_min_train_ratio", 0.08), 0.01, 0.30))
                    min_train = int(max(self.base_strict4_min_branch_train, round(strict4_ratio * float(tr_arr.size))))
                    use_branch = int(tr_local.size) >= int(min_train)
                    branch_detail_all[bi]["branch_train_size"][str(regime)] = int(tr_local.size)
                    branch_detail_all[bi]["branch_fallback"][str(regime)] = bool(not use_branch)
                    if use_branch:
                        active_local.append(int(bi))

                if active_local:
                    genomes_act = [genomes[bi] for bi in active_local]
                    l2s_act = [l2s[bi] for bi in active_local]
                    pred_loc, _ = self._batched_ridge_predict(
                        genomes=genomes_act,
                        X_train=Xtr[tr_local],
                        y_train=ytr[tr_local],
                        X_eval=Xva[va_local],
                        l2_values=l2s_act,
                        graph_cache=self.graph_cache,
                        batch_key_train=f"fold{int(fold_id)}|{str(regime)}|tr",
                        batch_key_eval=f"fold{int(fold_id)}|{str(regime)}|va",
                    )
                    for kpos, bi in enumerate(active_local):
                        pred_va[bi, va_local, :] = pred_loc[kpos]

                for bi in range(B):
                    yk = np.asarray(yva[va_local], dtype=float).reshape(-1)
                    pk = np.asarray(pred_va[bi, va_local, :], dtype=float).reshape(-1)
                    branch_detail_all[bi]["branch_rmse"][str(regime)] = float(_rmse(yk, pk))

            for bi in range(B):
                rm = float(_rmse(yva, pred_va[bi]))
                fold_rmse[bi].append(rm)
                fold_branch[bi].append(dict(branch_detail_all[bi]))

        # finalize and write cache
        for loc, i in enumerate(need_eval_idx):
            subset_idx, _k, meta = decoded[i]
            key_subset = tuple(sorted(int(v) for v in subset_idx))
            key = (key_subset, self._cache_sig(meta))
            key_int = [int(v) for v in key_subset]
            rm_arr = np.asarray(fold_rmse[loc], dtype=float)
            rmse_mean = float(np.mean(rm_arr))
            rmse_std = float(np.std(rm_arr))
            rmse_drift = float(np.mean(np.abs(np.diff(rm_arr)))) if rm_arr.size >= 2 else 0.0
            complexity = float(sum(float(self.candidates[j].complexity) for j in key_int))

            fam_counts: dict[str, int] = {}
            feat_counts: dict[int, int] = {}
            for j in key_int:
                c = self.candidates[j]
                fam_counts[str(c.family)] = int(fam_counts.get(str(c.family), 0) + 1)
                for f in c.features:
                    feat_counts[int(f)] = int(feat_counts.get(int(f), 0) + 1)
            fam_share = np.asarray([float(v) for v in fam_counts.values()], dtype=float)
            if fam_share.size > 0:
                fam_share = fam_share / float(np.sum(fam_share))
            feat_share = np.asarray([float(v) for v in feat_counts.values()], dtype=float)
            if feat_share.size > 0:
                feat_share = feat_share / float(np.sum(feat_share))
            fam_concentration = float(np.sum(fam_share**2)) if fam_share.size > 0 else 1.0
            feat_concentration = float(np.sum(feat_share**2)) if feat_share.size > 0 else 1.0

            complexity_scale = float(max(0.05, meta.get("complexity_scale", 1.0)))
            family_penalty_scale = float(max(0.05, meta.get("family_penalty_scale", 1.0)))
            feature_penalty_scale = float(max(0.05, meta.get("feature_penalty_scale", 1.0)))
            drift_weight = float(max(0.0, meta.get("drift_weight", 0.15)))
            tuned_l2 = float(max(0.0, meta.get("tuned_l2", self.base_ridge_l2)))
            strict4_ratio = float(np.clip(meta.get("strict4_min_train_ratio", 0.08), 0.01, 0.30))

            obj = np.asarray(
                [
                    float(rmse_mean),
                    float(rmse_std + drift_weight * rmse_drift),
                    float(
                        complexity_scale * (complexity / max(1.0, float(self.max_terms)))
                        + family_penalty_scale * fam_concentration
                        + feature_penalty_scale * feat_concentration
                    ),
                ],
                dtype=float,
            )
            detail = {
                "subset_size": int(len(key_int)),
                "subset_idx": [int(v) for v in key_int],
                "subset_names": [self.candidates[j].name for j in key_int],
                "subset_families": [self.candidates[j].family for j in key_int],
                "fold_rmse": [float(v) for v in rm_arr.tolist()],
                "fold_branch_detail": _jsonable(fold_branch[loc]),
                "rmse_mean": float(rmse_mean),
                "rmse_std": float(rmse_std),
                "rmse_drift": float(rmse_drift),
                "complexity_raw": float(complexity),
                "family_concentration": float(fam_concentration),
                "feature_concentration": float(feat_concentration),
                "tuned_l2": float(tuned_l2),
                "strict4_min_train_ratio": float(strict4_ratio),
                "complexity_scale": float(complexity_scale),
                "family_penalty_scale": float(family_penalty_scale),
                "feature_penalty_scale": float(feature_penalty_scale),
                "drift_weight": float(drift_weight),
                "decode_meta": _jsonable(dict(meta)),
            }
            self._cache[key] = (obj, detail)
            out_obj[i] = np.asarray(obj, dtype=float)

        return out_obj, out_vio

    def evaluate(self, x):
        try:
            subset_idx, _, meta = self._decode(np.asarray(x, dtype=float))
            obj, _ = self._evaluate_subset(subset_idx, meta)
            return np.asarray(obj, dtype=float)
        except Exception:
            return np.asarray([1e6, 1e3, 1e3], dtype=float)

    def cache_top(self, *, topn: int = 20) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for _, (obj, detail) in self._cache.items():
            rows.append(
                {
                    "obj_accuracy": float(obj[0]),
                    "obj_stability": float(obj[1]),
                    "obj_complexity": float(obj[2]),
                    **_jsonable(detail),
                }
            )
        rows.sort(
            key=lambda r: (
                float(r["obj_accuracy"]),
                float(r["obj_stability"]),
                float(r["obj_complexity"]),
            )
        )
        return rows[: int(max(1, topn))]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="NSGABLACK outer (subset optimization) + MLBLACK inner (symbolic ridge eval) on Work-CI."
    )
    parser.add_argument("--csv-path", type=str, default=default_work_ci_csv())
    parser.add_argument("--target-col", type=str, default="ci")
    parser.add_argument("--test-fold-col", type=str, default="test_fold_10")
    parser.add_argument("--pop-size", type=int, default=32)
    parser.add_argument("--generations", type=int, default=25)
    parser.add_argument("--rolling-folds", type=int, default=3)
    parser.add_argument("--rolling-val-ratio", type=float, default=0.18)
    parser.add_argument("--max-terms", type=int, default=12)
    parser.add_argument("--ridge-l2", type=float, default=1e-4)
    parser.add_argument("--strict4-branch-mode", action="store_true")
    parser.add_argument("--strict4-min-branch-train", type=int, default=64)
    parser.add_argument("--strict4-branch-parallel-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outer-strategy", type=str, default="portfolio", choices=["nsga2", "moead", "vns", "portfolio"])
    parser.add_argument("--portfolio-phases", type=str, default="nsga2,moead,vns")
    parser.add_argument("--portfolio-phase-weights", type=str, default="2,1,1")
    parser.add_argument("--moead-neighborhood-size", type=int, default=12)
    parser.add_argument("--moead-delta", type=float, default=0.9)
    parser.add_argument("--moead-nr", type=int, default=2)
    parser.add_argument("--vns-k-max", type=int, default=5)
    parser.add_argument("--vns-batch-size", type=int, default=32)
    parser.add_argument("--inner-opt-enabled", type=int, default=1)
    parser.add_argument("--inner-opt-adam-steps", type=int, default=80)
    parser.add_argument("--inner-opt-adam-lr", type=float, default=1e-2)
    parser.add_argument("--inner-opt-lbfgs-steps", type=int, default=25)
    parser.add_argument("--inner-opt-lbfgs-lr", type=float, default=0.8)
    parser.add_argument("--inner-opt-accept-rmse-tol", type=float, default=0.0)
    parser.add_argument("--inner-opt-accept-rel-tol", type=float, default=0.01)
    parser.add_argument("--inner-opt-guard-patience", type=int, default=3)
    parser.add_argument("--inner-opt-guard-check-interval", type=int, default=10)
    parser.add_argument("--inner-opt-alt-freeze-readout", type=int, default=1)
    parser.add_argument("--inner-opt-grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--inner-opt-residual-clip-q", type=float, default=0.98)
    parser.add_argument("--batched-eval", type=int, default=1)
    parser.add_argument("--reinvest-search", type=int, default=1)
    parser.add_argument("--reinvest-pop-mult", type=float, default=1.5)
    parser.add_argument("--reinvest-gen-mult", type=float, default=1.5)
    parser.add_argument("--reinvest-strict4-workers-mult", type=float, default=1.5)
    parser.add_argument("--dynamic-pool-enabled", type=int, default=1)
    parser.add_argument("--dynamic-pool-epochs", type=int, default=4)
    parser.add_argument("--dynamic-init-minimal", type=int, default=1)
    parser.add_argument("--dynamic-expand-max-new", type=int, default=24)
    parser.add_argument("--dynamic-focus-top-features", type=int, default=5)
    parser.add_argument("--dynamic-partner-topk", type=int, default=4)
    parser.add_argument("--dynamic-top-cache-use", type=int, default=20)
    parser.add_argument("--dynamic-max-pool-size", type=int, default=240)
    parser.add_argument("--mechanism-info-gain-enabled", type=int, default=1)
    parser.add_argument("--mechanism-info-gain-min-corr", type=float, default=0.02)
    parser.add_argument("--mechanism-novelty-enabled", type=int, default=1)
    parser.add_argument("--mechanism-novelty-max-corr", type=float, default=0.985)
    parser.add_argument("--mechanism-curriculum-enabled", type=int, default=1)
    parser.add_argument("--mechanism-curriculum-stage1-ratio", type=float, default=0.34)
    parser.add_argument("--mechanism-curriculum-stage2-ratio", type=float, default=0.67)
    parser.add_argument("--mechanism-dual-archive-enabled", type=int, default=1)
    parser.add_argument("--mechanism-counterfactual-enabled", type=int, default=1)
    parser.add_argument("--mechanism-counterfactual-noise", type=float, default=0.05)
    parser.add_argument("--mechanism-counterfactual-max-sensitivity", type=float, default=0.50)
    parser.add_argument("--graph-cache-enabled", type=int, default=1)
    parser.add_argument("--graph-cache-backend", type=str, default="sqlite", choices=["memory", "sqlite"])
    parser.add_argument("--graph-cache-db-path", type=str, default="")
    parser.add_argument("--graph-cache-namespace", type=str, default="work_ci_subset_bridge")
    parser.add_argument("--graph-cache-persist-values", type=int, default=0)
    parser.add_argument("--interval-alpha", type=float, default=0.1, help="Two-sided symmetric interval alpha (target coverage=1-alpha).")
    args = parser.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = ROOT / "examples" / "out" / f"nsgablack_symbolic_subset_bridge_work_ci_{stamp}"
    out_root.mkdir(parents=True, exist_ok=True)

    reader = WorkCiIntervalReader(
        csv_path=str(args.csv_path),
        target_col=str(args.target_col),
        test_fold_col=str(args.test_fold_col),
    )
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

    gate_names = (
        "is_holiday_day_or_window",
        "is_holiday_near",
        "is_holiday_mid",
        "is_nonwork_weekend",
    )
    gate_idx_list = [feature_names.index(nm) for nm in gate_names if nm in feature_names]
    strict4_gate_idx: tuple[int, int, int, int] | None = None
    strict4_enabled = bool(args.strict4_branch_mode)
    if strict4_enabled:
        if len(gate_idx_list) != 4:
            strict4_enabled = False
        else:
            strict4_gate_idx = (
                int(gate_idx_list[0]),
                int(gate_idx_list[1]),
                int(gate_idx_list[2]),
                int(gate_idx_list[3]),
            )

    batched_eval_enabled = bool(int(args.batched_eval))
    reinvest_enabled = bool(int(args.reinvest_search))
    effective_pop_size = int(max(4, int(args.pop_size)))
    effective_generations = int(max(1, int(args.generations)))
    effective_strict4_workers = int(max(1, int(args.strict4_branch_parallel_workers)))
    if batched_eval_enabled and reinvest_enabled:
        effective_pop_size = int(max(effective_pop_size, round(effective_pop_size * float(max(1.0, args.reinvest_pop_mult)))))
        effective_generations = int(
            max(effective_generations, round(effective_generations * float(max(1.0, args.reinvest_gen_mult))))
        )
        if strict4_enabled:
            effective_strict4_workers = int(
                max(
                    effective_strict4_workers,
                    round(effective_strict4_workers * float(max(1.0, args.reinvest_strict4_workers_mult))),
                )
            )
    effective_vns_batch_size = int(max(4, int(args.vns_batch_size), effective_pop_size))

    dynamic_pool_enabled = bool(int(args.dynamic_pool_enabled))
    dynamic_pool_epochs = int(max(1, args.dynamic_pool_epochs))
    dynamic_init_minimal = bool(int(args.dynamic_init_minimal))
    dynamic_expand_max_new = int(max(1, args.dynamic_expand_max_new))
    dynamic_focus_top_features = int(max(2, args.dynamic_focus_top_features))
    dynamic_partner_topk = int(max(2, args.dynamic_partner_topk))
    dynamic_top_cache_use = int(max(5, args.dynamic_top_cache_use))
    dynamic_max_pool_size = int(max(32, args.dynamic_max_pool_size))
    mechanism_info_gain_enabled = bool(int(args.mechanism_info_gain_enabled))
    mechanism_info_gain_min_corr = float(max(0.0, args.mechanism_info_gain_min_corr))
    mechanism_novelty_enabled = bool(int(args.mechanism_novelty_enabled))
    mechanism_novelty_max_corr = float(np.clip(args.mechanism_novelty_max_corr, 0.0, 0.999999))
    mechanism_curriculum_enabled = bool(int(args.mechanism_curriculum_enabled))
    mechanism_curriculum_stage1_ratio = float(np.clip(args.mechanism_curriculum_stage1_ratio, 0.0, 1.0))
    mechanism_curriculum_stage2_ratio = float(np.clip(args.mechanism_curriculum_stage2_ratio, 0.0, 1.0))
    mechanism_dual_archive_enabled = bool(int(args.mechanism_dual_archive_enabled))
    mechanism_counterfactual_enabled = bool(int(args.mechanism_counterfactual_enabled))
    mechanism_counterfactual_noise = float(max(1e-6, args.mechanism_counterfactual_noise))
    mechanism_counterfactual_max_sensitivity = float(max(0.0, args.mechanism_counterfactual_max_sensitivity))

    graph_cache_enabled = bool(int(args.graph_cache_enabled))
    graph_cache_backend = str(args.graph_cache_backend).strip().lower()
    graph_cache_db_path = str(args.graph_cache_db_path).strip()
    if graph_cache_enabled and graph_cache_backend == "sqlite" and not graph_cache_db_path:
        graph_cache_db_path = str((ROOT / ".mlblack_cache" / "work_ci_subset_expression_graph_cache.sqlite3"))
    graph_cache = ExpressionGraphCache(
        enabled=bool(graph_cache_enabled),
        backend=str(graph_cache_backend),
        db_path=str(graph_cache_db_path),
        namespace=str(args.graph_cache_namespace),
        persist_values=bool(int(args.graph_cache_persist_values)),
    )

    candidates = _build_candidate_pool(
        X_train,
        y_train,
        feature_names=feature_names,
        topk_for_pairs=6,
        include_pair_interactions=bool(not dynamic_init_minimal),
        include_gradient_enrich=bool(not dynamic_init_minimal),
    )

    if not dynamic_pool_enabled:
        candidates = _build_candidate_pool(
            X_train,
            y_train,
            feature_names=feature_names,
            topk_for_pairs=6,
            include_pair_interactions=True,
            include_gradient_enrich=True,
        )

    def _run_outer_once(
        *,
        run_candidates: Sequence[CandidateTerm],
        generations_this_epoch: int,
        seed_this_epoch: int,
    ) -> tuple[SymbolicSubsetSelectionProblem, dict[str, Any], dict[str, Any], float]:
        problem_local = SymbolicSubsetSelectionProblem(
            X_fit=X_train,
            y_fit=y_train,
            candidates=run_candidates,
            max_terms=int(max(2, args.max_terms)),
            ridge_l2=float(max(0.0, args.ridge_l2)),
            rolling_folds=int(max(1, args.rolling_folds)),
            rolling_val_ratio=float(np.clip(args.rolling_val_ratio, 0.05, 0.45)),
            min_train=max(256, int(round(0.4 * X_train.shape[0]))),
            strict4_branch_mode=bool(strict4_enabled),
            strict4_gate_idx=strict4_gate_idx,
            strict4_min_branch_train=int(max(8, args.strict4_min_branch_train)),
            strict4_branch_parallel_workers=int(effective_strict4_workers),
            inner_opt_enabled=bool(int(args.inner_opt_enabled)),
            inner_opt_adam_steps=int(max(0, args.inner_opt_adam_steps)),
            inner_opt_adam_lr=float(max(1e-8, args.inner_opt_adam_lr)),
            inner_opt_lbfgs_steps=int(max(0, args.inner_opt_lbfgs_steps)),
            inner_opt_lbfgs_lr=float(max(1e-8, args.inner_opt_lbfgs_lr)),
            inner_opt_accept_rmse_tol=float(max(0.0, args.inner_opt_accept_rmse_tol)),
            inner_opt_accept_rel_tol=float(max(0.0, args.inner_opt_accept_rel_tol)),
            inner_opt_guard_patience=int(max(1, args.inner_opt_guard_patience)),
            inner_opt_guard_check_interval=int(max(1, args.inner_opt_guard_check_interval)),
            inner_opt_alt_freeze_readout=bool(int(args.inner_opt_alt_freeze_readout)),
            inner_opt_grad_clip_norm=float(max(0.0, args.inner_opt_grad_clip_norm)),
            inner_opt_residual_clip_q=float(np.clip(args.inner_opt_residual_clip_q, 0.70, 0.999)),
            graph_cache=graph_cache,
        )

        outer_adapter_local, outer_meta_local = _build_outer_adapter(
            strategy=str(args.outer_strategy),
            pop_size=int(effective_pop_size),
            generations=int(max(1, generations_this_epoch)),
            portfolio_phases_csv=str(args.portfolio_phases),
            portfolio_weights_csv=str(args.portfolio_phase_weights),
            moead_neighborhood_size=int(max(2, args.moead_neighborhood_size)),
            moead_delta=float(args.moead_delta),
            moead_nr=int(max(1, args.moead_nr)),
            vns_k_max=int(max(1, args.vns_k_max)),
            vns_batch_size=int(effective_vns_batch_size),
        )

        low, high = _bounds_arrays(problem_local)
        rep_local = RepresentationPipeline(
            initializer=UniformInitializer(low=low, high=high),
            mutator=ContextGaussianMutation(
                base_sigma=0.18,
                sigma_key="mutation_sigma",
                low=low,
                high=high,
            ),
            repair=ClipRepair(low=low, high=high),
        )
        solver_local = ComposableSolver(
            problem_local,
            adapter=outer_adapter_local,
            representation_pipeline=rep_local,
        )
        if batched_eval_enabled:
            def _evaluate_population_batched(self: Any, population: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
                pop_arr = np.asarray(population, dtype=float)
                if pop_arr.ndim == 1:
                    pop_arr = pop_arr.reshape(1, -1)
                pop_size = int(pop_arr.shape[0])
                if bool(getattr(self, "snapshot_pre_evaluate_population", False)):
                    self._persist_snapshot(
                        population=pop_arr,
                        objectives=None,
                        violations=None,
                        include_pareto=True,
                        include_history=True,
                        include_decision_trace=True,
                        complete=False,
                    )
                objectives, violations = problem_local.evaluate_population_batch(pop_arr)
                self.evaluation_count += int(pop_size)
                self._persist_snapshot(
                    population=pop_arr,
                    objectives=objectives,
                    violations=violations,
                    include_pareto=True,
                    include_history=True,
                    include_decision_trace=True,
                    complete=True,
                )
                return objectives, violations

            solver_local.evaluate_population = types.MethodType(_evaluate_population_batched, solver_local)

        solver_local.max_steps = int(max(1, outer_meta_local.get("max_generations", generations_this_epoch)))
        solver_local.set_random_seed(int(seed_this_epoch))
        t0_local = time.perf_counter()
        run_local = solver_local.run()
        outer_sec_local = float(time.perf_counter() - t0_local)
        return problem_local, outer_meta_local, run_local, outer_sec_local

    epoch_generations: list[int] = [int(effective_generations)]
    if dynamic_pool_enabled and dynamic_pool_epochs > 1:
        base = int(effective_generations // dynamic_pool_epochs)
        rem = int(effective_generations - base * dynamic_pool_epochs)
        epoch_generations = [int(max(1, base + (1 if i < rem else 0))) for i in range(dynamic_pool_epochs)]

    outer_sec = 0.0
    outer_meta: dict[str, Any] = {"strategy": str(args.outer_strategy), "max_generations": int(effective_generations)}
    run: dict[str, Any] = {"status": "completed", "steps_executed": 0}
    top_cache: list[dict[str, Any]] = []
    problem: SymbolicSubsetSelectionProblem | None = None
    best_row: dict[str, Any] | None = None
    best_genome: list[dict[str, Any]] | None = None
    best_k = 0
    dynamic_epoch_logs: list[dict[str, Any]] = []
    total_epochs = int(max(1, len(epoch_generations)))

    for ep, gens_this in enumerate(epoch_generations):
        problem_ep, outer_meta_ep, run_ep, sec_ep = _run_outer_once(
            run_candidates=candidates,
            generations_this_epoch=int(gens_this),
            seed_this_epoch=int(args.seed + ep),
        )
        problem = problem_ep
        outer_sec += float(sec_ep)
        run = dict(run_ep)
        run["steps_executed"] = int(run.get("steps_executed", 0)) + int(sum(epoch_generations[:ep]))
        outer_meta = dict(outer_meta_ep)
        top_cache_ep = problem_ep.cache_top(topn=max(50, dynamic_top_cache_use))
        if not top_cache_ep:
            continue
        top_cache = list(top_cache_ep)

        row0 = dict(top_cache_ep[0])
        idx0 = [int(v) for v in row0.get("subset_idx", [])]
        genome0 = [{"name": candidates[i].name, "expr": dict(candidates[i].expr)} for i in idx0]
        if best_row is None:
            best_row = dict(row0)
            best_genome = list(genome0)
            best_k = int(row0.get("subset_size", len(idx0)))
        else:
            cur_key = (
                float(row0.get("obj_accuracy", float("inf"))),
                float(row0.get("obj_stability", float("inf"))),
                float(row0.get("obj_complexity", float("inf"))),
            )
            best_key = (
                float(best_row.get("obj_accuracy", float("inf"))),
                float(best_row.get("obj_stability", float("inf"))),
                float(best_row.get("obj_complexity", float("inf"))),
            )
            if cur_key < best_key:
                best_row = dict(row0)
                best_genome = list(genome0)
                best_k = int(row0.get("subset_size", len(idx0)))

        selected_keys: set[str] = set()
        archive_rows = list(top_cache_ep[:dynamic_top_cache_use])
        if bool(mechanism_dual_archive_enabled):
            top_acc = sorted(top_cache_ep, key=lambda rr: float(rr.get("obj_accuracy", float("inf"))))[: max(5, dynamic_top_cache_use // 2)]
            top_stb = sorted(top_cache_ep, key=lambda rr: float(rr.get("obj_stability", float("inf"))))[: max(5, dynamic_top_cache_use // 2)]
            seen_rows: set[str] = set()
            merged_rows: list[dict[str, Any]] = []
            for rr in list(top_acc) + list(top_stb):
                key_rr = json.dumps([int(v) for v in rr.get("subset_idx", [])])
                if key_rr in seen_rows:
                    continue
                seen_rows.add(key_rr)
                merged_rows.append(rr)
            archive_rows = merged_rows[:dynamic_top_cache_use]

        for r in archive_rows:
            for j in [int(v) for v in r.get("subset_idx", [])]:
                if 0 <= int(j) < len(candidates):
                    selected_keys.add(json.dumps(candidates[int(j)].expr, sort_keys=True))

        n_new = 0
        n_after_prune = len(candidates)
        if dynamic_pool_enabled and ep < len(epoch_generations) - 1 and idx0:
            l2_ep = float(max(0.0, row0.get("tuned_l2", args.ridge_l2)))
            fit_ep = evaluate_genome_with_ridge(
                genome0,
                X_train=X_train,
                y_train=y_train,
                X_eval=X_train,
                y_eval=y_train,
                l2=l2_ep,
            )
            pred_tr = _as_2d(np.asarray(fit_ep.get("pred_train"), dtype=float))
            res_tr = _as_2d(np.asarray(y_train - pred_tr, dtype=float))
            # Curriculum staging: progressively unlock harder operators.
            stage_ratio = float((ep + 1) / max(1, total_epochs))
            stage1 = float(np.clip(mechanism_curriculum_stage1_ratio, 0.0, 1.0))
            stage2 = float(np.clip(mechanism_curriculum_stage2_ratio, 0.0, 1.0))
            if stage2 < stage1:
                stage2 = stage1
            if bool(mechanism_curriculum_enabled):
                enable_piecewise = bool(stage_ratio >= stage1)
                enable_dyn = bool(stage_ratio >= stage2)
                enable_gate = bool(stage_ratio >= stage2)
            else:
                enable_piecewise = True
                enable_dyn = True
                enable_gate = True

            new_terms = _expand_candidate_pool_from_residual(
                X=X_train,
                y_residual=res_tr,
                feature_names=feature_names,
                base_genome=genome0,
                base_weight=_as_2d(np.asarray(fit_ep.get("weight"), dtype=float)),
                existing=candidates,
                max_new_terms=int(dynamic_expand_max_new),
                focus_top_features=int(dynamic_focus_top_features),
                partner_topk=int(dynamic_partner_topk),
                enable_piecewise=bool(enable_piecewise),
                enable_dynamic_interactions=bool(enable_dyn),
                enable_gate_interactions=bool(enable_gate),
                info_gain_enabled=bool(mechanism_info_gain_enabled),
                info_gain_min_abs_corr=float(mechanism_info_gain_min_corr),
                novelty_enabled=bool(mechanism_novelty_enabled),
                novelty_max_abs_corr=float(mechanism_novelty_max_corr),
                counterfactual_enabled=bool(mechanism_counterfactual_enabled),
                counterfactual_noise_scale=float(mechanism_counterfactual_noise),
                counterfactual_max_sensitivity=float(mechanism_counterfactual_max_sensitivity),
            )
            n_new = int(len(new_terms))
            if n_new > 0:
                candidates = list(candidates) + list(new_terms)
            candidates = _prune_candidate_pool(
                candidates=candidates,
                keep_expr_keys=selected_keys,
                feature_names=feature_names,
                max_pool_size=int(dynamic_max_pool_size),
            )
            n_after_prune = int(len(candidates))

        dynamic_epoch_logs.append(
            {
                "epoch": int(ep + 1),
                "generations": int(gens_this),
                "duration_sec": float(sec_ep),
                "pool_size_before": int(len(problem_ep.candidates)),
                "pool_size_after": int(n_after_prune),
                "new_terms_added": int(n_new),
                "best_obj_accuracy": float(row0.get("obj_accuracy", float("inf"))),
                "best_subset_size": int(row0.get("subset_size", len(idx0))),
                "curriculum_stage_ratio": float((ep + 1) / max(1, total_epochs)),
                "dual_archive_rows": int(len(archive_rows)),
            }
        )

    if problem is None or best_row is None or best_genome is None:
        raise RuntimeError("outer search produced empty evaluation cache")

    best_decode_meta = {
        "rmse_mean": float(best_row.get("rmse_mean", float("inf"))),
        "rmse_std": float(best_row.get("rmse_std", float("inf"))),
        "obj_accuracy": float(best_row.get("obj_accuracy", float("inf"))),
        "obj_stability": float(best_row.get("obj_stability", float("inf"))),
        "obj_complexity": float(best_row.get("obj_complexity", float("inf"))),
        "decode_meta": _jsonable(best_row.get("decode_meta", {})),
        "tuned_l2": float(best_row.get("tuned_l2", max(0.0, args.ridge_l2))),
        "strict4_min_train_ratio": float(best_row.get("strict4_min_train_ratio", 0.08)),
    }
    best_subset_idx = [int(v) for v in best_row.get("subset_idx", [])]

    fit_final = _three_layer_fit_predict(
        genome=best_genome,
        X_train=X_train,
        y_train=y_train,
        X_eval=X_test,
        y_eval=y_test,
        l2=float(max(0.0, best_decode_meta["tuned_l2"])),
        inner_opt_enabled=bool(int(args.inner_opt_enabled)),
        inner_opt_adam_steps=int(max(0, args.inner_opt_adam_steps)),
        inner_opt_adam_lr=float(max(1e-8, args.inner_opt_adam_lr)),
        inner_opt_lbfgs_steps=int(max(0, args.inner_opt_lbfgs_steps)),
        inner_opt_lbfgs_lr=float(max(1e-8, args.inner_opt_lbfgs_lr)),
        inner_opt_accept_rmse_tol=float(max(0.0, args.inner_opt_accept_rmse_tol)),
        inner_opt_accept_rel_tol=float(max(0.0, args.inner_opt_accept_rel_tol)),
        inner_opt_guard_patience=int(max(1, args.inner_opt_guard_patience)),
        inner_opt_guard_check_interval=int(max(1, args.inner_opt_guard_check_interval)),
        inner_opt_alt_freeze_readout=bool(int(args.inner_opt_alt_freeze_readout)),
        inner_opt_grad_clip_norm=float(max(0.0, args.inner_opt_grad_clip_norm)),
        inner_opt_residual_clip_q=float(np.clip(args.inner_opt_residual_clip_q, 0.70, 0.999)),
    )
    sym_pred_train = _as_2d(np.asarray(fit_final.get("pred_train"), dtype=float))
    sym_pred_test = _as_2d(np.asarray(fit_final.get("pred_eval"), dtype=float))
    sym_rmse = float(_rmse(y_test, sym_pred_test))
    sym_mae = float(_mae(y_test, sym_pred_test))
    interval_alpha = float(np.clip(args.interval_alpha, 1e-6, 0.99))
    sym_lo, sym_hi, sym_calib_q = _build_symmetric_interval(
        y_train=y_train,
        pred_train=sym_pred_train,
        pred_eval=sym_pred_test,
        alpha=interval_alpha,
    )
    sym_interval = _interval_metrics(
        y_true=y_test,
        lower=sym_lo,
        upper=sym_hi,
        alpha=interval_alpha,
    )

    xgb = XGBoostSurrogateTrainer(
        config=XGBoostTrainerConfig(
            artifact_id="subset_bridge_xgb_baseline",
            n_estimators=360,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            tree_method="hist",
            random_seed=42,
        )
    )
    xgb_art = xgb.fit(
        ProcessedDataset(
            X_train=X_train,
            y_train=y_train,
            feature_names=feature_names,
            target_names=(str(args.target_col),),
        )
    )
    xgb_pred = np.asarray(xgb_art.predict(X_test), dtype=float).reshape(-1, 1)
    xgb_rmse = _rmse(y_test, xgb_pred)
    xgb_mae = _mae(y_test, xgb_pred)
    xgb_train_pred = np.asarray(xgb_art.predict(X_train), dtype=float).reshape(-1, 1)
    xgb_lo, xgb_hi, xgb_calib_q = _build_symmetric_interval(
        y_train=y_train,
        pred_train=xgb_train_pred,
        pred_eval=xgb_pred,
        alpha=interval_alpha,
    )
    xgb_interval = _interval_metrics(
        y_true=y_test,
        lower=xgb_lo,
        upper=xgb_hi,
        alpha=interval_alpha,
    )

    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "out_root": str(out_root),
        "config": {
            "csv_path": str(args.csv_path),
            "target_col": str(args.target_col),
            "test_fold_col": str(args.test_fold_col),
            "pop_size": int(args.pop_size),
            "generations": int(args.generations),
            "effective_pop_size": int(effective_pop_size),
            "effective_generations": int(effective_generations),
            "rolling_folds": int(args.rolling_folds),
            "rolling_val_ratio": float(args.rolling_val_ratio),
            "max_terms": int(args.max_terms),
            "ridge_l2": float(args.ridge_l2),
            "strict4_branch_mode_requested": bool(args.strict4_branch_mode),
            "strict4_branch_mode_enabled": bool(strict4_enabled),
            "strict4_min_branch_train": int(args.strict4_min_branch_train),
            "strict4_branch_parallel_workers": int(args.strict4_branch_parallel_workers),
            "effective_strict4_branch_parallel_workers": int(effective_strict4_workers),
            "seed": int(args.seed),
            "outer_strategy": str(args.outer_strategy),
            "portfolio_phases": str(args.portfolio_phases),
            "portfolio_phase_weights": str(args.portfolio_phase_weights),
            "moead_neighborhood_size": int(args.moead_neighborhood_size),
            "moead_delta": float(args.moead_delta),
            "moead_nr": int(args.moead_nr),
            "vns_k_max": int(args.vns_k_max),
            "vns_batch_size": int(args.vns_batch_size),
            "effective_vns_batch_size": int(effective_vns_batch_size),
            "batched_eval_enabled": bool(batched_eval_enabled),
            "reinvest_search": bool(reinvest_enabled),
            "reinvest_pop_mult": float(args.reinvest_pop_mult),
            "reinvest_gen_mult": float(args.reinvest_gen_mult),
            "reinvest_strict4_workers_mult": float(args.reinvest_strict4_workers_mult),
            "dynamic_pool_enabled": bool(dynamic_pool_enabled),
            "dynamic_pool_epochs": int(dynamic_pool_epochs),
            "dynamic_init_minimal": bool(dynamic_init_minimal),
            "dynamic_expand_max_new": int(dynamic_expand_max_new),
            "dynamic_focus_top_features": int(dynamic_focus_top_features),
            "dynamic_partner_topk": int(dynamic_partner_topk),
            "dynamic_top_cache_use": int(dynamic_top_cache_use),
            "dynamic_max_pool_size": int(dynamic_max_pool_size),
            "mechanism_info_gain_enabled": bool(mechanism_info_gain_enabled),
            "mechanism_info_gain_min_corr": float(mechanism_info_gain_min_corr),
            "mechanism_novelty_enabled": bool(mechanism_novelty_enabled),
            "mechanism_novelty_max_corr": float(mechanism_novelty_max_corr),
            "mechanism_curriculum_enabled": bool(mechanism_curriculum_enabled),
            "mechanism_curriculum_stage1_ratio": float(mechanism_curriculum_stage1_ratio),
            "mechanism_curriculum_stage2_ratio": float(mechanism_curriculum_stage2_ratio),
            "mechanism_dual_archive_enabled": bool(mechanism_dual_archive_enabled),
            "mechanism_counterfactual_enabled": bool(mechanism_counterfactual_enabled),
            "mechanism_counterfactual_noise": float(mechanism_counterfactual_noise),
            "mechanism_counterfactual_max_sensitivity": float(mechanism_counterfactual_max_sensitivity),
            "graph_cache_enabled": bool(graph_cache_enabled),
            "graph_cache_backend": str(graph_cache_backend),
            "graph_cache_db_path": str(graph_cache_db_path),
            "graph_cache_namespace": str(args.graph_cache_namespace),
            "graph_cache_persist_values": bool(int(args.graph_cache_persist_values)),
            "interval_alpha": float(interval_alpha),
            "inner_opt_enabled": bool(int(args.inner_opt_enabled)),
            "inner_opt_adam_steps": int(args.inner_opt_adam_steps),
            "inner_opt_adam_lr": float(args.inner_opt_adam_lr),
            "inner_opt_lbfgs_steps": int(args.inner_opt_lbfgs_steps),
            "inner_opt_lbfgs_lr": float(args.inner_opt_lbfgs_lr),
            "inner_opt_accept_rmse_tol": float(args.inner_opt_accept_rmse_tol),
            "inner_opt_accept_rel_tol": float(args.inner_opt_accept_rel_tol),
            "inner_opt_guard_patience": int(args.inner_opt_guard_patience),
            "inner_opt_guard_check_interval": int(args.inner_opt_guard_check_interval),
            "inner_opt_alt_freeze_readout": bool(int(args.inner_opt_alt_freeze_readout)),
            "inner_opt_grad_clip_norm": float(args.inner_opt_grad_clip_norm),
            "inner_opt_residual_clip_q": float(args.inner_opt_residual_clip_q),
            "outer_decision_encoding": "expanded_structure_plus_hyperparams_v1",
            "nesting_architecture": "outer_structure_search -> middle_param_optimizer -> inner_rolling_eval",
        },
        "dataset": {
            "n_train": int(X_train.shape[0]),
            "n_test": int(X_test.shape[0]),
            "n_features": int(X_train.shape[1]),
            "feature_names": list(feature_names),
        },
        "outer_search": {
            "duration_sec": float(outer_sec),
            "outer_meta": _jsonable({**dict(outer_meta), "dynamic_pool_epochs": dynamic_epoch_logs}),
            "n_candidates": int(len(candidates)),
            "n_families": int(len(problem.families)),
            "families": [str(v) for v in problem.families],
            "run_result": _jsonable(run),
            "n_cached_evals": int(len(problem._cache)),
            "top_cache": top_cache[:20],
            "graph_cache": _jsonable(graph_cache.snapshot()),
        },
        "best_solution": {
            "k_decoded": int(best_k),
            "subset_size": int(len(best_subset_idx)),
            "subset_idx": [int(i) for i in best_subset_idx],
            "subset_names": _jsonable(best_row.get("subset_names", [])),
            "subset_families": _jsonable(best_row.get("subset_families", [])),
            "decode_meta": _jsonable(best_decode_meta),
            "obj_accuracy": float(best_row.get("obj_accuracy", float("inf"))),
            "obj_stability": float(best_row.get("obj_stability", float("inf"))),
            "obj_complexity": float(best_row.get("obj_complexity", float("inf"))),
            "inner_opt_info": _jsonable(fit_final.get("inner_opt_info", {})),
        },
        "test_compare": {
            "symbolic_subset_rmse": float(sym_rmse),
            "symbolic_subset_mae": float(sym_mae),
            "xgboost_rmse": float(xgb_rmse),
            "xgboost_mae": float(xgb_mae),
            "delta_symbolic_minus_xgb": float(sym_rmse - xgb_rmse),
            "interval_metrics": {
                "symbolic": {
                    "calib_abs_residual_q": float(sym_calib_q),
                    **_jsonable(sym_interval),
                },
                "xgboost": {
                    "calib_abs_residual_q": float(xgb_calib_q),
                    **_jsonable(xgb_interval),
                },
            },
        },
    }
    report_path = out_root / "summary.json"
    report_path.write_text(json.dumps(_jsonable(report), ensure_ascii=False, indent=2), encoding="utf-8")
    graph_cache.close()

    print("NSGABLACK_SYMBOLIC_SUBSET_BRIDGE_DONE")
    print(f"summary={report_path}")
    print(
        "rmse: "
        f"symbolic_subset={float(sym_rmse):.6f}, "
        f"xgboost={float(xgb_rmse):.6f}, "
        f"delta={float(sym_rmse - xgb_rmse):.6f}"
    )
    print(
        "interval: "
        f"alpha={float(interval_alpha):.3f} | "
        f"symbolic(PICP={float(sym_interval['picp']):.4f}, PINAW={float(sym_interval['pinaw']):.4f}, IS={float(sym_interval['interval_score']):.4f}) | "
        f"xgb(PICP={float(xgb_interval['picp']):.4f}, PINAW={float(xgb_interval['pinaw']):.4f}, IS={float(xgb_interval['interval_score']):.4f})"
    )


if __name__ == "__main__":
    main()
