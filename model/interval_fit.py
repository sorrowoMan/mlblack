from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from core.models.symbolic_torch_model import SymbolicTorchRegressor
from core.symbolic.symbolic_dsl import expression_to_string
from core.symbolic.symbolic_structure_search import evaluate_genome_with_ridge

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


def _parse_int_list_csv(text: str, *, default: Sequence[int]) -> list[int]:
    vals: list[int] = []
    for s in str(text).split(","):
        ss = s.strip()
        if not ss:
            continue
        try:
            vals.append(int(ss))
        except Exception:
            continue
    out = [int(v) for v in vals if int(v) > 0]
    if not out:
        out = [int(v) for v in default if int(v) > 0]
    out = sorted(set(out))
    return out


def _parse_float_list_csv(text: str, *, default: Sequence[float]) -> list[float]:
    vals: list[float] = []
    for s in str(text).split(","):
        ss = s.strip()
        if not ss:
            continue
        try:
            vals.append(float(ss))
        except Exception:
            continue
    out = [float(v) for v in vals if np.isfinite(v)]
    if not out:
        out = [float(v) for v in default if np.isfinite(v)]
    out = sorted(set(out))
    return out


def _make_lag_from_history(train_series: np.ndarray, test_series: np.ndarray, lag: int) -> tuple[np.ndarray, np.ndarray]:
    tr = np.asarray(train_series, dtype=float).reshape(-1)
    te = np.asarray(test_series, dtype=float).reshape(-1)
    ntr = int(tr.size)
    nte = int(te.size)
    l = int(max(1, lag))
    full = np.concatenate([tr, te], axis=0)
    out = np.empty_like(full, dtype=float)
    out[:l] = np.nan
    out[l:] = full[:-l]
    tr_lag = np.asarray(out[:ntr], dtype=float)
    te_lag = np.asarray(out[ntr : ntr + nte], dtype=float)
    # deterministic fill for earliest rows only
    fill = float(tr[0]) if ntr > 0 else 0.0
    tr_lag = np.where(np.isfinite(tr_lag), tr_lag, fill)
    te_lag = np.where(np.isfinite(te_lag), te_lag, fill)
    return tr_lag, te_lag


def _fmt_float(v: float, *, precision: int = 6) -> str:
    if not np.isfinite(v):
        return str(v)
    return f"{float(v):.{int(max(1, precision))}g}"


def _stable_text_hash(text: str) -> int:
    acc = 0
    for ch in str(text):
        acc = (acc * 131 + ord(ch)) % 2147483629
    return int(acc)


def _derive_fit_seed(base_seed: int | None, genome: Sequence[Mapping[str, Any]]) -> int | None:
    if base_seed is None:
        return None
    signature_parts: list[str] = []
    for i, term in enumerate(genome):
        name = str(term.get("name", f"term_{i}"))
        expr = term.get("expr", term)
        try:
            expr_text = str(expression_to_string(expr, precision=6))
        except Exception:
            expr_text = str(expr)
        signature_parts.append(f"{name}:{expr_text}")
    signature = "|".join(signature_parts)
    return int((int(base_seed) + _stable_text_hash(signature)) % 2147483647)


def _build_formula_summary(
    *,
    genome: Sequence[Mapping[str, Any]],
    weight: np.ndarray,
    bias: np.ndarray,
    precision: int = 6,
) -> dict[str, Any]:
    w = np.asarray(weight, dtype=float)
    b = np.asarray(bias, dtype=float).reshape(-1)

    if w.ndim == 2:
        if w.shape[1] >= 1:
            w1 = np.asarray(w[:, 0], dtype=float).reshape(-1)
        else:
            w1 = np.asarray(w.reshape(-1), dtype=float)
    else:
        w1 = np.asarray(w.reshape(-1), dtype=float)
    if w1.size < len(genome):
        w1 = np.pad(w1, (0, int(len(genome) - w1.size)), constant_values=0.0)
    elif w1.size > len(genome):
        w1 = w1[: len(genome)]

    intercept = float(b[0]) if b.size > 0 else 0.0
    items: list[dict[str, Any]] = []
    expr_parts: list[str] = [f"{_fmt_float(intercept, precision=precision)}"]
    for i, term in enumerate(genome):
        wi = float(w1[i])
        expr = term.get("expr", term)
        try:
            expr_text = str(expression_to_string(expr, precision=max(4, precision)))
        except Exception:
            expr_text = str(term.get("name", f"term_{i}"))
        items.append(
            {
                "index": int(i),
                "name": str(term.get("name", f"term_{i}")),
                "expression": str(expr_text),
                "coefficient": float(wi),
            }
        )
        sign = "+" if wi >= 0.0 else "-"
        expr_parts.append(f" {sign} {_fmt_float(abs(wi), precision=precision)}*({expr_text})")
    return {
        "formula_intercept": float(intercept),
        "formula_terms": items,
        "formula_human_readable": "".join(expr_parts),
    }


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


def _conformal_q(scores: np.ndarray, alpha: float) -> float:
    s = np.asarray(scores, dtype=float).reshape(-1)
    s = s[np.isfinite(s)]
    if s.size <= 0:
        return 0.0
    a = float(np.clip(alpha, 1e-6, 0.99))
    # finite-sample conformal quantile: ceil((n+1)*(1-alpha))/n
    n = int(s.size)
    k = int(np.ceil((n + 1) * (1.0 - a)))
    k = int(min(max(1, k), n))
    s_sorted = np.sort(s)
    return float(s_sorted[k - 1])


def _build_native_quantile_interval(
    *,
    genome: Sequence[Mapping[str, Any]],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    alpha: float,
    calib_ratio: float,
    quantile_l2: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    xtr = _as_2d(np.asarray(X_train, dtype=float))
    ytr = _as_2d(np.asarray(y_train, dtype=float)).reshape(-1)
    xev = _as_2d(np.asarray(X_eval, dtype=float))
    a = float(np.clip(alpha, 1e-6, 0.99))
    ql = float(a / 2.0)
    qu = float(1.0 - a / 2.0)

    # build symbolic design matrix once
    phi_tr = np.asarray(evaluate_genome_numpy(genome, xtr), dtype=float)
    phi_ev = np.asarray(evaluate_genome_numpy(genome, xev), dtype=float)

    n = int(phi_tr.shape[0])
    cal_n = int(max(32, min(int(round(float(np.clip(calib_ratio, 0.05, 0.4)) * n)), n // 3)))
    fit_n = int(max(64, n - cal_n))
    fit_idx = np.arange(0, fit_n, dtype=int)
    cal_idx = np.arange(fit_n, n, dtype=int)

    info: dict[str, Any] = {
        "method": "native_quantile_cqr",
        "quantile_low": float(ql),
        "quantile_high": float(qu),
        "fit_size": int(fit_idx.size),
        "calib_size": int(cal_idx.size),
        "quantile_l2": float(max(0.0, quantile_l2)),
    }

    try:
        from sklearn.linear_model import QuantileRegressor
    except Exception as exc:
        lo, hi, q = _build_symmetric_interval(
            y_train=ytr.reshape(-1, 1),
            pred_train=np.zeros((n, 1), dtype=float),
            pred_eval=np.zeros((xev.shape[0], 1), dtype=float),
            alpha=a,
        )
        info["fallback"] = "symmetric_no_sklearn"
        info["fallback_error"] = f"{type(exc).__name__}: {exc}"
        return lo, hi, info

    try:
        model_lo = QuantileRegressor(quantile=ql, alpha=float(max(0.0, quantile_l2)), fit_intercept=True, solver="highs")
        model_hi = QuantileRegressor(quantile=qu, alpha=float(max(0.0, quantile_l2)), fit_intercept=True, solver="highs")
        model_lo.fit(phi_tr[fit_idx], ytr[fit_idx])
        model_hi.fit(phi_tr[fit_idx], ytr[fit_idx])

        lo_cal = np.asarray(model_lo.predict(phi_tr[cal_idx]), dtype=float).reshape(-1)
        hi_cal = np.asarray(model_hi.predict(phi_tr[cal_idx]), dtype=float).reshape(-1)
        y_cal = np.asarray(ytr[cal_idx], dtype=float).reshape(-1)
        scores = np.maximum(np.maximum(lo_cal - y_cal, y_cal - hi_cal), 0.0)
        qhat = _conformal_q(scores, a)

        lo_ev = np.asarray(model_lo.predict(phi_ev), dtype=float).reshape(-1) - qhat
        hi_ev = np.asarray(model_hi.predict(phi_ev), dtype=float).reshape(-1) + qhat
        lo_ev, hi_ev = np.minimum(lo_ev, hi_ev), np.maximum(lo_ev, hi_ev)

        info["conformal_qhat"] = float(qhat)
        info["fallback"] = ""
        return lo_ev, hi_ev, info
    except Exception as exc:
        # safe fallback: symbolic ridge residual interval
        seed = evaluate_genome_with_ridge(
            genome,
            X_train=xtr,
            y_train=ytr.reshape(-1, 1),
            X_eval=xev,
            y_eval=None,
            l2=1e-4,
        )
        lo, hi, q = _build_symmetric_interval(
            y_train=ytr.reshape(-1, 1),
            pred_train=_as_2d(np.asarray(seed.get("pred_train"), dtype=float)),
            pred_eval=_as_2d(np.asarray(seed.get("pred_eval"), dtype=float)),
            alpha=a,
        )
        info["fallback"] = "symmetric_after_quantile_failure"
        info["fallback_error"] = f"{type(exc).__name__}: {exc}"
        info["conformal_qhat"] = float(q)
        return lo, hi, info


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
    random_seed: int | None = None,
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
        fit_seed = _derive_fit_seed(random_seed, genome)
        if fit_seed is not None:
            torch.manual_seed(int(fit_seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(fit_seed))
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


