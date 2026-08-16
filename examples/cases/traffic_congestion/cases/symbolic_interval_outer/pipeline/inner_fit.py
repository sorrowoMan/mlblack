from __future__ import annotations

from ..runtime.legacy_imports import *
from ..evaluation.metrics import *

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

__all__ = ['_three_layer_fit_predict']
