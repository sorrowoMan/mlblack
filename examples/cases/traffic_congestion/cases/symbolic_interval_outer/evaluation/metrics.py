from __future__ import annotations

from ..runtime.legacy_imports import *

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

def _fmt_float(v: float, *, precision: int = 6) -> str:
    if not np.isfinite(v):
        return str(v)
    return f"{float(v):.{int(max(1, precision))}g}"

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

__all__ = ['_rmse', '_mae', '_as_2d', '_fmt_float', '_build_formula_summary', '_build_symmetric_interval', '_conformal_q', '_build_native_quantile_interval', '_interval_metrics']
