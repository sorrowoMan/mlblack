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

__all__ = ['_rmse', '_mae', '_as_2d', '_build_symmetric_interval', '_interval_metrics']
