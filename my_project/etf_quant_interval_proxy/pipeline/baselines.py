from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from my_project.etf_quant_interval_proxy.config import EtfQuantIntervalConfig


@dataclass(frozen=True)
class IntervalBaselineResult:
    metric_rows: tuple[dict[str, Any], ...]
    interval_rows: tuple[dict[str, Any], ...]
    rolling_rows: tuple[dict[str, Any], ...]
    backtest_rows: tuple[dict[str, Any], ...]


def _point_metrics(y_true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    y = np.asarray(y_true, dtype=float).reshape(-1)
    p = np.asarray(pred, dtype=float).reshape(-1)
    return {
        "rmse": float(mean_squared_error(y, p) ** 0.5),
        "mae": float(mean_absolute_error(y, p)),
        "r2": float(r2_score(y, p)),
        "direction_accuracy": float(np.mean(np.sign(y) == np.sign(p))),
        "true_positive_rate": float(np.mean(y > 0.0)),
        "pred_positive_rate": float(np.mean(p > 0.0)),
    }


def _interval_metrics(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray, alpha: float) -> dict[str, float]:
    y = np.asarray(y_true, dtype=float).reshape(-1)
    lo = np.asarray(lower, dtype=float).reshape(-1)
    hi = np.asarray(upper, dtype=float).reshape(-1)
    width = np.maximum(0.0, hi - lo)
    below = y < lo
    above = y > hi
    winkler = width.copy()
    winkler[below] += (2.0 / float(alpha)) * (lo[below] - y[below])
    winkler[above] += (2.0 / float(alpha)) * (y[above] - hi[above])
    return {
        "coverage": float(np.mean((y >= lo) & (y <= hi))),
        "avg_width": float(np.mean(width)),
        "median_width": float(np.median(width)),
        "winkler_score": float(np.mean(winkler)),
        "under_coverage_rate": float(np.mean(below)),
        "over_coverage_rate": float(np.mean(above)),
    }


def _model_factories(seed: int) -> dict[str, Any]:
    return {
        "ridge": lambda: make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "gradient_boosting": lambda: GradientBoostingRegressor(
            n_estimators=180,
            learning_rate=0.035,
            max_depth=2,
            min_samples_leaf=5,
            random_state=int(seed),
        ),
        "random_forest": lambda: RandomForestRegressor(
            n_estimators=180,
            max_depth=6,
            min_samples_leaf=6,
            random_state=int(seed),
            n_jobs=-1,
        ),
    }


def _fit_predict_with_residual_interval(
    *,
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    y_tr = np.asarray(y_train, dtype=float).reshape(-1)
    model.fit(np.asarray(X_train, dtype=float), y_tr)
    pred_train = np.asarray(model.predict(np.asarray(X_train, dtype=float)), dtype=float).reshape(-1)
    pred_test = np.asarray(model.predict(np.asarray(X_test, dtype=float)), dtype=float).reshape(-1)
    residual = y_tr - pred_train
    q_low = float(np.quantile(residual, float(alpha) / 2.0))
    q_high = float(np.quantile(residual, 1.0 - (float(alpha) / 2.0)))
    return pred_train, pred_test, pred_test + q_low, pred_test + q_high


def _naive_zero_prediction(
    *,
    y_train: np.ndarray,
    n_test: int,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    y_tr = np.asarray(y_train, dtype=float).reshape(-1)
    pred_train = np.zeros_like(y_tr, dtype=float)
    pred_test = np.zeros(int(n_test), dtype=float)
    q_low = float(np.quantile(y_tr, float(alpha) / 2.0))
    q_high = float(np.quantile(y_tr, 1.0 - (float(alpha) / 2.0)))
    return pred_train, pred_test, pred_test + q_low, pred_test + q_high


def _safe_corr(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, dtype=float).reshape(-1)
    b = np.asarray(right, dtype=float).reshape(-1)
    if a.size < 2 or a.size != b.size:
        return 0.0
    if float(np.std(a)) <= 1.0e-12 or float(np.std(b)) <= 1.0e-12:
        return 0.0
    value = float(np.corrcoef(a, b)[0, 1])
    return value if np.isfinite(value) else 0.0


def _rank_values(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty(arr.size, dtype=float)
    ranks[order] = np.arange(arr.size, dtype=float)
    return ranks


def _max_drawdown(returns: np.ndarray) -> float:
    r = np.asarray(returns, dtype=float).reshape(-1)
    if r.size == 0:
        return 0.0
    curve = np.cumprod(1.0 + np.clip(r, -0.95, 2.0))
    peak = np.maximum.accumulate(curve)
    dd = curve / np.maximum(peak, 1.0e-12) - 1.0
    return float(np.min(dd))


def _backtest_metrics(
    *,
    feature_space: str,
    model_name: str,
    y_test: np.ndarray,
    pred_test: np.ndarray,
    test_time_idx: np.ndarray,
    test_symbols: tuple[str, ...],
    horizon: int,
) -> dict[str, Any]:
    y = np.asarray(y_test, dtype=float).reshape(-1)
    pred = np.asarray(pred_test, dtype=float).reshape(-1)
    times = np.asarray(test_time_idx, dtype=int).reshape(-1)
    symbols = np.asarray(tuple(str(v) for v in test_symbols), dtype=object).reshape(-1)
    rank_pearson: list[float] = []
    rank_spearman: list[float] = []
    top_returns: list[float] = []
    equal_returns: list[float] = []
    long_short_returns: list[float] = []
    top_symbols: list[str] = []
    for t in sorted(set(int(v) for v in times)):
        mask = times == int(t)
        if int(np.sum(mask)) < 2:
            continue
        actual_t = y[mask]
        pred_t = pred[mask]
        symbol_t = symbols[mask]
        rank_pearson.append(_safe_corr(pred_t, actual_t))
        rank_spearman.append(_safe_corr(_rank_values(pred_t), _rank_values(actual_t)))
        top_idx = int(np.argmax(pred_t))
        bottom_idx = int(np.argmin(pred_t))
        top_returns.append(float(actual_t[top_idx]))
        equal_returns.append(float(np.mean(actual_t)))
        long_short_returns.append(float(actual_t[top_idx] - actual_t[bottom_idx]))
        top_symbols.append(str(symbol_t[top_idx]))
    top_arr = np.asarray(top_returns, dtype=float)
    equal_arr = np.asarray(equal_returns, dtype=float)
    ls_arr = np.asarray(long_short_returns, dtype=float)
    changes = sum(1 for prev, cur in zip(top_symbols[:-1], top_symbols[1:]) if prev != cur)
    periods = max(1, len(top_arr))
    periods_per_year = 252.0 / max(1.0, float(horizon))
    annual_scale = periods_per_year / float(periods)
    top_total = float(np.prod(1.0 + np.clip(top_arr, -0.95, 2.0)) - 1.0) if periods else 0.0
    equal_total = float(np.prod(1.0 + np.clip(equal_arr, -0.95, 2.0)) - 1.0) if periods else 0.0
    return {
        "feature_space": str(feature_space),
        "model": str(model_name),
        "rank_windows": int(periods),
        "mean_pearson_rank_ic": float(np.mean(rank_pearson)) if rank_pearson else 0.0,
        "mean_spearman_rank_ic": float(np.mean(rank_spearman)) if rank_spearman else 0.0,
        "top1_mean_return": float(np.mean(top_arr)) if periods else 0.0,
        "equal_weight_mean_return": float(np.mean(equal_arr)) if periods else 0.0,
        "top1_minus_equal_mean_return": float(np.mean(top_arr - equal_arr)) if periods else 0.0,
        "long_short_mean_return": float(np.mean(ls_arr)) if periods else 0.0,
        "top1_total_return_proxy": top_total,
        "equal_weight_total_return_proxy": equal_total,
        "top1_annualized_return_proxy": float((1.0 + top_total) ** annual_scale - 1.0) if periods else 0.0,
        "equal_weight_annualized_return_proxy": float((1.0 + equal_total) ** annual_scale - 1.0) if periods else 0.0,
        "top1_max_drawdown_proxy": _max_drawdown(top_arr),
        "equal_weight_max_drawdown_proxy": _max_drawdown(equal_arr),
        "top1_positive_rate": float(np.mean(top_arr > 0.0)) if periods else 0.0,
        "turnover_proxy": float(changes / max(1, len(top_symbols) - 1)) if len(top_symbols) > 1 else 0.0,
    }


def _rolling_rows(
    *,
    feature_space: str,
    model_name: str,
    y_test: np.ndarray,
    pred_test: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    test_time_idx: np.ndarray,
    alpha: float,
    rolling_window: int,
) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    times = np.asarray(test_time_idx, dtype=int).reshape(-1)
    unique_times = np.asarray(sorted(set(int(v) for v in times)), dtype=int)
    window = max(5, int(rolling_window))
    for start in range(0, max(1, len(unique_times)), window):
        chunk_times = set(int(v) for v in unique_times[start : start + window])
        mask = np.asarray([int(v) in chunk_times for v in times], dtype=bool)
        if int(np.sum(mask)) < 5:
            continue
        point = _point_metrics(np.asarray(y_test)[mask], np.asarray(pred_test)[mask])
        interval = _interval_metrics(np.asarray(y_test)[mask], np.asarray(lower)[mask], np.asarray(upper)[mask], alpha)
        rows.append(
            {
                "feature_space": str(feature_space),
                "model": str(model_name),
                "window_start_time_idx": int(min(chunk_times)),
                "window_end_time_idx": int(max(chunk_times)),
                "n_rows": int(np.sum(mask)),
                "rmse": float(point["rmse"]),
                "mae": float(point["mae"]),
                "coverage": float(interval["coverage"]),
                "avg_width": float(interval["avg_width"]),
                "winkler_score": float(interval["winkler_score"]),
            }
        )
    return tuple(rows)


def fit_interval_baselines(
    *,
    raw_train: np.ndarray,
    raw_test: np.ndarray,
    basis_train: np.ndarray,
    basis_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    test_time_idx: np.ndarray,
    test_symbols: tuple[str, ...],
    cfg: EtfQuantIntervalConfig,
) -> IntervalBaselineResult:
    alpha = float(cfg.interval_alpha)
    feature_sets = {
        "raw_features": (np.asarray(raw_train, dtype=float), np.asarray(raw_test, dtype=float)),
        "orthogonal_sources": (np.asarray(basis_train, dtype=float), np.asarray(basis_test, dtype=float)),
        "raw_plus_orthogonal_sources": (
            np.hstack([np.asarray(raw_train, dtype=float), np.asarray(basis_train, dtype=float)]),
            np.hstack([np.asarray(raw_test, dtype=float), np.asarray(basis_test, dtype=float)]),
        ),
    }
    metric_rows: list[dict[str, Any]] = []
    interval_rows: list[dict[str, Any]] = []
    rolling_rows: list[dict[str, Any]] = []
    backtest_rows: list[dict[str, Any]] = []

    pred_train, pred_test, lower, upper = _naive_zero_prediction(
        y_train=y_train,
        n_test=np.asarray(y_test).shape[0],
        alpha=alpha,
    )
    train_point = _point_metrics(y_train, pred_train)
    test_point = _point_metrics(y_test, pred_test)
    interval = _interval_metrics(y_test, lower, upper, alpha)
    metric_rows.append(
        {
            "feature_space": "naive",
            "model": "naive_zero_return",
            "feature_count": 0,
            "train_rmse": float(train_point["rmse"]),
            "train_mae": float(train_point["mae"]),
            "train_r2": float(train_point["r2"]),
            "test_rmse": float(test_point["rmse"]),
            "test_mae": float(test_point["mae"]),
            "test_r2": float(test_point["r2"]),
            "direction_accuracy": float(test_point["direction_accuracy"]),
            "true_positive_rate": float(test_point["true_positive_rate"]),
            "pred_positive_rate": float(test_point["pred_positive_rate"]),
        }
    )
    interval_rows.append(
        {
            "feature_space": "naive",
            "model": "naive_zero_return",
            "alpha": float(alpha),
            "nominal_coverage": float(1.0 - alpha),
            **interval,
        }
    )
    rolling_rows.extend(
        _rolling_rows(
            feature_space="naive",
            model_name="naive_zero_return",
            y_test=y_test,
            pred_test=pred_test,
            lower=lower,
            upper=upper,
            test_time_idx=test_time_idx,
            alpha=alpha,
            rolling_window=int(cfg.rolling_window),
        )
    )
    backtest_rows.append(
        _backtest_metrics(
            feature_space="naive",
            model_name="naive_zero_return",
            y_test=y_test,
            pred_test=pred_test,
            test_time_idx=test_time_idx,
            test_symbols=test_symbols,
            horizon=int(cfg.horizon),
        )
    )

    for feature_space, (X_tr, X_te) in feature_sets.items():
        if X_tr.shape[1] <= 0:
            continue
        for model_name, factory in _model_factories(int(cfg.seed)).items():
            pred_train, pred_test, lower, upper = _fit_predict_with_residual_interval(
                model=factory(),
                X_train=X_tr,
                y_train=y_train,
                X_test=X_te,
                alpha=alpha,
            )
            train_point = _point_metrics(y_train, pred_train)
            test_point = _point_metrics(y_test, pred_test)
            interval = _interval_metrics(y_test, lower, upper, alpha)
            metric_rows.append(
                {
                    "feature_space": str(feature_space),
                    "model": str(model_name),
                    "feature_count": int(X_tr.shape[1]),
                    "train_rmse": float(train_point["rmse"]),
                    "train_mae": float(train_point["mae"]),
                    "train_r2": float(train_point["r2"]),
                    "test_rmse": float(test_point["rmse"]),
                    "test_mae": float(test_point["mae"]),
                    "test_r2": float(test_point["r2"]),
                    "direction_accuracy": float(test_point["direction_accuracy"]),
                    "true_positive_rate": float(test_point["true_positive_rate"]),
                    "pred_positive_rate": float(test_point["pred_positive_rate"]),
                }
            )
            interval_rows.append(
                {
                    "feature_space": str(feature_space),
                    "model": str(model_name),
                    "alpha": float(alpha),
                    "nominal_coverage": float(1.0 - alpha),
                    **interval,
                }
            )
            rolling_rows.extend(
                _rolling_rows(
                    feature_space=str(feature_space),
                    model_name=str(model_name),
                    y_test=y_test,
                    pred_test=pred_test,
                    lower=lower,
                    upper=upper,
                    test_time_idx=test_time_idx,
                    alpha=alpha,
                    rolling_window=int(cfg.rolling_window),
                )
            )
            backtest_rows.append(
                _backtest_metrics(
                    feature_space=str(feature_space),
                    model_name=str(model_name),
                    y_test=y_test,
                    pred_test=pred_test,
                    test_time_idx=test_time_idx,
                    test_symbols=test_symbols,
                    horizon=int(cfg.horizon),
                )
            )
    return IntervalBaselineResult(
        metric_rows=tuple(metric_rows),
        interval_rows=tuple(interval_rows),
        rolling_rows=tuple(rolling_rows),
        backtest_rows=tuple(backtest_rows),
    )


def summarize_winners(rows: tuple[Mapping[str, Any], ...]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for model_name in sorted({str(row.get("model")) for row in rows}):
        model_rows = [dict(row) for row in rows if str(row.get("model")) == model_name]
        if not model_rows:
            continue
        winner = min(model_rows, key=lambda row: float(row.get("test_rmse", float("inf"))))
        raw = next((row for row in model_rows if row.get("feature_space") == "raw_features"), None)
        out[str(model_name)] = {
            "winner": str(winner.get("feature_space")),
            "winner_test_rmse": float(winner.get("test_rmse")),
            "raw_test_rmse": None if raw is None else float(raw.get("test_rmse")),
        }
    return out


__all__ = ["IntervalBaselineResult", "fit_interval_baselines", "summarize_winners"]
