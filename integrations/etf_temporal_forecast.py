from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from blackbase.resources import ResourceContext, coerce_resource_context


DEFAULT_DATASET_URL = (
    Path(__file__).resolve().parents[1]
    / "runs"
    / "etf_temporal_forecast"
    / "cache"
    / "multi_etf_returns_momodel_kaggle.parquet"
)


@dataclass(frozen=True)
class EtfTemporalForecastConfig:
    dataset_url: str = str(DEFAULT_DATASET_URL)
    dataset_label: str = "multi_etf_returns_momodel_kaggle"
    models: tuple[str, ...] = ("ridge", "hist_gradient_boosting")
    target_horizon: int = 1
    transaction_cost: float = 0.0005
    output_dir: str = "runs/etf_temporal_forecast"


@dataclass(frozen=True)
class WalkForwardSpec:
    min_train_size: int = 1200
    test_size: int = 200
    step_size: int = 200
    mode: str = "expanding"
    train_window_size: int = 1440
    max_folds: int = 2
    max_train_panel_rows: int = 12000
    max_test_panel_rows: int = 4000


@dataclass(frozen=True)
class EtfTemporalForecastResult:
    summary: Mapping[str, Any]
    output_dir: Path
    records: tuple[Mapping[str, Any], ...] = field(default_factory=tuple)


def run_etf_temporal_forecast_multi_seed(
    *,
    cfg: EtfTemporalForecastConfig | Mapping[str, Any] | None = None,
    walkforward: WalkForwardSpec | Mapping[str, Any] | None = None,
    seeds: Sequence[int] = (42,),
    suite_id: str = "etf_temporal_forecast",
    output_dir: str | Path | None = None,
    potential_params_override: Mapping[str, Any] | None = None,
    resource_context: Mapping[str, Any] | ResourceContext | None = None,
    panel_builder: Any | None = None,
) -> EtfTemporalForecastResult:
    config = _coerce_config(cfg)
    wf = _coerce_walkforward(walkforward)
    out_dir = Path(output_dir or config.output_dir).expanduser().resolve()

    returns = _load_returns(config.dataset_url)
    grant = coerce_resource_context(resource_context)
    threads = max(1, int(grant.threads or 1))
    panel = (
        panel_builder.build_panel(returns)
        if panel_builder is not None
        else _build_panel(returns, horizon=int(config.target_horizon))
    )
    records: list[dict[str, Any]] = []
    for seed in tuple(int(s) for s in seeds):
        records.extend(
            _evaluate_seed(
                panel,
                config,
                wf,
                int(seed),
                potential_params_override,
                threads=threads,
            )
        )

    summary = _summarize(
        records,
        config,
        wf,
        returns,
        suite_id=suite_id,
        lane_bundle=potential_params_override,
        resource_context=grant,
    )
    return EtfTemporalForecastResult(summary=summary, output_dir=out_dir, records=tuple(records))


def run_etf_walkforward_multi_seed(
    *,
    cfg: EtfTemporalForecastConfig | Mapping[str, Any] | None = None,
    walkforward: WalkForwardSpec | Mapping[str, Any] | None = None,
    seeds: Sequence[int] = (42,),
    suite_id: str = "etf_temporal_forecast",
    output_dir: str | Path | None = None,
    potential_params_override: Mapping[str, Any] | None = None,
    resource_context: Mapping[str, Any] | ResourceContext | None = None,
    panel_builder: Any | None = None,
) -> EtfTemporalForecastResult:
    return run_etf_temporal_forecast_multi_seed(
        cfg=cfg,
        walkforward=walkforward,
        seeds=seeds,
        suite_id=suite_id,
        output_dir=output_dir,
        potential_params_override=potential_params_override,
        resource_context=resource_context,
        panel_builder=panel_builder,
    )


def _coerce_config(cfg: EtfTemporalForecastConfig | Mapping[str, Any] | None) -> EtfTemporalForecastConfig:
    if cfg is None:
        return EtfTemporalForecastConfig()
    if isinstance(cfg, EtfTemporalForecastConfig):
        return cfg
    values = dict(cfg)
    models = values.get("models", values.get("baseline_models", EtfTemporalForecastConfig.models))
    if isinstance(models, str):
        models = _parse_models(models)
    values["models"] = tuple(str(m) for m in models)
    allowed = {field.name for field in EtfTemporalForecastConfig.__dataclass_fields__.values()}
    return EtfTemporalForecastConfig(**{k: v for k, v in values.items() if k in allowed})


def _coerce_walkforward(walkforward: WalkForwardSpec | Mapping[str, Any] | None) -> WalkForwardSpec:
    if walkforward is None:
        return WalkForwardSpec()
    if isinstance(walkforward, WalkForwardSpec):
        return walkforward
    values = dict(walkforward)
    allowed = {field.name for field in WalkForwardSpec.__dataclass_fields__.values()}
    return WalkForwardSpec(**{k: v for k, v in values.items() if k in allowed})


def _parse_models(text: str) -> tuple[str, ...]:
    models: list[str] = []
    for raw in str(text or "").replace(";", ",").split(","):
        item = raw.strip()
        if item:
            models.append(item)
    return tuple(models) if models else EtfTemporalForecastConfig.models


def _load_returns(path: str | Path) -> pd.DataFrame:
    dataset_path = Path(path).expanduser().resolve()
    df = pd.read_parquet(dataset_path)
    if not isinstance(df.index, pd.DatetimeIndex):
        date_col = "date" if "date" in df.columns else df.columns[0]
        df = df.set_index(pd.to_datetime(df[date_col]))
        df = df.drop(columns=[date_col], errors="ignore")
    df = df.sort_index()
    numeric = df.select_dtypes(include=["number"]).astype(float)
    if numeric.empty:
        raise ValueError(f"ETF dataset has no numeric return columns: {dataset_path}")
    return numeric.replace([np.inf, -np.inf], np.nan).dropna(how="any")


def _build_panel(returns: pd.DataFrame, *, horizon: int) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    market = returns.mean(axis=1)
    for ticker in returns.columns:
        r = returns[ticker].astype(float)
        frame = pd.DataFrame(
            {
                "date": returns.index,
                "ticker": str(ticker),
                "target": r.shift(-int(horizon)),
                "ret_lag_1": r,
                "ret_lag_2": r.shift(1),
                "mom_5": r.rolling(5).mean(),
                "mom_20": r.rolling(20).mean(),
                "vol_20": r.rolling(20).std(),
                "market_ret_1": market,
                "market_mom_5": market.rolling(5).mean(),
                "relative_mom_20": r.rolling(20).mean() - market.rolling(20).mean(),
            }
        )
        frames.append(frame)
    panel = pd.concat(frames, ignore_index=True)
    panel["ticker_code"] = pd.Categorical(panel["ticker"]).codes.astype(float)
    return panel.dropna(axis=0).reset_index(drop=True)


def _evaluate_seed(
    panel: pd.DataFrame,
    cfg: EtfTemporalForecastConfig,
    wf: WalkForwardSpec,
    seed: int,
    lane_bundle: Mapping[str, Any] | None,
    *,
    threads: int,
) -> list[dict[str, Any]]:
    dates = tuple(pd.Index(panel["date"]).drop_duplicates().sort_values())
    records: list[dict[str, Any]] = []
    start = int(wf.min_train_size)
    fold_idx = 0
    while start + int(wf.test_size) <= len(dates):
        if int(wf.max_folds) > 0 and fold_idx >= int(wf.max_folds):
            break
        train_dates = dates[:start]
        if str(wf.mode).lower() == "rolling":
            train_dates = train_dates[-int(wf.train_window_size) :]
        test_dates = dates[start : start + int(wf.test_size)]
        train_dates = _cap_dates(train_dates, panel, int(wf.max_train_panel_rows), from_tail=True)
        test_dates = _cap_dates(test_dates, panel, int(wf.max_test_panel_rows), from_tail=False)
        train = panel[panel["date"].isin(train_dates)].copy()
        test = panel[panel["date"].isin(test_dates)].copy()
        records.append(
            _evaluate_fold(
                train,
                test,
                cfg,
                seed=seed,
                fold_idx=fold_idx,
                lane_bundle=lane_bundle,
                threads=threads,
            )
        )
        start += int(wf.step_size)
        fold_idx += 1
    return records


def _cap_dates(dates: Sequence[pd.Timestamp], panel: pd.DataFrame, max_panel_rows: int, *, from_tail: bool) -> tuple[pd.Timestamp, ...]:
    if int(max_panel_rows) <= 0:
        return tuple(dates)
    asset_count = max(1, int(panel["ticker"].nunique()))
    max_dates = max(1, int(max_panel_rows) // asset_count)
    limited = tuple(dates)[-max_dates:] if from_tail else tuple(dates)[:max_dates]
    return tuple(limited)


def _evaluate_fold(
    train: pd.DataFrame,
    test: pd.DataFrame,
    cfg: EtfTemporalForecastConfig,
    *,
    seed: int,
    fold_idx: int,
    lane_bundle: Mapping[str, Any] | None,
    threads: int,
) -> dict[str, Any]:
    feature_cols = [
        "ret_lag_1",
        "ret_lag_2",
        "mom_5",
        "mom_20",
        "vol_20",
        "market_ret_1",
        "market_mom_5",
        "relative_mom_20",
        "ticker_code",
    ]
    x_train = train[feature_cols].to_numpy(dtype=float)
    y_train = train["target"].to_numpy(dtype=float)
    x_test = test[feature_cols].to_numpy(dtype=float)
    y_test = test["target"].to_numpy(dtype=float)
    model_names = _models_from_bundle(cfg, lane_bundle)
    predictions: dict[str, np.ndarray] = {}
    model_metrics: dict[str, dict[str, float]] = {}
    for name in model_names:
        estimator = _build_model(name, seed, threads=threads)
        estimator.fit(x_train, y_train)
        pred = np.asarray(estimator.predict(x_test), dtype=float).reshape(-1)
        predictions[name] = pred
        model_metrics[name] = {"rmse": _rmse(y_test, pred), "rank_ic": _rank_ic(test, pred)}
    blended = _blend_predictions(predictions, model_metrics, cfg, lane_bundle)
    portfolio = _portfolio_metrics(test, blended, lane_bundle=lane_bundle, transaction_cost=float(cfg.transaction_cost))
    return {
        "seed": int(seed),
        "fold": int(fold_idx),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_start": str(pd.Timestamp(train["date"].min()).date()),
        "train_end": str(pd.Timestamp(train["date"].max()).date()),
        "test_start": str(pd.Timestamp(test["date"].min()).date()),
        "test_end": str(pd.Timestamp(test["date"].max()).date()),
        "models": tuple(model_names),
        "model_metrics": model_metrics,
        "composite_test_rmse": _rmse(y_test, blended),
        "composite_direction_accuracy": float(np.mean(np.sign(blended) == np.sign(y_test))),
        "composite_rank_ic": _rank_ic(test, blended),
        "composite_hit_rate": float(portfolio["hit_rate"]),
        "composite_net_sharpe_proxy": float(portfolio["net_sharpe_proxy"]),
        "composite_max_drawdown_abs": float(portfolio["max_drawdown_abs"]),
        "composite_turnover_proxy": float(portfolio["turnover_proxy"]),
    }


def _models_from_bundle(cfg: EtfTemporalForecastConfig, lane_bundle: Mapping[str, Any] | None) -> tuple[str, ...]:
    models = list(cfg.models)
    if lane_bundle is not None:
        for lane in tuple(lane_bundle.get("lanes", ()) or ()):
            for model_name in tuple(dict(lane).get("lane_models", ()) or ()):
                models.append(str(model_name))
    aliases = {"mlp_torch": "mlp_sklearn", "random_forest": "random_forest"}
    normalized = [aliases.get(str(item), str(item)) for item in models]
    return tuple(dict.fromkeys(normalized))


def _build_model(name: str, seed: int, *, threads: int = 1) -> Any:
    key = str(name).lower().strip()
    if key == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    if key == "elasticnet":
        return make_pipeline(StandardScaler(), ElasticNet(alpha=0.0005, l1_ratio=0.15, max_iter=4000, random_state=seed))
    if key in {"hist_gradient_boosting", "hgbt", "gbdt"}:
        return HistGradientBoostingRegressor(max_iter=160, learning_rate=0.04, l2_regularization=0.05, random_state=seed)
    if key == "random_forest":
        return RandomForestRegressor(
            n_estimators=96,
            max_depth=7,
            min_samples_leaf=20,
            n_jobs=max(1, int(threads)),
            random_state=seed,
        )
    if key == "mlp_sklearn":
        return make_pipeline(
            StandardScaler(),
            MLPRegressor(hidden_layer_sizes=(64, 32), alpha=0.001, max_iter=240, early_stopping=True, random_state=seed),
        )
    raise ValueError(f"unknown ETF forecast model: {name}")


def _blend_predictions(
    predictions: Mapping[str, np.ndarray],
    model_metrics: Mapping[str, Mapping[str, float]],
    cfg: EtfTemporalForecastConfig,
    lane_bundle: Mapping[str, Any] | None,
) -> np.ndarray:
    if not predictions:
        raise ValueError("no predictions to blend")
    if lane_bundle is None:
        return _weighted_average(predictions, tuple(cfg.models), model_metrics, mode="inverse_rmse")
    lane_preds: list[np.ndarray] = []
    lane_weights: list[float] = []
    for lane in tuple(lane_bundle.get("lanes", ()) or ()):
        lane_dict = dict(lane)
        if not bool(lane_dict.get("enabled", True)):
            continue
        names = tuple(str(x) for x in tuple(lane_dict.get("lane_models", ()) or ()))
        names = tuple("mlp_sklearn" if name == "mlp_torch" else name for name in names)
        lane_pred = _weighted_average(predictions, names, model_metrics, mode=str(lane_dict.get("blend", "uniform")))
        lane_preds.append(lane_pred)
        lane_weights.append(float(lane_dict.get("alpha", 1.0)))
    if not lane_preds:
        return _weighted_average(predictions, tuple(predictions), model_metrics, mode="inverse_rmse")
    weights = np.asarray(lane_weights, dtype=float)
    weights = weights / max(float(np.sum(np.abs(weights))), 1e-12)
    stacked = np.vstack(lane_preds)
    return np.sum(stacked * weights.reshape(-1, 1), axis=0)


def _weighted_average(
    predictions: Mapping[str, np.ndarray],
    names: Sequence[str],
    model_metrics: Mapping[str, Mapping[str, float]],
    *,
    mode: str,
) -> np.ndarray:
    present = [str(name) for name in names if str(name) in predictions]
    if not present:
        present = list(predictions)
    if str(mode) == "inverse_rmse":
        weights = np.asarray([1.0 / max(float(model_metrics[name].get("rmse", 1.0)), 1e-9) for name in present], dtype=float)
    else:
        weights = np.ones(len(present), dtype=float)
    weights = weights / max(float(np.sum(weights)), 1e-12)
    stacked = np.vstack([predictions[name] for name in present])
    return np.sum(stacked * weights.reshape(-1, 1), axis=0)


def _portfolio_metrics(test: pd.DataFrame, pred: np.ndarray, *, lane_bundle: Mapping[str, Any] | None, transaction_cost: float) -> dict[str, float]:
    data = test[["date", "ticker", "target"]].copy()
    data["pred"] = pred
    top_k = int(dict(lane_bundle or {}).get("active_top_k", 10))
    top_k = max(1, min(top_k, int(data["ticker"].nunique())))
    daily_returns: list[float] = []
    turnovers: list[float] = []
    previous: set[str] | None = None
    for _, group in data.groupby("date", sort=True):
        selected = group.sort_values("pred", ascending=False).head(top_k)
        names = set(str(x) for x in selected["ticker"])
        raw = float(selected["target"].mean())
        turnover = 0.0 if previous is None else 1.0 - (len(names & previous) / max(len(names | previous), 1))
        daily_returns.append(raw - float(transaction_cost) * turnover)
        turnovers.append(turnover)
        previous = names
    arr = np.asarray(daily_returns, dtype=float)
    std = float(np.std(arr))
    sharpe = 0.0 if std <= 1e-12 else math.sqrt(252.0) * float(np.mean(arr)) / std
    equity = np.cumprod(1.0 + arr)
    peak = np.maximum.accumulate(equity)
    drawdown = equity / np.maximum(peak, 1e-12) - 1.0
    return {
        "net_sharpe_proxy": float(sharpe),
        "max_drawdown_abs": float(abs(np.min(drawdown))) if drawdown.size else 0.0,
        "turnover_proxy": float(np.mean(turnovers)) if turnovers else 0.0,
        "hit_rate": float(np.mean(arr > 0.0)) if arr.size else 0.0,
    }


def _rank_ic(test: pd.DataFrame, pred: np.ndarray) -> float:
    data = test[["date", "target"]].copy()
    data["pred"] = pred
    values: list[float] = []
    for _, group in data.groupby("date", sort=True):
        if len(group) < 3:
            continue
        corr = group["pred"].rank().corr(group["target"].rank())
        if pd.notna(corr):
            values.append(float(corr))
    return float(np.mean(values)) if values else 0.0


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(math.sqrt(float(mean_squared_error(y_true, y_pred))))


def _summarize(
    records: Sequence[Mapping[str, Any]],
    cfg: EtfTemporalForecastConfig,
    wf: WalkForwardSpec,
    returns: pd.DataFrame,
    *,
    suite_id: str,
    lane_bundle: Mapping[str, Any] | None,
    resource_context: ResourceContext,
) -> dict[str, Any]:
    aggregate = {
        f"{metric}_mean": _mean(records, metric)
        for metric in (
            "composite_test_rmse",
            "composite_direction_accuracy",
            "composite_rank_ic",
            "composite_hit_rate",
            "composite_net_sharpe_proxy",
            "composite_max_drawdown_abs",
            "composite_turnover_proxy",
        )
    }
    aggregate["composite_rank_ic_std"] = _std(records, "composite_rank_ic")
    return {
        "suite_id": str(suite_id),
        "case": "etf_temporal_forecast",
        "dataset": {
            "url": str(cfg.dataset_url),
            "label": str(cfg.dataset_label),
            "rows": int(len(returns)),
            "assets": int(len(returns.columns)),
            "start": str(returns.index.min().date()),
            "end": str(returns.index.max().date()),
        },
        "config": {
            "models": list(cfg.models),
            "target_horizon": int(cfg.target_horizon),
            "transaction_cost": float(cfg.transaction_cost),
        },
        "walkforward": wf.__dict__,
        "lane_bundle": lane_bundle,
        "resource_context": resource_context.as_dict(),
        "aggregate": aggregate,
        "fold_count": int(len(records)),
        "records": list(records),
    }


def _mean(records: Sequence[Mapping[str, Any]], key: str) -> float:
    values = [float(item.get(key, 0.0) or 0.0) for item in records]
    return float(np.mean(values)) if values else 0.0


def _std(records: Sequence[Mapping[str, Any]], key: str) -> float:
    values = [float(item.get(key, 0.0) or 0.0) for item in records]
    return float(np.std(values)) if values else 0.0


__all__ = [
    "DEFAULT_DATASET_URL",
    "EtfTemporalForecastConfig",
    "EtfTemporalForecastResult",
    "WalkForwardSpec",
    "run_etf_temporal_forecast_multi_seed",
    "run_etf_walkforward_multi_seed",
]
