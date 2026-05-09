from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from my_project.etf_quant_interval_proxy.config import EtfQuantIntervalConfig


@dataclass(frozen=True)
class EtfPanelDataset:
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_names: tuple[str, ...]
    train_time_idx: np.ndarray
    test_time_idx: np.ndarray
    train_symbols: tuple[str, ...]
    test_symbols: tuple[str, ...]
    panel_frame: pd.DataFrame
    returns_frame: pd.DataFrame
    metadata: dict[str, Any]


def _load_returns_frame(cfg: EtfQuantIntervalConfig) -> pd.DataFrame:
    local_path = Path(str(cfg.dataset_url)).expanduser()
    if local_path.exists() and local_path.is_file():
        df = _read_local_dataset(local_path)
        df.attrs["actual_dataset_label"] = str(cfg.dataset_label)
        df.attrs["actual_dataset_url"] = str(local_path.resolve())
    else:
        df = _load_cached_or_remote_frame(cfg)

    numeric = df.select_dtypes(include=["number"]).copy()
    if numeric.empty:
        raise ValueError(f"No numeric ETF return columns found in dataset: {cfg.dataset_url}")
    numeric = numeric.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="all")
    numeric = numeric.fillna(0.0)
    numeric.columns = [str(col).strip().upper() for col in numeric.columns]
    if int(cfg.max_rows) > 0:
        numeric = numeric.iloc[: int(cfg.max_rows)].copy()
    numeric.index = np.arange(len(numeric), dtype=int)
    numeric.attrs["actual_dataset_label"] = str(getattr(df, "attrs", {}).get("actual_dataset_label", cfg.dataset_label))
    numeric.attrs["actual_dataset_url"] = str(getattr(df, "attrs", {}).get("actual_dataset_url", cfg.dataset_url))
    return numeric


def _read_local_dataset(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix in {".csv", ".txt"}:
        frame = pd.read_csv(path)
        date_cols = [col for col in frame.columns if str(col).lower() in {"date", "datetime", "timestamp"}]
        if date_cols:
            frame = frame.drop(columns=date_cols)
        return frame
    raise ValueError(f"Unsupported local ETF dataset format: {path}")


def _load_cached_or_remote_frame(cfg: EtfQuantIntervalConfig) -> pd.DataFrame:
    cache_dir = Path(str(cfg.cache_dir)).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "etf_returns.parquet"
    if cache_path.exists() and cache_path.stat().st_size > 0:
        df = pd.read_parquet(cache_path)
    else:
        df = _download_returns_parquet(cfg, cache_path)
    return df


def _download_returns_parquet(cfg: EtfQuantIntervalConfig, cache_path: Path) -> pd.DataFrame:
    try:
        import requests

        response = requests.get(str(cfg.dataset_url), timeout=35)
        response.raise_for_status()
        cache_path.write_bytes(response.content)
        df = pd.read_parquet(cache_path)
        df.attrs["actual_dataset_label"] = str(cfg.dataset_label)
        df.attrs["actual_dataset_url"] = str(cfg.dataset_url)
        return df
    except Exception:
        if not bool(cfg.allow_public_sample_fallback):
            raise
        df = _download_qstrader_public_etf_sample()
        df.to_parquet(cache_path, index=False)
        return df


def _download_qstrader_public_etf_sample() -> pd.DataFrame:
    frames = []
    for symbol in ("SPY", "AGG"):
        url = f"https://raw.githubusercontent.com/quantstart/qstrader/master/data/{symbol}.csv"
        request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(request, timeout=30) as response:
            frame = pd.read_csv(response)
        close_col = "Adj Close" if "Adj Close" in frame.columns else "Close"
        item = frame[["Date", close_col]].copy()
        item["Date"] = pd.to_datetime(item["Date"])
        item = item.rename(columns={close_col: symbol}).set_index("Date")
        frames.append(item)
    prices = pd.concat(frames, axis=1).sort_index().ffill().dropna()
    returns = prices.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    returns.attrs["actual_dataset_label"] = "qstrader_public_spy_agg_etf_sample"
    returns.attrs["actual_dataset_url"] = "https://github.com/quantstart/qstrader/tree/master/data"
    return returns.reset_index(drop=True)


def _compound_return(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = np.clip(arr, -0.95, 2.0)
    return float(np.expm1(np.sum(np.log1p(arr))))


def _feature_dict(*, returns: pd.DataFrame, t: int, target: str, symbols: tuple[str, ...], lookback: int) -> dict[str, float]:
    out: dict[str, float] = {}
    current = returns.iloc[t]
    hist = returns.iloc[max(0, t - lookback + 1) : t + 1]
    market_ret = float(current.mean())
    out["market_ret_1d"] = market_ret
    out["market_dispersion_1d"] = float(current.std(ddof=0))
    out["market_positive_share_1d"] = float((current > 0.0).mean())
    out["market_mom_5"] = float(hist.tail(5).mean(axis=1).sum())
    out["market_mom_20"] = float(hist.tail(20).mean(axis=1).sum())
    out["market_vol_20"] = float(hist.tail(20).mean(axis=1).std(ddof=0))

    own = hist[str(target)]
    out["own_ret_1d"] = float(current[str(target)])
    out["own_mom_3"] = float(own.tail(3).sum())
    out["own_mom_5"] = float(own.tail(5).sum())
    out["own_mom_20"] = float(own.tail(20).sum())
    out["own_vol_5"] = float(own.tail(5).std(ddof=0))
    out["own_vol_20"] = float(own.tail(20).std(ddof=0))
    out["own_downside_vol_20"] = float(own.tail(20).clip(upper=0.0).std(ddof=0))
    out["own_vs_market_1d"] = float(current[str(target)] - market_ret)
    out["own_mom5_vs_market"] = float(out["own_mom_5"] - out["market_mom_5"])
    out["own_vol20_vs_market"] = float(out["own_vol_20"] - out["market_vol_20"])

    for sym in symbols:
        col = hist[str(sym)]
        out[f"src_{sym}_ret_1d"] = float(current[str(sym)])
        out[f"src_{sym}_mom_5"] = float(col.tail(5).sum())
        out[f"src_{sym}_mom_20"] = float(col.tail(20).sum())
        out[f"src_{sym}_vol_20"] = float(col.tail(20).std(ddof=0))
        out[f"is_target_{sym}"] = 1.0 if str(sym) == str(target) else 0.0
    return out


def load_etf_panel_dataset(cfg: EtfQuantIntervalConfig) -> EtfPanelDataset:
    returns = _load_returns_frame(cfg)
    symbols = tuple(str(col) for col in returns.columns)
    horizon = max(1, int(cfg.horizon))
    lookback = max(3, int(cfg.lookback))
    if len(returns) <= lookback + horizon + 5:
        raise ValueError(
            f"ETF returns dataset too short for lookback={lookback}, horizon={horizon}: rows={len(returns)}"
        )

    records: list[dict[str, Any]] = []
    features: list[dict[str, float]] = []
    for t in range(lookback - 1, len(returns) - horizon):
        for target in symbols:
            feat = _feature_dict(returns=returns, t=t, target=target, symbols=symbols, lookback=lookback)
            y = _compound_return(returns[str(target)].iloc[t + 1 : t + 1 + horizon].to_numpy(dtype=float))
            row = {"time_idx": int(t), "target_symbol": str(target), "y_forward_return": float(y)}
            records.append(row)
            features.append(feat)

    frame = pd.DataFrame(records)
    X_frame = pd.DataFrame(features).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    feature_names = tuple(str(col) for col in X_frame.columns)
    X = X_frame.to_numpy(dtype=float)
    y = frame["y_forward_return"].to_numpy(dtype=float)
    unique_times = np.asarray(sorted(frame["time_idx"].unique()), dtype=int)
    split_pos = int(np.clip(np.floor(len(unique_times) * float(cfg.train_ratio)), 1, len(unique_times) - 1))
    split_time = int(unique_times[split_pos - 1])
    train_mask = frame["time_idx"].to_numpy(dtype=int) <= split_time
    test_mask = ~train_mask

    metadata = {
        "dataset_label": str(returns.attrs.get("actual_dataset_label", cfg.dataset_label)),
        "dataset_url": str(returns.attrs.get("actual_dataset_url", cfg.dataset_url)),
        "dataset_rows": int(len(returns)),
        "panel_rows": int(len(frame)),
        "symbols": symbols,
        "horizon": int(horizon),
        "lookback": int(lookback),
        "target_task": "regression",
        "target_semantics": "future_horizon_compounded_return",
        "split_time_idx": int(split_time),
        "feature_count": int(X.shape[1]),
    }
    return EtfPanelDataset(
        X_train=X[train_mask],
        y_train=y[train_mask],
        X_test=X[test_mask],
        y_test=y[test_mask],
        feature_names=feature_names,
        train_time_idx=frame.loc[train_mask, "time_idx"].to_numpy(dtype=int),
        test_time_idx=frame.loc[test_mask, "time_idx"].to_numpy(dtype=int),
        train_symbols=tuple(str(v) for v in frame.loc[train_mask, "target_symbol"].to_numpy()),
        test_symbols=tuple(str(v) for v in frame.loc[test_mask, "target_symbol"].to_numpy()),
        panel_frame=pd.concat([frame, X_frame], axis=1),
        returns_frame=returns,
        metadata=metadata,
    )


__all__ = ["EtfPanelDataset", "load_etf_panel_dataset"]
