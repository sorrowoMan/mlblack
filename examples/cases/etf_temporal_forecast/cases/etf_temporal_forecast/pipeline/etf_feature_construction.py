"""ETF feature construction pipeline: transforms raw returns into labeled feature panel."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class FeatureBuildSpec:
    """Feature construction specification."""

    target_horizon: int = 1
    window_5: int = 5
    window_20: int = 20


class EtfFeatureBuilder:
    """
    Encapsulates panel construction and feature engineering.

    Input: Raw ETF returns DataFrame (index: date, columns: tickers)
    Output: Panel DataFrame with features and target

    This is extracted from mlblack.integrations.etf_temporal_forecast._build_panel()
    for clarity and reusability.
    """

    def __init__(self, spec: FeatureBuildSpec | None = None):
        self.spec = spec or FeatureBuildSpec()

    def build_panel(self, returns: pd.DataFrame) -> pd.DataFrame:
        """
        Convert raw returns to feature panel.

        Args:
            returns: DataFrame(index=date, columns=tickers), all numeric

        Returns:
            Panel DataFrame with columns: date, ticker, features..., target
        """
        frames: list[pd.DataFrame] = []
        market = returns.mean(axis=1)

        for ticker in returns.columns:
            r = returns[ticker].astype(float)
            frame = pd.DataFrame(
                {
                    "date": returns.index,
                    "ticker": str(ticker),
                    "target": r.shift(-int(self.spec.target_horizon)),
                    "ret_lag_1": r,
                    "ret_lag_2": r.shift(1),
                    "mom_5": r.rolling(self.spec.window_5).mean(),
                    "mom_20": r.rolling(self.spec.window_20).mean(),
                    "vol_20": r.rolling(self.spec.window_20).std(),
                    "market_ret_1": market,
                    "market_mom_5": market.rolling(self.spec.window_5).mean(),
                    "relative_mom_20": r.rolling(self.spec.window_20).mean()
                    - market.rolling(self.spec.window_20).mean(),
                }
            )
            frames.append(frame)

        panel = pd.concat(frames, ignore_index=True)
        panel["ticker_code"] = pd.Categorical(panel["ticker"]).codes.astype(float)
        return panel.dropna(axis=0).reset_index(drop=True)
