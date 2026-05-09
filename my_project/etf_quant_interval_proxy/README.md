# etf_quant_interval_proxy

Standard mlblack project scaffold for an ETF-style quant interval demo on real
public ETF return data.

Dataset source:

- `P2SAMAPA/p2-etf-rough-path-forecaster-results` on Hugging Face.
- The default parquet contains daily return columns for 14 ETFs, including QQQ,
  XLK, XLF, XLY, XLP, XLI, XLU, XLV, XLB, XTL, XBI, GDX, XME, and IWM.
- If the Hugging Face parquet cannot be reached, the loader falls back to the
  public QSTrader SPY/AGG OHLCV CSV sample and converts adjusted closes into
  daily returns. This keeps the scaffold runnable without API keys, while the
  larger ETF panel can be used when the network/source is available.

This is not a trading system and not investment advice. It is a framework demo
for source object construction, orthogonal source governance, and interval-style
forecast reports.

Layer order:

`ETF return panel -> rolling/relative source objects -> orthogonal source governance -> fixed interval heads`

Current task:

- Build a panel where each row is `(time_idx, target_etf)`.
- Predict the target ETF's future horizon compounded return.
- Compare `raw_features`, `orthogonal_sources`, and
  `raw_plus_orthogonal_sources`.
- Use fixed downstream heads so this remains an `mlblack` trainer/proxy scaffold;
  outer structure search can be added later through `nsgablack`.

Reports:

- `baseline_metrics.csv`: RMSE/MAE/R2 by feature space and model.
- `interval_metrics.csv`: residual interval coverage, width, and Winkler score.
- `rolling_metrics.csv`: rolling test-window RMSE/coverage/width.
- `rank_backtest_metrics.csv`: direction, rank IC, top-1 allocation proxy,
  equal-weight comparison, drawdown, and turnover metrics.
- `orthogonal_source_table.csv`: selected source objects and stability fields.
- `summary.json`: config, dataset contract, source report, and winners.

The report includes a `naive_zero_return` baseline. Its prediction is always
zero future return, with train-set empirical residual quantiles used for the
interval. This is the minimum sanity check for ETF return prediction, because
short-horizon return targets often look accurate under RMSE simply due to small
target scale.

Run:

```powershell
python my_project\etf_quant_interval_proxy\run_solver.py --check
python my_project\etf_quant_interval_proxy\run_solver.py --suite-id etf_proxy_v1
```

Use a larger local ETF parquet/CSV without changing the scaffold:

```powershell
python my_project\etf_quant_interval_proxy\run_solver.py `
  --suite-id etf_large_panel_v1 `
  --dataset-url "C:\path\to\large_etf_returns.parquet" `
  --dataset-label large_etf_returns
```

The file should contain one numeric column per ETF return series. Optional date
columns named `date`, `datetime`, or `timestamp` are ignored for feature
construction.
