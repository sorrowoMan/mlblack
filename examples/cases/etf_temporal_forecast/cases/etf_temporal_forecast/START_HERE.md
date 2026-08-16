# START_HERE

## 1) 这个 case 验证什么

`etf_temporal_forecast` 验证 mlblack 的 ETF walk-forward temporal forecast 闭环。

- 数据：40 个 ETF、2668 个交易日、2007-04-12 到 2017-11-10、无缺失。
- mlblack 负责 feature/target construction、walk-forward split、model evaluation 和 metric aggregation。
- 默认 quickstart 使用 `ridge + hist_gradient_boosting`，不是把 MLP 当主力模型。
- 后续 nsgablack 的 `etf_lane_outer_search` 会调用这个标准 case 做内层评估。

详细结构、指标和面试叙事见 `README.md`。

## 2) Check

```powershell
cd C:\Users\hp\Desktop\mlblack
python examples\cases\etf_temporal_forecast\run_project.py --check --build-check
```

## 3) Quickstart

```powershell
python examples\cases\etf_temporal_forecast\cases\etf_temporal_forecast\run_solver.py --quickstart
```

Quickstart 限制 folds 和 panel rows，用于快速验证链路和指标输出。

## 4) Serious run

```powershell
python examples\cases\etf_temporal_forecast\cases\etf_temporal_forecast\run_solver.py --serious
```

Serious profile 使用更多 folds/seeds/models，更接近面试可讲的研究运行。

## 5) 关键指标

| 指标 | 含义 |
|---|---|
| `composite_test_rmse_mean` | ETF panel return prediction RMSE。 |
| `composite_direction_accuracy_mean` | 预测方向是否正确。 |
| `composite_rank_ic_mean` | 每日横截面 prediction ranking 与 realized return ranking 的相关性。 |
| `composite_rank_ic_std` | Rank IC 的 fold/seed 稳定性。 |
| `composite_hit_rate_mean` | Top-k proxy portfolio 日收益为正的比例。 |
| `composite_net_sharpe_proxy_mean` | 扣除 turnover cost 后的 Sharpe proxy。 |
| `composite_max_drawdown_abs_mean` | Proxy equity curve 最大回撤。 |
| `composite_turnover_proxy_mean` | Top-k 持仓变化率。 |

## 6) 预期信号

有效运行不一定要显著降低 point RMSE，更重要的是观察 rank IC、hit rate、Sharpe proxy、drawdown 和 turnover 是否形成稳定 tradeoff。

## 7) 已生成的正式插件产物

运行后，`runs/etf_temporal_forecast/` 会额外生成：

- `etf_temporal_forecast.etf_report.json`
- `etf_temporal_forecast.etf_report.md`
- `etf_temporal_forecast.observability.json`
- `etf_temporal_forecast.observability.md`
