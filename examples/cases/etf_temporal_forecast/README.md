# etf_temporal_forecast（ETF 时序预测）

`etf_temporal_forecast` 是一个 mlblack 标准脚手架 case，用真实 ETF 日收益率面板做 walk-forward temporal forecast。

## 这个 case 验证什么

- mlblack 负责 ETF dataset、feature/target construction、walk-forward split、模型训练和指标汇总。
- 数据来自 legacy backup 的 `multi_etf_returns_momodel_kaggle.parquet`，默认放在 `runs/etf_temporal_forecast/cache/`。
- 默认 quickstart 使用 `ridge + hist_gradient_boosting`，不是把 MLP 当主角。
- serious profile 会扩展到 `elasticnet`、`random_forest`、`mlp_sklearn`，后续可接 PatchTST / Temporal Transformer。

## 数据规模

| 项 | 值 |
|---|---:|
| trading days | 2668 |
| ETF assets | 40 |
| date range | 2007-04-12 到 2017-11-10 |
| missing ratio | 0.0 |

所以数据不算太少；如果运行很快，主要是 quickstart 限制了 folds 和 panel rows。`--serious` 会使用更多 folds、seeds 和 models。

## 模型路线

| 层级 | 模型 | 定位 |
|---|---|---|
| baseline | `ridge` / `elasticnet` | 稳定、可解释的量化 baseline。 |
| nonlinear tabular | `hist_gradient_boosting` / `random_forest` | 捕捉非线性和特征交互，ETF 场景通常比裸 MLP 更实用。 |
| neural baseline | `mlp_sklearn` | 仅作为 simple neural baseline。 |
| future temporal neural | PatchTST / Temporal Transformer / TFT | 作为后续接入的 sequence model，不阻塞当前 case。 |

## 特征与目标

每个 ETF 每日一行 panel sample：

| 字段 | 含义 |
|---|---|
| `target` | 下一期 ETF return。 |
| `ret_lag_1`, `ret_lag_2` | 单资产短期滞后收益。 |
| `mom_5`, `mom_20` | 单资产短/中期 momentum。 |
| `vol_20` | 单资产 20 日波动。 |
| `market_ret_1`, `market_mom_5` | 横截面平均市场状态。 |
| `relative_mom_20` | 单资产相对市场 momentum。 |
| `ticker_code` | ETF identity code。 |

## 指标（Metrics）

| 指标 | 含义 |
|---|---|
| `composite_test_rmse_mean` | panel-level return prediction RMSE。 |
| `composite_direction_accuracy_mean` | 预测方向是否与真实收益同号。 |
| `composite_rank_ic_mean` | 每日横截面预测 ranking 与真实 return ranking 的 Spearman-like IC。 |
| `composite_rank_ic_std` | fold/seed 间 rank IC stability。 |
| `composite_hit_rate_mean` | top-k proxy portfolio 日收益为正的比例。 |
| `composite_net_sharpe_proxy_mean` | 扣除 turnover cost 后的 Sharpe proxy。 |
| `composite_max_drawdown_abs_mean` | proxy equity curve 最大回撤绝对值。 |
| `composite_turnover_proxy_mean` | top-k 持仓集合变化率。 |

## 结构（Structure）

| 路径 | 作用 |
|---|---|
| `run_case.py` | 兼容 mlblack case 风格的 CLI 入口。 |
| `run_solver.py` | 标准脚手架生成入口，已改为 ETF walk-forward runner。 |
| `build_solver.py` | 标准脚手架保留的 trainer assembly 入口。 |
| `config/scaffold.json` | 脚手架配置和 ETF case metadata。 |
| `mlblack.integrations.etf_temporal_forecast` | 可被 nsgablack 调用的稳定 integration surface。 |

## 运行

```powershell
cd C:\Users\hp\Desktop\mlblack
python examples\cases\etf_temporal_forecast\run_case.py --check
python examples\cases\etf_temporal_forecast\run_case.py --quickstart
python examples\cases\etf_temporal_forecast\run_case.py --serious
```

## 当前 quickstart 指标

`python examples\cases\etf_temporal_forecast\run_case.py --quickstart` 已在默认数据上验证过：

| 指标 | quickstart 值 |
|---|---:|
| dataset | `2668x40`, `2007-04-12..2017-11-10` |
| fold_count | `2` |
| models | `ridge`, `hist_gradient_boosting` |
| `composite_test_rmse_mean` | `0.0103047594` |
| `composite_direction_accuracy_mean` | `0.484625` |
| `composite_rank_ic_mean` | `0.0147542540` |
| `composite_rank_ic_std` | `0.0007056482` |
| `composite_hit_rate_mean` | `0.505` |
| `composite_net_sharpe_proxy_mean` | `-0.0487808271` |
| `composite_max_drawdown_abs_mean` | `0.08429593` |
| `composite_turnover_proxy_mean` | `0.7587169281` |

这说明数据量并不小；运行快是因为 quickstart 只跑 2 个 folds 和轻量模型。真正研究运行应使用 `--serious` 或显式增加 `--wf-max-folds` / seeds / models。

## 面试叙事

这个 case 的价值不是“用了复杂模型”，而是建立可验证的 ETF research loop：

1. 用 Ridge/ElasticNet 做稳定 baseline。
2. 用 tree/boosting 做主力非线性 tabular model。
3. 用 walk-forward 和 rank IC / hit rate / Sharpe proxy 检查是否真的有横截面预测信号。
4. 后续再让 nsgablack 搜 lane weights、top-k、thresholds 和 risk knobs。
