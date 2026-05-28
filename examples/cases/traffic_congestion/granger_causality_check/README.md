# Granger 因果检验 (Granger Causality Check)

一句话：检验交通拥堵指数 CI 与外部因素（天气、AQI、节假日、风力）之间是否存在统计显著的 Granger 因果关系。

## 是否使用 mlblack / nsgablack

纯诊断工具，不涉及 mlblack/nsgablack 组件组装。

## 这个 case 验证什么

通过 pairwise Granger causality test 和 VAR 模型，验证：
1. 外部因素（天气/AQI/节假日/风力）是否能帮助预测 CI 变化（X → CI）
2. CI 是否能反向预测外部因素（CI → X），排除伪因果
3. 多变量 VAR 最优滞后阶数

## 搜索向量

| 变量 | 含义 | 范围 |
|---|---|---|
| lag | Granger 检验滞后阶数 | 1-7 |

## 目标和指标

| 目标 | 方向 | 含义 |
|---|---|---|
| F-test p-value | < 0.05 | Granger 因果显著 |

## 组件组合

| 层 | 组件 | 来源 |
|---|---|---|
| 统计检验 | statsmodels.granger causalitytests | 外部库 |
| 备选 | scipy.signal.correlate | 外部库 |
| 数据加载 | pandas CSV | 内置 |

## 效果对比

### Granger Causality (X → CI)

| Factor | Best Lag | F-stat | p-value | Significant? |
|---|---|---|---|---|
| weather_dummy | 7 | 1.13 | 0.3420 | no |
| wind | 5 | 0.65 | 0.6618 | no |
| aqi | 6 | 1.67 | 0.1254 | no |
| life_impact | 1 | 1.26 | 0.2619 | no |
| is_bad_weather | 7 | 1.14 | 0.3349 | no |
| is_aqi_high | 7 | 0.93 | 0.4836 | no |
| is_holiday_near | 1 | 0.20 | 0.6565 | no |
| is_holiday_mid | 7 | 1.13 | 0.3413 | no |
| is_nonwork_weekend | 7 | 1.27 | 0.2607 | no |
| is_holiday_day_or_window | 7 | 1.70 | 0.1054 | no |

### Reverse Direction (CI → X)

| Factor | Best Lag | F-stat | p-value | Significant? |
|---|---|---|---|---|
| weather_dummy | 2 | 1.62 | 0.1974 | no |
| wind | 1 | 19.17 | 0.0000 | **YES** |
| aqi | 2 | 0.60 | 0.5466 | no |
| life_impact | 2 | 2.13 | 0.1195 | no |
| is_bad_weather | 2 | 1.55 | 0.2135 | no |
| is_aqi_high | 7 | 2.80 | 0.0068 | **YES** |
| is_holiday_near | 1 | 1.18 | 0.2770 | no |
| is_holiday_mid | 2 | 0.26 | 0.7722 | no |
| is_nonwork_weekend | 7 | 1.11 | 0.3515 | no |
| is_holiday_day_or_window | 7 | 1.14 | 0.3344 | no |

### VAR 多变量模型

| Variables | Optimal Lag | AIC |
|---|---|---|
| ci, wind, aqi | 7 | 8.7 |

### 关键发现

- 外部因素（天气、AQI、节假日）**不能**显著预测 CI 变化
- CI **反向预测**风力 (lag=1, F=19.17, p<0.0001) 和 AQI 高位状态 (lag=7, F=2.80, p=0.0068)
- 这暗示 CI 可能是交通流量的联动信号，而非被外部因素驱动的响应变量

## 结构

| 路径 | 作用 |
|---|---|
| build_trainer.py | Granger 因果检验主脚本 |
| run_trainer.py | CLI 入口（委托 build_trainer.main） |
| START_HERE.md | 快速入门 |
| README.md | 本文档 |
| adapter/, bias/, capabilities/, pipeline/, problem/, representation/ | 脚手架占位目录 |

## 运行和验证

```powershell
python build_trainer.py --maxlag 7
```
