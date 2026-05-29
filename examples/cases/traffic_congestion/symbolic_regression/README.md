# Symbolic Regression / Traffic CI（交通拥堵指数符号公式拟合 — 线性基线）

一句话：使用 mlblack 梯度下降适配器，在符号回归兼容的线性系数向量上训练 MSE 回归基线，为后续符号表达式搜索提供对比基准。

## 是否使用 mlblack / nsgablack

纯 mlblack。不使用 nsgablack。本 case 仅构建 single inner trainer 的线性回归 baseline。

## 这个 case 验证什么

在真实交通 CI 数据上建立线性回归基准：

- Representation 将扁平系数向量（含截距项）编解码，兼容后续符号表达式的系数空间。
- Problem 计算 MSE 损失 + 解析梯度 dL/dw = (2/N) X^T (Xw - y)，X 含 bias 列。
- GradientDescentAdapter 驱动参数更新，无额外正则化。
- 输出 Train/Valid RMSE、截距和 Top 5 特征系数，为符号表达式搜索提供对比基线。

能力信号：GradientDescentAdapter 在 z-score 标准化后的 30 维交通特征上收敛至接近 sklearn OLS 解析解，验证 mlblack 最小线性训练链路可用于后续符号回归。

## 搜索向量

| 变量 | 维度 | 范围 |
|---|---|---|
| w (截距 + 线性系数) | 31 | [-0.03, 0.03] 初始化，无界更新 |

## 目标和指标

| 目标 | 方向 | 含义 |
|---|---|---|
| MSE(w) = mean((Xw - y)^2) | minimize | 预测误差 |

## 组件组合

| 层 | 组件 | 来源 |
|---|---|---|
| Problem | CIDirectRegressionProblem (MSE + 解析梯度 + bias 列) | 自定义 |
| Representation | CIDirectRepresentation (flat 系数向量 / 截距 + 权重) | 自定义 light codec |
| Adapter | GradientDescentAdapter | 框架 adapter.gradient_descent |
| Pipeline | ZScoreNormalizeComponent | 框架 pipeline |

## 效果对比

| Method | Train RMSE | Valid RMSE | Time | vs baseline |
|---|---|---|---|---|
| sklearn LinearRegression (OLS) | 8.74 | 9.13 | 0.02s | baseline |
| mlblack GD (50 steps, lr=0.01) | 13.81 | 14.60 | 0.01s | 1.60x |
| mlblack GD (200 steps, lr=0.01) | 8.84 | 9.10 | 0.04s | 1.00x |
| mlblack GD (500 steps, lr=0.01) | 8.79 | 9.01 | 0.10s | 0.99x |

> 500 步 GD 截距收敛至 28.20（sklearn: 28.20），系数与 OLS 高度一致。200 步已接近解析解，验证 GD 适配器在符号回归兼容系数空间中的可行性。本表为线性基线；后续符号表达式搜索将在此基础上对比。

## 结构

| 路径 | 作用 |
|---|---|
| `build_solver.py` | Assembly 入口 + 数据加载 + z-score 标准化 + GD 训练 + RMSE 报告 + 系数分析。 |
| `assembly/` | 项目 scaffold 配置。 |
| `catalog/` | Catalog entries 注册。 |
| `problem/` | 自定义问题目录（本 case 内置 Problem 在 build_trainer.py）。 |
| `pipeline/representation/` | 自定义表示目录（本 case 内置 Representation 在 build_trainer.py）。 |

## 运行和验证

```powershell
cd C:\Users\hp\Desktop\mlblack\examples\cases\traffic_congestion\symbolic_regression
python build_solver.py --steps 50
python build_solver.py --steps 200
python build_solver.py --check
python -m compileall -q .
```
