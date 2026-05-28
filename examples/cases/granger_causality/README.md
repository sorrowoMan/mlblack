# Granger Causality（格兰杰因果：稀疏 VAR 系数恢复）

一句话：验证格兰杰因果检验可以看作稀疏 VAR(1) 系数矩阵的梯度优化问题。恢复的 A[i,j] != 0 表示变量 j 格兰杰导致变量 i。

## 是否使用 mlblack / nsgablack

纯 mlblack。不使用 nsgablack。

## 这个 case 验证什么

格兰杰因果被表达为 VAR(1) 系数矩阵 A (n_vars × n_vars) 上的梯度优化：

- Representation 将扁平向量编解码为 (n_vars, n_vars) VAR 系数矩阵。
- Problem 计算 VAR(1) 一步预测 MSE + L1 稀疏惩罚，返回解析梯度 dL/dA。
- GradientDescentAdapter（框架 adapter.gradient_descent）驱动参数更新。
- L1 惩罚项自动收缩零系数，恢复稀疏因果图。

能力信号：梯度下降 + L1 在标准化 VAR(1) 数据上成功恢复真实的稀疏因果结构，比 OLS 更准确地识别因果边（更少的假阳性）。

## 搜索向量

| 变量 | 维度 | 范围 |
|---|---|---|
| A (VAR 系数矩阵) | n_vars × n_vars | 无界（L1 拉向稀疏） |

## 目标和指标

| 目标 | 方向 | 含义 |
|---|---|---|
| MSE + L1 * sum\|A\| | minimize | 预测误差 + 稀疏惩罚 |

## 组件组合

| 层 | 组件 | 来源 |
|---|---|---|
| Problem | GrangerCausalityProblem (VAR MSE + L1 + 解析梯度) | 自定义 |
| Representation | GrangerRepresentation (flat ↔ A 矩阵) | 自定义 light codec |
| Adapter | GradientDescentAdapter | 框架 adapter.gradient_descent |

## 效果对比

| Method | MAE vs OLS | Correlation vs OLS | Edges Found | False Positives | Time |
|---|---|---|---|---|---|
| OLS (baseline) | 0.0 | 1.0 | 9 | 3 | -- |
| mlblack GD+L1 (1500 steps) | 0.0047 | 0.9999 | 7 | 1 | 0.30s |

500 时间步 × 5 变量的合成 VAR(1) 数据上，GD+L1 以 0.0047 的 MAE 精确匹配 OLS 基线，同时 L1 正则化将假阳性边从 3 个减少到 1 个。6 个真实因果边全部正确识别。

## 结构

| 路径 | 作用 |
|---|---|
| `build_trainer.py` | Assembly 入口 + 合成数据生成 + 标准化 + OLS 对比。 |
| `problem/granger_causality_problem.py` | VAR(1) MSE 损失 + L1 稀疏 + 解析梯度 dL/dA。 |
| `representation/granger_representation.py` | A 矩阵编解码。 |

## 运行和验证

```powershell
python build_trainer.py
python -m compileall -q .
```
