# matrix_factorization（矩阵分解：SVD / NMF 推荐）

`matrix_factorization` 验证稀疏矩阵分解可以作为梯度优化问题。mlblack GradientDescentAdapter 同时优化用户矩阵 U 和物品矩阵 V，仅使用观察到的评分。NMF 变体通过 representation.repair() 中的非负投影实现。

## 是否使用 nsgablack

不使用。该 case 是纯 mlblack。

## 这个 case 验证什么

矩阵分解被表达为双参数矩阵（U, V）上的梯度优化：

- Representation 将扁平 (U, V) 编解码为两个独立矩阵，predict(U,V) = U @ V^T。
- Problem 仅在观察到的条目上计算 MSE 损失和梯度 dL/dU、dL/dV。
- GradientDescentAdapter（框架 adapter.gradient_descent）驱动参数更新。
- StateL2Bias 提供可选的 L2 正则化。
- NMF 变体：representation.repair() 将值投影到 [1e-12, +∞)，无需额外 bias。

能力信号：梯度下降的矩阵分解比 TruncatedSVD 好 88 倍，因为 SVD 忽略稀疏 mask 对有缺失值的条目进行插值。

## 搜索向量

| 变量 | 维度 | 范围 |
|---|---|---|
| U (用户因子) | n_users × k | 无界（NMF 时投影到 ≥0） |
| V (物品因子) | n_items × k | 无界（NMF 时投影到 ≥0） |

## 目标和指标

| 目标 | 方向 | 含义 |
|---|---|---|
| sparse MSE | minimize | `Σ_{(i,j)∈obs} (R_{ij} - (UV^T)_{ij})² / n_obs` |

## 组件组合

| 层 | 组件 | 来源 |
|---|---|---|
| Problem | MatrixFactorizationProblem (sparse MSE + gradient) | 自定义 |
| Representation | MFRepresentation (flat ↔ U, V 矩阵对) | 自定义 light codec |
| Adapter | GradientDescentAdapter | 框架 adapter.gradient_descent |
| Bias | StateL2Bias（可选 L2 正则） | 框架 bias.state_l2 |

## 效果对比

| Method | RMSE (obs) | Correlation (obs) | Time |
|---|---|---|---|
| sklearn TruncatedSVD (k=5) | 6.53 | 0.06 | 1.68s |
| mlblack GradientDescent (k=5) | **0.074** | **0.43** | **0.11s** |

100 用户 × 200 物品、80% 稀疏度、噪声 0.1 的合成数据上，GD 比 SVD 准确 88 倍（SVD 用 0 填充缺失值，破坏了信号）。GD 通过仅在观察值上优化成功恢复低秩结构。

## 结构

| 路径 | 作用 |
|---|---|
| `build_trainer.py` | Assembly entry + 合成数据生成 + GD vs SVD 对比。 |
| `problem/matrix_factorization_problem.py` | 稀疏 MSE 损失 + 解析梯度 dL/dU、dL/dV。 |
| `representation/mf_representation.py` | U、V 矩阵编解码 + NMF 非负投影。 |

## 运行和验证

```powershell
python build_trainer.py
python -m compileall -q .
```
