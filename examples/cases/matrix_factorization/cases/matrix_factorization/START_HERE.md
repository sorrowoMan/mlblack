# START_HERE

## 1) 这个 case 验证什么
`matrix_factorization` 验证矩阵分解可以作为梯度优化问题。

- mlblack 将评分矩阵分解为 `U @ V.T`，其中搜索向量是 `[U.flatten(), V.flatten()]`。
- Problem 在观测条目上计算稀疏 MSE 并回传解析梯度。
- Adapter 使用框架内置的梯度下降。
- 可选：Bias 使用框架 `StateL2Bias` 平滑状态；NMF 模式通过 `representation.repair()` 做非负投影。

## 2) 运行
```powershell
Set-Location "C:\Users\hp\Desktop\mlblack"
python examples\cases\matrix_factorization\run_project.py --check --build-check
python examples\cases\matrix_factorization\cases\matrix_factorization\run_solver.py
```

## 3) 关键指标
| 指标 | 含义 |
|---|---|
| MSE (observed) | 观测条目上的均方误差，越低越好。 |
| RMSE (observed) | 观测条目上的均方根误差。 |
| Reconstruction RMSE | 全矩阵重建误差（含未观测条目）。 |
| GD / SVD RMSE | 梯度下降 vs sklearn TruncatedSVD 的对比。 |

## 4) 预期信号
从合成低秩数据开始，MSE 通过梯度下降稳定下降，几百步后 GD RMSE 应接近或优于 SVD 基线。
