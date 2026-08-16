# Matrix Factorization

这个 Case 把矩阵分解表达成 `mlblack` 的 ML 语义任务。它本身不需要 `nsgablack`；只有当外层 Project 要搜索 rank、regularization、初始化或预算 tradeoff 时，才需要优化搜索 Case。

## 边界

| 层 | 责任 |
| --- | --- |
| `problem/` | sparse observed-entry MSE 和 feedback |
| `pipeline/` | 数据准备和 model-state encode/decode helper |
| `adapter/` | gradient-based 参数更新 |
| `bias/` | 可选 regularization，例如 state L2 |
| `build_solver.py` | canonical Case assembly entry |

旧 `pipeline/representation/` 引用只是 compatibility remnants。新的模型编码逻辑应放在 Case `pipeline/` 或框架语义模块中说明，不作为正式 Case-level 目录。

## 运行

```powershell
python examples\cases\matrix_factorization\run_project.py --check --build-check
python examples\cases\matrix_factorization\cases\matrix_factorization\run_solver.py
python -m compileall -q examples\cases\matrix_factorization
```
