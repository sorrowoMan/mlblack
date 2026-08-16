# Granger Causality

这个 Case 展示 Granger-causality 风格的统计诊断。它是单个 ML/统计 Case；只有当外层 Project 要搜索 lag window、feature pack 或 decision threshold 时，才需要 `nsgablack` 优化搜索语义。

## 边界

| 层 | 责任 |
| --- | --- |
| `problem/` | 统计目标和诊断语义 |
| `pipeline/` | 数据准备、lag/window 处理 |
| `build_solver.py` | canonical Case assembly entry |

旧 `pipeline/representation/` 引用只是 compatibility remnants。新的编码逻辑应放在 `pipeline/` 或框架语义模块。

## 运行

```powershell
python examples\cases\granger_causality\run_project.py --check --build-check
python examples\cases\granger_causality\cases\granger_causality\run_solver.py
python -m compileall -q examples\cases\granger_causality
```
