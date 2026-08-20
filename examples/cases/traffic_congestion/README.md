# traffic_congestion

这是交通拥堵研究的标准 Project 示例。

当前正式 Case 包括统计诊断、XGBoost 基线与统一控制面的符号回归：

- `symbolic_regression`：通过 `LearningSolver` 与 NSGABlack Adapter 执行的符号回归。
- `arimax_factor_attribution`、`gam_linearity_check`、`granger_causality_check`、
  `shap_contribution_check`：一次性诊断 Case。
- `xgboost_baseline`：第三方 estimator Problem 路径。

旧 `legacy_nowcasting` 运行器以及内部循环创建 Solver 的两个伪外层 Case 已删除。
复杂的正式嵌套符号学习示例由 `examples/cases/symbolic_orthogonal_nested`
通过标准父子 Case 协议提供。

正式入口：

```powershell
python examples/cases/traffic_congestion/run_project.py --check
python examples/cases/traffic_congestion/run_project.py --group symbolic --check --build-check
```

可运行单元位于 `cases/`；Case 内 `run_solver.py` 只用于调试。Project 编排和
`ResourceContext` 授权只由本级 Project 负责。
