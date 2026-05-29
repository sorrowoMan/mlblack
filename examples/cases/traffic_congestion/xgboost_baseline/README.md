# XGBoost Baseline — Traffic CI Prediction

XGBoost baseline for predicting Congestion Index (CI) from 38 engineered features.

## 是否使用 mlblack / nsgablack
纯 mlblack (inner trainer only, no outer orchestration)

## 搜索向量

| 变量 | 含义 | 范围 |
|---|---|---|
| max_depth | Tree max depth | [2, 8] |
| learning_rate | Boosting learning rate | [0.02, 0.3] |

## 目标和指标

| 目标 | 方向 | 含义 |
|---|---|---|
| valid_rmse | minimize | Validation root mean squared error |

## 组件组合

| 层 | 组件 | 来源 |
|---|---|---|
| Problem | SupervisedEstimatorFitRegressionProblem | 框架 problem.supervised_estimator_fit |
| Representation | EstimatorSpecRepresentation + XGBoost factory | 框架 representation.estimator_spec |
| Adapter | EstimatorSpecSearchAdapter | 框架 adapter.estimator_spec_search |

## 效果对比

| Method | Valid RMSE | Valid R2 | Time | vs baseline |
|---|---|---|---|---|
| XGBoost (mlblack) | 9.03 | 0.850 | 77s | baseline |
| Symbolic Regression (mlblack) | *pending* | *pending* | *pending* | *pending* |

## 结构

| 路径 | 作用 |
|---|---|
| build_solver.py | 主入口：加载数据、组装 trainer、运行、输出报告 |
| build_solver.py | 脚手架配置 |
| catalog/ | 组件注册表 |

## 运行和验证

```powershell
python build_solver.py
python build_solver.py --check
```
