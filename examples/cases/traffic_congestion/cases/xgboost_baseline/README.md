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
| Adapter | GaussianSearchAdapter | NSGABlack `search.random_gaussian` |

## 效果对比

| Method | Valid RMSE | Valid R2 | Time | vs baseline |
|---|---|---|---|---|
| XGBoost (mlblack) | 9.03 | 0.850 | 77s | baseline |
| Symbolic Regression (mlblack) | *pending* | *pending* | *pending* | *pending* |

## 结构

| 路径 | 作用 |
|---|---|
| build_solver.py | canonical Case assembly entry |
| run_solver.py | Case-local debug entry |
| catalog/ | 组件注册表 |

## 运行和验证

```powershell
python examples\cases\traffic_congestion\run_project.py --check --build-check
python examples\cases\traffic_congestion\cases\xgboost_baseline\run_solver.py
python examples\cases\traffic_congestion\run_project.py --check
```
