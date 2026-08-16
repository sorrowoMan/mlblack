# XGBoost Baseline — Traffic CI Prediction

This is the XGBoost baseline case for predicting traffic Congestion Index (CI).
It uses the framework `xgboost` preset (`tree_boosting_estimator_search`)
which tunes `max_depth` and `learning_rate` via `EstimatorSpecSearchAdapter`.

## Quickstart

```powershell
python examples\cases\traffic_congestion\run_project.py --check
python examples\cases\traffic_congestion\run_project.py --check --build-check
```

For single-Case debugging only:

```powershell
python examples\cases\traffic_congestion\cases\xgboost_baseline\run_solver.py
python examples\cases\traffic_congestion\cases\xgboost_baseline\run_solver.py --check
```

## What this baseline covers
- Full CI feature matrix (weather, holiday, AQI, lags, rolling stats, temporal encodings)
- 80/20 random train/valid split
- XGBoost with DE-style spec search on 8 candidates over 50 steps
- Report: train/valid RMSE and R2

## Moving beyond this baseline
- Symbolic regression baseline (mlblack `preset.orthogonal_linear_point` + symbolic problem)
- Nested optimization: nsgablack outer ensemble selection + mlblack inner XGBoost fitting
