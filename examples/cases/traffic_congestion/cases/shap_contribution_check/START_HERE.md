# START_HERE

## Purpose

This case validates whether **feature importance rankings are consistent** between XGBoost
(nonlinear tree ensemble) and linear regression. It is a model-paradigm robustness check for
attribution analysis.

## What It Tests

- **Linear**: |coefficient| as feature importance
- **XGBoost**: built-in gain-based feature importance
- **SHAP (TreeSHAP)**: mean |SHAP value| across samples
- **Permutation**: drop-column importance on XGBoost

If rankings align (Spearman > 0.7, top-k agreement >= 3/4), linear attribution conclusions are
trustworthy even under nonlinear data.

## Run

```powershell
python examples\cases\traffic_congestion\run_project.py --check --build-check
python examples\cases\traffic_congestion\cases\shap_contribution_check\run_solver.py
# With options:
python examples\cases\traffic_congestion\cases\shap_contribution_check\run_solver.py --n-estimators 300 --top-k 15
```

## Dependencies

- Required: numpy, pandas, scikit-learn
- Optional: xgboost, shap, scipy (script handles missing gracefully)

```powershell
pip install xgboost shap scipy
```

## Verify

```powershell
python -m compileall -q .
```
