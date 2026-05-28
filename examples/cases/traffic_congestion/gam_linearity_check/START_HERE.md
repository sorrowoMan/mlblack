# START_HERE

This is a **diagnostic tool**, not a trainer scaffold.

## Purpose

Validates whether linear model assumptions hold for the traffic CI dataset by comparing linear regression coefficients against GAM (Generalized Additive Model) partial dependence curves using B-splines.

## Run

```powershell
python build_trainer.py
```

Options:

```
--n-knots 6     B-spline knot count
--degree 3      B-spline degree
--top-k 8       Top features to highlight
```

## What it checks

1. Trains a linear regression baseline and records per-feature coefficients
2. Builds a GAM via B-spline expansion + RidgeCV regression
3. Compares RMSE improvement: if GAM >> Linear, nonlinear patterns exist
4. Compares partial dependence ranges per feature against linear projections
5. Flags features where GAM span exceeds 1.5x linear span or correlation < 0.8

## Interpretation

- If GAM RMSE is close to Linear RMSE: linear assumptions broadly hold
- If many features flagged as nonlinear: consider nonlinear models (XGBoost, symbolic regression)
- This is a preprocessing diagnostic before model selection, not a final model
