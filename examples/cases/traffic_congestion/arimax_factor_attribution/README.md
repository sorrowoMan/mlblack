# ARIMAX Factor Attribution (交通拥堵指数因子归因)

Decompose CI variation into contributions from different factor groups using ARIMAX models.

## Framework

This is a standalone analysis case within the mlblack project, using statsmodels ARIMAX + factor group decomposition. No nsgablack orchestration is used here; the analysis is purely statistical.

## What This Case Validates

- Marginal contribution of each factor group to CI prediction (|coeff| sum / total)
- Predictive importance of each factor group via drop-one-group delta AIC
- Which external regressors matter most for short-term CI forecasting

## Search Vector

N/A -- this is a fixed ARIMAX model, not an optimization-driven search.

## Objectives and Metrics

| Objective | Direction | Meaning |
|---|---|---|
| AIC | minimize | Model quality (lower = better fit with penalty for complexity) |
| Contribution % | maximize | Fraction of total |coefficient| weight per group |

## Component Composition

| Layer | Component | Source |
|---|---|---|
| Problem | N/A -- direct ARIMAX fit | statsmodels |
| Representation | N/A -- raw feature columns | CSV data |
| Adapter | N/A -- MLE estimation | statsmodels ARIMA.fit() |

## Effect Comparison

### Contribution % by Factor Group

| Factor Group | Contribution % | Interpretation |
|---|---|---|
| Time_Cyclic | 71.9% | Day-of-week / day-of-year seasonality dominates |
| Holiday | 13.7% | Holiday / non-work day effects |
| CI_Lags | 12.0% | Historical CI momentum |
| Life | 1.5% | Life-impact events |
| Weather | 0.4% | Weather-driven CI shifts |
| CI_Rolling | 0.3% | Rolling mean/std patterns |
| AQI | 0.2% | Air quality impact |

### Drop-One-Group Impact (Delta AIC)

Full ARIMAX(2,0,1): AIC=12195.5, BIC=12363.9 (1689 samples, 26 external features).

| Dropped Group | Delta AIC | Impact |
|---|---|---|
| Weather | +5.8 | HIGH |
| CI_Rolling | +4.8 | HIGH |
| AQI | +1.0 | LOW |
| CI_Lags | -3.3 | LOW |
| Holiday | -5.1 | LOW |
| Time_Cyclic | -7.5 | LOW |
| Life | -2.0 | LOW |

### Key Findings

- **Time_Cyclic** features account for 71.9% of coefficient magnitude, but removing them does NOT degrade AIC (delta -7.5), suggesting the model over-weights these features while AR/MA terms absorb the seasonal signal.
- **Weather** and **CI_Rolling** are the critical groups: removing either raises AIC by +4.8 or more.
- **Holiday** group has the second-largest coefficient sum (13.7%) but negative delta AIC when removed - a counterintuitive result likely due to multicollinearity with Time_Cyclic features.
- Convergence warnings indicate the model with 26 standardized exogenous regressors may benefit from dimension reduction or regularization.

## Structure

| Path | Role |
|---|---|
| build_solver.py | ARIMAX factor attribution script |
| START_HERE.md | Quickstart and documentation |
| README.md | This file |

## Run

```powershell
python build_solver.py
```
