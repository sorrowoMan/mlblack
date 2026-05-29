# START_HERE: ARIMAX Factor Attribution

Decompose Traffic CI variation into contributions from distinct factor groups using ARIMAX models.

## What This Does

- Loads CI data without flow/speed/occ (no leakage features)
- Builds ARIMAX(p,d,q) with external regressors grouped by category
- Computes marginal contribution (|coefficient| sum) per factor group
- Tests factor importance by dropping one group at a time and measuring delta AIC

## Factor Groups

| Group | Features | Count |
|---|---|---|
| Weather | weather_dummy, wind, is_bad_weather | 3 |
| AQI | aqi, is_aqi_high | 2 |
| Holiday | is_holiday_near, is_holiday_mid, is_nonwork_weekend, is_holiday_day_or_window | 4 |
| Life | life_impact | 1 |
| CI_Lags | ci_lag1..lag28 (8 lags) | 8 |
| CI_Rolling | ci_roll3/7/14 means, ci_roll7_std | 4 |
| Time_Cyclic | dow_sin, dow_cos, doy_sin, doy_cos | 4 |

## Run

```powershell
python build_solver.py
python build_solver.py --ar-order 3 --ma-order 2
python build_solver.py --diff 1
```

## Output

1. Full model AIC/BIC
2. Contribution % table: which factor group drives CI the most
3. Drop-one-group impact table: delta AIC per removed group

## Requirements

- pandas, numpy, scikit-learn
- statsmodels (optional; graceful fallback to LinearRegression)
