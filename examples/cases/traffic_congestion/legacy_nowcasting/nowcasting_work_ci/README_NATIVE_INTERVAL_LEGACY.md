# Native Interval Variant (Copy)

This is a copied script variant that implements:

1. Native low/high interval heads via quantile regression (`QuantileRegressor`)
2. Conformal calibration (CQR-style) on a calibration split
3. Fallback to symmetric residual interval only when quantile fitting fails

Script:

- [run_nowcasting_symbolic_subset_bridge_work_ci_native_interval.py](/C:/Users/hp/Desktop/mlblack/nowcasting_work_ci/run_nowcasting_symbolic_subset_bridge_work_ci_native_interval.py)

## Key flags

```powershell
--interval-method native_quantile_cqr
--interval-alpha 0.1
--interval-calib-ratio 0.2
--interval-quantile-l2 1e-4
--drop-same-day-flow-speed-occ 1
```

## Notes

- This is still the **nowcasting** package when using `ci_interval_opt_table.csv`.
- The native interval script now defaults to dropping same-day `total_flow/avg_speed/avg_occ` to reduce concurrent leakage in prediction-style runs.
- For strict forecasting evaluation, switch dataset to no-same-day versions.
