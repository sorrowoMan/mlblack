# Native Interval Mode (Scaffolded)

`native_quantile_cqr` is now a mode of the unified entry:

- Main entry: [run_solver.py](/C:/Users/hp/Desktop/mlblack/nowcasting_work_ci/run_solver.py)
- Compatibility wrapper: [run_nowcasting_symbolic_subset_bridge_work_ci_native_interval.py](/C:/Users/hp/Desktop/mlblack/nowcasting_work_ci/run_nowcasting_symbolic_subset_bridge_work_ci_native_interval.py)

## Run

```powershell
python C:\Users\hp\Desktop\mlblack\nowcasting_work_ci\run_solver.py `
  --interval-method native_quantile_cqr `
  --interval-alpha 0.1 `
  --interval-calib-ratio 0.2 `
  --interval-quantile-l2 1e-4
```

## Important flags

```powershell
--interval-method native_quantile_cqr
--interval-alpha 0.1
--interval-calib-ratio 0.2
--interval-quantile-l2 1e-4
--drop-same-day-flow-speed-occ 1
```

## Note

This mode uses native low/high quantile heads + conformal calibration (CQR style), with safe fallback to symmetric residual interval if quantile fitting fails.
