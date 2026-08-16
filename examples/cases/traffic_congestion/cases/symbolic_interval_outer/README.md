# symbolic_interval_outer

Standard Case wrapper for the original native interval/CQR symbolic traffic implementation.

This Case keeps the full symbolic outer search inside the `traffic_congestion` Project:

- outer semantic layer: `nsgablack` subset/structure search;
- inner semantic layer: `mlblack` symbolic fitting and native quantile/CQR interval heads;
- Project substrate: `traffic_congestion/run_project.py` owns orchestration and L0 resource grants.

Run checks:

```powershell
python examples\cases\traffic_congestion\run_project.py --check --build-check
python examples\cases\traffic_congestion\cases\symbolic_interval_outer\run_solver.py --check
```

