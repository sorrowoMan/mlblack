# symbolic_mechanism_outer

Standard Case wrapper for the original traffic symbolic mechanism reconstruction implementation.

This is the real nested symbolic regression path:

- outer semantic layer: `nsgablack` subset/structure search;
- inner semantic layer: `mlblack` symbolic ridge / parameter fitting;
- Project substrate: `traffic_congestion/run_project.py` owns orchestration and L0 resource grants.

Run checks:

```powershell
python examples\cases\traffic_congestion\run_project.py --check --build-check
python examples\cases\traffic_congestion\cases\symbolic_mechanism_outer\run_solver.py --check
```

