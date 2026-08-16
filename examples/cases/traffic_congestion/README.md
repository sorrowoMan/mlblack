# traffic_congestion

This directory is the standard Project wrapper for the traffic congestion study.

The real traffic symbolic regression implementation is now attached here as standard Cases:

- `symbolic_mechanism_outer`: original mechanism reconstruction / equation discovery path.
- `symbolic_interval_outer`: original native interval / CQR forecasting path.
- `symbolic_regression`: lightweight linear symbolic-compatible baseline only.

Formal entrypoint:

```powershell
python examples/cases/traffic_congestion/run_project.py --check
python examples/cases/traffic_congestion/run_project.py --group symbolic --check --build-check
```

Runnable Case internals are under `cases/`. Case-local `run_solver.py` remains a debug entry only; Project orchestration and ResourceContext grants belong here.

Project outputs from the restored symbolic implementation are written under `out/`.
