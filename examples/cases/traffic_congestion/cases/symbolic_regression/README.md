# Traffic Congestion: Linear Symbolic Baseline

This is a lightweight linear symbolic-compatible baseline. It is not the full nested symbolic regression implementation.

The restored full symbolic implementation lives in:

- `cases/symbolic_mechanism_outer`
- `cases/symbolic_interval_outer`

## Boundary

| Layer | Responsibility |
| --- | --- |
| `problem/` | regression objective, metrics, and feedback |
| `pipeline/` | feature preparation and model-state encode/decode helpers |
| `adapter/` | fitting/update behavior |
| `plugins/` | audit/report side effects |
| `build_solver.py` | canonical Case assembly entry |

Assembly belongs in `build_solver.py`, and Project-level ordering/resources belong to `run_project.py` plus `project_config.py`; model encoding belongs under `pipeline/` or framework semantic modules.

## Run

```powershell
python examples\cases\traffic_congestion\run_project.py --check --build-check
python examples\cases\traffic_congestion\cases\symbolic_regression\run_solver.py --steps 50
python examples\cases\traffic_congestion\cases\symbolic_regression\run_solver.py --steps 200
python examples\cases\traffic_congestion\run_project.py --check
python -m compileall -q examples\cases\traffic_congestion
```
