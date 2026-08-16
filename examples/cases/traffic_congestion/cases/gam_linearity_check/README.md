# Traffic Congestion: GAM Linearity Check

This is a standalone diagnostic for checking whether linear assumptions are reasonable for a traffic congestion prediction task.

## Boundary

- `build_solver.py` is the script entry retained for compatibility.
- The scaffold-like subdirectories are placeholders and should not be treated as formal assembly surfaces.
- If this diagnostic becomes part of a multi-Case workflow, use the shared Project substrate for ordering, artifacts, and resource grants.

## Run

```powershell
python examples\cases\traffic_congestion\run_project.py --check --build-check
python examples\cases\traffic_congestion\cases\gam_linearity_check\run_solver.py
python examples\cases\traffic_congestion\cases\gam_linearity_check\run_solver.py --n-knots 8 --degree 4
```
