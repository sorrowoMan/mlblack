# Traffic Congestion: Granger Causality Check

This is a standalone statistical diagnostic for traffic congestion factors. It does not assemble a full `mlblack` or `nsgablack` workflow.

## Boundary

- `build_solver.py` is the script entry retained for compatibility.
- The scaffold-like subdirectories are placeholders and should not be treated as formal assembly surfaces.
- If this diagnostic becomes part of a multi-Case workflow, wrap it as a standard Case and let the shared Project substrate handle ordering and resources.

## Run

```powershell
python examples\cases\traffic_congestion\run_project.py --check --build-check
python examples\cases\traffic_congestion\cases\granger_causality_check\run_solver.py --maxlag 7
```
