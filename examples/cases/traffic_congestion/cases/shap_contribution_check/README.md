# Traffic Congestion: SHAP Contribution Check

This is a standalone feature-contribution diagnostic comparing linear, XGBoost, SHAP, and permutation-importance signals. It is analysis material, not a formal cross-Case orchestration surface.

## Boundary

- `build_solver.py` is the script entry retained for compatibility.
- The scaffold-like subdirectories are placeholders and should not be treated as formal assembly surfaces.
- If this analysis becomes part of a larger Project, expose it as a standard Case and pass artifacts through the shared Project substrate.

## Run

```powershell
python examples\cases\traffic_congestion\run_project.py --check --build-check
python examples\cases\traffic_congestion\cases\shap_contribution_check\run_solver.py
python examples\cases\traffic_congestion\cases\shap_contribution_check\run_solver.py --n-estimators 300 --top-k 15
python -m compileall -q examples\cases\traffic_congestion
```
