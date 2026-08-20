# Traffic Congestion: Linear Symbolic Baseline

这是 traffic Project 中正式保留的符号回归 Case。它通过统一
`LearningSolver -> ComposableSolver -> Adapter` 控制面运行；更复杂的父子
Case 符号学习示例位于 `examples/cases/symbolic_orthogonal_nested`。

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
python examples\cases\traffic_congestion\run_project.py --group symbolic
python -m compileall -q examples\cases\traffic_congestion
```
