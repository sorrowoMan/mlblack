# nowcasting_work_ci

`nowcasting_work_ci` is the packaged nowcasting scenario scaffold that connects:

- `nsgablack_side/` for outer solver/problem assembly
- `mlblack_side/` for model/runtime/training assembly

## Preferred entry

Use [`run.py`](/C:/Users/hp/Desktop/mlblack/nowcasting_work_ci/run.py) as the long-term CLI entry.

```powershell
python C:\Users\hp\Desktop\mlblack\nowcasting_work_ci\run.py --interval-method native_quantile_cqr
python C:\Users\hp\Desktop\mlblack\nowcasting_work_ci\run.py --interval-method symmetric_residual
```

## Stable public surface

These top-level files are the public surface we intend to keep stable:

- [`run.py`](/C:/Users/hp/Desktop/mlblack/nowcasting_work_ci/run.py): preferred CLI entry
- [`build_solver.py`](/C:/Users/hp/Desktop/mlblack/nowcasting_work_ci/build_solver.py): stable solver-build import surface
- [`assembly.py`](/C:/Users/hp/Desktop/mlblack/nowcasting_work_ci/assembly.py): stable runtime-assembly forwarder

Scenario experiment entry:

- [`run_nowcasting_orthogonal_symbolic_work_ci.py`](/C:/Users/hp/Desktop/mlblack/nowcasting_work_ci/run_nowcasting_orthogonal_symbolic_work_ci.py): orthogonal-basis-first symbolic experiment on `work_ci`

## Deprecated surface

These top-level files remain for compatibility only and should not receive new business logic:

- [`run_solver.py`](/C:/Users/hp/Desktop/mlblack/nowcasting_work_ci/run_solver.py)
- [`aggregate_and_plot_results.py`](/C:/Users/hp/Desktop/mlblack/nowcasting_work_ci/aggregate_and_plot_results.py)
- [`run_deterministic_smoke_regression.py`](/C:/Users/hp/Desktop/mlblack/nowcasting_work_ci/run_deterministic_smoke_regression.py)
- [`run_nowcasting_symbolic_subset_bridge_work_ci.py`](/C:/Users/hp/Desktop/mlblack/nowcasting_work_ci/run_nowcasting_symbolic_subset_bridge_work_ci.py)
- [`run_nowcasting_symbolic_subset_bridge_work_ci_native_interval.py`](/C:/Users/hp/Desktop/mlblack/nowcasting_work_ci/run_nowcasting_symbolic_subset_bridge_work_ci_native_interval.py)

Real legacy implementations live in `compat/`. Operational helpers live in `tools/`.

## Runtime outputs

Generated artifacts should go under:

- `C:\Users\hp\Desktop\mlblack\_scenario_runs\nowcasting_work_ci\`

Optional override:

- `MLBLACK_SCENARIO_RUNS_ROOT`

## Internal docs

Internal architecture and runtime notes now live under [`docs/README.md`](/C:/Users/hp/Desktop/mlblack/nowcasting_work_ci/docs/README.md).

Top-level layout and surface rules are defined in [`STANDARD_LAYOUT.md`](/C:/Users/hp/Desktop/mlblack/nowcasting_work_ci/STANDARD_LAYOUT.md).
