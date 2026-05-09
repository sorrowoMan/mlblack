# Architecture Split (nsgablack / mlblack)

Current split:

- `nsgablack_side/`
  - `problem/config.py`
  - `pipeline/config.py`
  - `adapter/config.py`
  - `evaluation/config.py`
  - `plugins/config.py`
  - `build_solver.py`
  - `run_solver.py`
- top-level framework layers
  - `bias/`
    - `config.py`
    - `branch_policy.py`
    - `objective_policy.py`
    - `dynamic_pool_policy.py`
  - `model/`
    - `config.py`
    - `interval_fit.py`
  - `evaluation/`
    - `config.py`
    - `problem_callbacks.py`
  - `plugins/`
    - `config.py`
    - `report_writer.py`
    - `report_writer_plugin.py`
  - `problem/`
    - `contracts.py`
    - `bridge.py`
    - `proxy.py`
  - `pipeline/`
    - `base.py`
    - `config.py`
    - `feature_space_builder.py`
    - `identity.py`
    - `zscore.py`
    - `feature_space.py`
  - `workflow/`
    - `hook_bus.py`
    - `orchestrator.py`
- `mlblack_side/`
  - `problem/problem_model.py`
  - `runtime/config.py`
  - `runtime/assembly.py`
  - `runtime/build_runtime.py`
  - `runtime/actions/*`
  - `runtime/stages.py`
  - `runtime/workflow.py`
  - `runtime/workflow_runtime.py`
- top-level
  - `run.py` as unified entrypoint
  - `compat/` for legacy CLI implementations
  - `tools/` for smoke/report helpers
  - generated artifacts live outside package root at `_scenario_runs/nowcasting_work_ci/`

Rule of thumb:

- generic policy logic goes to top-level `bias/`
- generic model fitting / interval strategy logic goes to top-level `model/`
- generic evaluation callback/runtime logic goes to top-level `evaluation/`
- generic report / hook side-effects go to top-level `plugins/`
- generic problem bridge/proxy contracts go to top-level `problem/`
- generic symbolic feature-space logic goes to top-level `pipeline/feature_space.py`
- feature assembly of lag/drop/temporal/regime/candidate-pool goes to top-level `pipeline/feature_space_builder.py`
- generic control-plane orchestration and hook bus go to top-level `workflow/`
- `nowcasting_work_ci/mlblack_side/*` should only keep scenario assembly, problem thin shell, and stage implementation

Runtime scaffold alignment:

- `runtime/config.py`: CLI/runtime config only
- `runtime/assembly.py`: registration helpers only (`reg_*`, `assemble_runtime_context`)
- `runtime/build_runtime.py`: total runtime assembly entry
- `runtime/actions/*`: business stage implementations (`parse_cli`, `build_runtime`, `outer_search`, `evaluate_final`, `assemble_result`)
- `runtime/stages.py`: control-plane stage runner only; only wires stage sequence

Stage-action deepening rule:

- if one stage action becomes too large, it may fan out into stage-local helper modules under `runtime/actions/`
- current example:
  - `outer_search_stage.py`: stage-facing shell
  - `outer_search_problem.py`: problem/solver epoch execution
  - `outer_search_tracking.py`: best-solution tracking and epoch logging
  - `outer_search_dynamic_pool.py`: dynamic expansion/prune path
