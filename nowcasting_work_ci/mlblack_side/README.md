# mlblack_side

This side is now organized around scenario assembly only:

- `problem/`: scenario problem shell only; reusable bridge/proxy contracts now live in top-level `problem/`
- `runtime/`: scenario scaffold only
- `config/`: shared runtime dataclass config

`runtime/` is now split in the nsgablack style:

- `config.py`: runtime CLI/config source of truth
- `assembly.py`: `reg_*` registration helpers only
- `build_runtime.py`: total runtime assembly entry
- `actions/`: business stage implementation modules
- `stages.py`: stage runner / control plane wiring only
- `workflow.py`: orchestrator entry

When a single stage action is still too large, it can split again into stage-local helpers.
Current example:

- `outer_search_stage.py`: stage-facing wrapper
- `outer_search_problem.py`: epoch-level problem/solver execution
- `outer_search_tracking.py`: best-row / decode-meta / epoch-log tracking
- `outer_search_dynamic_pool.py`: dynamic candidate-pool expansion and pruning

Framework-root counterparts now live outside the scenario folder:

- top-level `pipeline/`: reusable feature-space facade and feature-space builder
- top-level `model/`: model fitting/inference kernels (`interval_fit`, `config.py`)
- top-level `evaluation/`: evaluation callbacks and runtime config
- top-level `plugins/`: report writer and runtime hooks
- top-level `bias/`: branch/objective/dynamic-pool policy

Compatibility entry:

- `runtime.py` forwards to `runtime/workflow.py`.
