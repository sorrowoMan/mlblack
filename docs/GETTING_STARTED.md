# GETTING_STARTED

## 1) Unified CLI

`mlblack` now provides a single command entry:

```powershell
python -m mlblack --help
```

Main command groups:

- `catalog`: pass-through to catalog registry tooling
- `doctor`: pass-through to project doctor checks
- `scaffold`: scaffold init/run commands

Examples:

```powershell
python -m mlblack catalog summary --profile framework-core
python -m mlblack doctor --path . --strict --format problem
```

## 2) Scaffold Workflow (Recommended Baseline)

Initialize a new scaffold project:

```powershell
python -m mlblack scaffold init --path C:\path\to\my_mlblack_project
```

Run the generated project:

```powershell
cd C:\path\to\my_mlblack_project
python run_train.py --config configs\train_config.json
```

For grouped `symbolic_stagewise`, scaffold also generates:

```powershell
python run_train.py --config configs\train_config_stagewise_grouped.json
```

Or run directly from repo with one config:

```powershell
python -m mlblack scaffold run --config examples\configs\work_ci_xgboost_portable.json
```

## 3) Work-CI Workflow (Your Current Main Scenario)

Fixed-parameter full-fold + rolling:

```powershell
python examples\run_work_ci_symbolic_fixed_cv_rolling.py ^
  --config examples\configs\work_ci_symbolic_torch_interval_no_flow_speed_occ_lag.json
```

Outer NSGABLACK + inner interval trainer:

```powershell
python examples\run_interval_pareto_nsgablack.py ^
  --config examples\configs\work_ci_symbolic_torch_interval_no_flow_speed_occ_lag.json
```

## 4) Optional Env Variables

For portable local runs, these vars are supported by example runners:

- `MLBLACK_WORK_CI_CSV`
- `MLBLACK_WORK_CI_CSV_NO_FLOW_SPEED_OCC`
- `MLBLACK_WORK_CI_CSV_NO_FLOW_SPEED_OCC_LAG`
- `MLBLACK_OUTPUTS_DIR`
- `MLBLACK_REPORTS_DIR`
- `NSGABLACK_ROOT`

Config fields such as `data.csv_path` and `train.output_dir` support env references:

- Windows style: `%VAR_NAME%`
- Braced style: `${VAR_NAME}`

## 5) Optional Flow Capabilities

`run_train_flow` and `run_semantic_train_flow` support lifecycle capabilities.

- Use `TrainFlowSpec.capabilities` to attach capability instances
- Use `FlowAssemblySpec.capabilities` for declarative config path (`JSON -> registry -> instantiate`)
- In scaffold config, declare `train.capabilities` as:
  - `[{ "key": "noop", "params": { "name": "trace_cap" } }]`
- Built-in `metric_guard` example:
  - `[{ "key": "metric_guard", "params": { "rules": [{ "split": "test", "metric": "rmse", "op": "le", "threshold": 1.5 }], "hard_fail": true } }]`
- Use `TrainFlowSpec.capability_strict=True` for fail-fast mode

Reference: `docs/ARCHITECTURE_PURPOSE.md`

## 6) Context + Snapshot Runtime State

Flow now supports nsgablack-style runtime state:

- `ContextStore`: lightweight keys and `*_ref` references
- `SnapshotStore`: heavy runtime payload snapshots

You can inject your own stores through flow spec:

- `TrainFlowSpec(context_store=..., snapshot_store=...)`
- `SemanticTrainFlowSpec(context_store=..., snapshot_store=...)`

The flow report includes:

- `report["state"]["context_refs"]`
- `report["state"]["snapshot_count"]`
- `report["state"]["snapshots"]`

Backends:

- `memory` (default)
- `sqlite` (single-file persistent state)

Scaffold JSON example:

```json
{
  "train": {
    "state_backend": {
      "context": {
        "backend": "sqlite",
        "db_path": "runs/runtime_state.sqlite3",
        "namespace": "ctx"
      },
      "snapshot": {
        "backend": "sqlite",
        "db_path": "runs/runtime_state.sqlite3",
        "namespace": "snap"
      }
    }
  }
}
```

## 7) ModelSpec Subspace Selection

You can declare model-specific feature/target subspace without changing trainer internals.

- Runtime API: `SemanticTrainFlowSpec(model_spec=ModelSpec(...))`
- Scaffold JSON: `train.model_spec`

Example:

```json
{
  "train": {
    "model_spec": {
      "model_id": "m_x1x3_to_y1",
      "feature_names": ["x1", "x3"],
      "target_names": ["y1"],
      "strict": true
    }
  }
}
```

State refs written to context:

- `model.spec_ref`
- `model.processed_ref`

### 7.1 Portfolio Parallel + GPU Assignment

`run_semantic_portfolio_flow` now supports portfolio-level runtime scheduling via `SemanticTrainFlowSpec`.

Preferred new entry:

- `execution.backend`
- `execution.max_workers`
- `execution.fail_fast`
- `execution.gpu_strategy`
- `execution.gpu_devices`
- `execution.default_device`

Legacy compatibility fields still work:

- `portfolio_parallel_mode`: `serial | thread | process`
- `portfolio_max_workers`: max worker count in parallel mode
- `portfolio_fail_fast`: stop on first failed sub-run
- `portfolio_gpu_strategy`: `none | fixed | round_robin | auto`
- `portfolio_gpu_devices`: explicit device pool, e.g. `[0, 1]`, `["cuda:0", "cuda:1"]`

Example:

```python
spec = SemanticTrainFlowSpec(
    ...,
    portfolio_parallel_mode="thread",
    portfolio_max_workers=4,
    portfolio_fail_fast=False,
    portfolio_gpu_strategy="round_robin",
    portfolio_gpu_devices=("cuda:0", "cuda:1"),
)
```

Notes:

- Torch trainer `device` now accepts `cuda:<index>` (for example `cuda:0`)
- Scaffold projects now generate `schema/execution_schema.py`, which exposes the same formal `L0` enum/catalog for UI tooling.
- `auto` strategy keeps manually set trainer `device` if it is not `auto`
- Portfolio summary now includes runtime scheduling metadata and per-model assigned device

## 8) Multi-Source Data Compose

Scaffold data layer supports mixed sources and merge:

- `csv`
- `sqlite_sql`
- `sqlite_table`
- `sql` (requires `sqlalchemy` + DB driver)

Example:

```json
{
  "data": {
    "sources": [
      { "name": "main", "kind": "csv", "path": "data/main.csv" },
      {
        "name": "weather",
        "kind": "sqlite_sql",
        "db_path": "data/ext.sqlite3",
        "sql": "SELECT date, aqi, wind FROM weather_daily",
        "prefix": "wx_"
      }
    ],
    "merge_on": ["date"],
    "merge_how": "inner",
    "target_col": "target",
    "date_col": "date",
    "feature_recipe": "raw_all_numeric",
    "split_mode": "ratio",
    "test_ratio": 0.2,
    "random_seed": 42
  }
}
```

## 9) `symbolic_stagewise` Config Styles (`flat + grouped`)

`symbolic_stagewise` now officially supports two equivalent config styles under `train.trainer_params`:

- `flat`: legacy-compatible flat keys such as `search_max_added_terms`
- `grouped`: nested sections such as `search_core.max_added_terms`

Recommendation:

- keep existing flat configs as-is if they are already stable
- use grouped style for new configs, scaffold presets, and team-facing examples

### 9.1 Flat Example

```json
{
  "train": {
    "trainer_key": "symbolic_stagewise",
    "trainer_params": {
      "artifact_id": "surrogate_stagewise",
      "force_linear_base": "auto",
      "search_max_added_terms": 12,
      "search_topk_features": 8,
      "search_online_beam_enabled": true,
      "search_online_beam_width": 6,
      "search_inner_opt_enabled": true,
      "search_inner_opt_method": "adam"
    }
  }
}
```

### 9.2 Grouped Example

```json
{
  "train": {
    "trainer_key": "symbolic_stagewise",
    "trainer_params": {
      "artifact_id": "surrogate_stagewise",
      "strategy": {
        "force_linear_base": "auto",
        "keep_search_trace": true
      },
      "search_core": {
        "max_added_terms": 12,
        "topk_features": 8,
        "max_expr_depth": 6
      },
      "search_family": {
        "unary_ops": ["square", "sin", "cos", "tanh"],
        "auto_nested_allowed_ops": ["square", "sin", "cos", "tanh"]
      },
      "search_online_beam": {
        "enabled": true,
        "width": 6,
        "bundle_size": 3
      },
      "search_inner_opt": {
        "enabled": true,
        "method": "adam",
        "adam_steps": 60
      }
    }
  }
}
```

### 9.3 Mixed Example and Precedence Rule

Mixed style is also valid inside the same `trainer_params` object:

```json
{
  "train": {
    "trainer_key": "symbolic_stagewise",
    "trainer_params": {
      "strategy": {
        "force_linear_base": "off"
      },
      "force_linear_base": "on"
    }
  }
}
```

Normalization rule:

- grouped sections are expanded first
- then top-level flat keys are applied
- therefore flat keys override grouped keys when both target the same field

### 9.4 Preset Merge Rule in Scaffold Projects

In scaffold projects, `config.py` preset `trainer_params` and runtime JSON `train.trainer_params` are merged one level deep.

That means:

- flat override is ideal for changing one or two fields
- grouped override is supported
- if you override a grouped section such as `search_core`, provide the full section you want to keep, because the merge is not recursive inside that section

### 9.5 Supported Group Names

Current grouped section names for `symbolic_stagewise`:

- `strategy`
- `auto_mode`
- `search_core`
- `search_overfit`
- `search_gradient`
- `search_family`
- `search_prune`
- `search_path_memory`
- `search_graph_cache`
- `search_online_beam`
- `search_joint_bundle`
- `search_inner_opt`
- `artifact_runtime`

## 10) High-Order Nested Function Patterns (Stagewise Symbolic)

`symbolic_stagewise` supports generic multi-layer unary nesting and now defaults to auto nested discovery.

Examples:

- `sin(square)` (legacy format, still supported)
- `sin(square(tanh))`
- `exp(log(abs(x)))`

Notes:

- Unary ops must be valid DSL unary ops: `identity,square,sin,cos,tanh,exp,log,abs,sqrt`
- Effective nesting is still bounded by `search_max_expr_depth`
- To allow deeper structures, increase `search_max_expr_depth` (and optionally `search_max_added_terms`)
- Nested mode:
  - `search_nested_mode="auto"` (default): auto-generate nested chains from `search_auto_nested_allowed_ops`
  - `search_nested_mode="manual"`: only use `search_nested_unary_patterns`
  - `search_nested_mode="hybrid"`: use both
- Auto controls:
  - `search_auto_nested_allowed_ops`
  - `search_auto_nested_min_depth`
  - `search_auto_nested_max_depth`
  - `search_auto_nested_beam_width`
  - `search_auto_nested_max_patterns_per_feature`

### 10.1 Gradient-Projection With Nested Focus

Gradient residual projection now supports both:

- legacy form: `x_i * phi(x_j, ...)`
- enhanced form: `g(x_i) * phi(x_j, ...)` where `g` can be unary/nested

Key controls:

- `search_grad_projection_focus_include_transforms` (default `true`)
- `search_grad_projection_focus_topk_transforms` (default `2`)

## 11) strict4 Branch Parallel + GPU Round-Robin

For fixed-holiday rolling eval, strict4 branch training now supports:

- `--strict4-parallel-mode serial|thread|process`
- `--strict4-max-workers`
- `--strict4-gpu-strategy none|fixed|round_robin|auto`
- `--strict4-gpu-devices` (for example `0,1` or `cuda:0,cuda:1`)

Recommended templates:

- Branch hparams: `examples/configs/strict4_branch_hparams_parallel_gpu_rr.json`
- Compare template: `examples/configs/work_ci_strict4_parallel_gpu_compare_template.json`

One-command compare:

```powershell
python examples\run_work_ci_strict4_parallel_gpu_compare.py ^
  --config examples\configs\work_ci_strict4_parallel_gpu_compare_template.json
```
