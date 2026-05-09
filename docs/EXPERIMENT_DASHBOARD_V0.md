# Experiment Dashboard v0

This v0 layer adds experiment-process visualization without changing trainer internals.

## 1) Enable tracker capability

In semantic/scaffold config:

```json
{
  "train": {
    "capabilities": [
      {
        "key": "experiment_tracker",
        "params": {
          "db_path": "runs/experiments.sqlite3",
          "namespace": "work_ci",
          "tag": "baseline_v1",
          "io_mode": "batched",
          "commit_interval": 12
        }
      }
    ]
  }
}
```

Key params:
- `db_path`: sqlite path for experiment events/metrics
- `namespace`: run group name
- `tag`: optional label for run grouping
- `io_mode`: sqlite write mode (`legacy|safe|batched`)
- `commit_interval`: batched mode commit frequency (`0` means flush on finish)

## 2) Run training flow

Use normal flow entry (`run_semantic_train_flow` or `run_project_scaffold`).

Tracker writes:
- `experiment_runs`
- `experiment_events`
- `experiment_metrics`
- `experiment_training_trace` (when trainer metadata contains `search_trace.iterations`)
- `experiment_run_catalog`
- `experiment_artifact_catalog`

Flow report will contain:
- `report["experiment_tracker"]["run_id"]`
- `report["experiment_tracker"]["db_path"]`

## 3) Open dashboard

Formal CLI entry:

```powershell
python -m mlblack experiment ui --db runs/experiments.sqlite3
```

Deprecated compatibility wrapper:

```powershell
streamlit run examples/run_experiment_dashboard.py -- --db runs/experiments.sqlite3
```

Dashboard panels:
- run catalog / artifact catalog dual view
- filters for:
  - `fold_summary`
  - `rmse_std`
  - `coverage_error_mean`
  - `status`
  - `trainer_name`
  - `head_task`
- selected run / artifact detail projection
- fold summary / stability / compatibility drift / signature contract detail
- lifecycle event timeline
- metrics table + curve
- training trace table:
  - parameter/readout change proxy (`weight_l2_before/after`)
  - gradient mismatch curve (`grad_overall_mismatch`)
  - function-family and operation path (`selected_family`, `operation`)
 - hard-fit diagnostics:
  - `delta_weight_l2`
  - `delta_grad_mismatch`
  - `hard_fit_index = |delta_weight_l2| / (|delta_grad_mismatch| + eps)`
 - replace attribution:
  - grouped by `selected_family`
  - `count`, `mean_gain`, `median_gain`, `win_rate`
 - operation transition quality:
 - grouped by `prev_operation -> operation`
 - `count`, `mean_gain`, `win_rate`
- function-cluster snapshot:
  - active terms replayed from `add/replace/prune`
  - cluster-size evolution over iterations
  - last-iteration readout top terms (`readout.after.top_terms`)

The dashboard is now `catalog-first`:

1. filter experiments from run/artifact catalog
2. select one run or artifact
3. drill into raw event / metric / trace diagnostics for that run

## 4) Scope

v0 focuses on observability of process sensitivity:
- stage transitions
- metric snapshots
- model-spec linked runs
- symbolic stagewise internal evolution when `search_trace` is present

This is intentionally decoupled from trainer logic and can be toggled by capability config.
