# orthogonal_source_baseline

Standard scaffold for testing the model-agnostic Orthogonal Source Layer.

## Purpose

This project validates the architecture:

```text
Raw features
  -> Orthogonal Source Layer
  -> strong downstream learner
```

The source layer is not a symbolic head and does not own prediction semantics.
It produces `basis_matrix + source_metadata + stability/overlap report`.
Downstream learners then consume the orthogonal source objects as ordinary
features.

The reported feature spaces are intentionally separate:

- `raw_features`: downstream learner receives only original columns.
- `orthogonal_sources`: downstream learner receives only selected source objects.
- `raw_plus_orthogonal_sources`: downstream learner receives original columns plus source objects.

This separates source compression from source augmentation. Open tabular data can
lose information when raw features are replaced by a small source basis; that is
a different claim from whether source objects help as an auditable feature layer.

Downstream baselines currently include fixed-structure linear, tree, boosting,
and neural MLP regressors. The neural baseline is still a fixed mlblack-side
trainer; it does not perform symbolic structure search.

The `neural_mlp` baseline is assembled through the formal mlblack neural trainer
protocol:

- `trainer_key=sklearn_mlp`
- `pipeline_key=zscore`
- target z-score scaling in the benchmark adapter, with inverse transform before metrics
- `neural_training_report.*` for early-stopping and convergence audit
- `neural_training_curve.*` for loss and validation-score curves

## Responsibility Split

- `problem/`: known-relation datasets reused as controlled benchmark scenarios.
- `pipeline/`: orthogonal source construction and strong baseline fitting.
- `config/`: reproducible suite configuration.
- `orchestration/`: scenario/suite runner.
- `reporting/`: markdown/csv/json output writers.
- `build_solver.py`: standard assembly surface.
- `run_solver.py`: thin scaffold entrypoint.

Reusable framework component:

- `core/orthogonal_source/layer.py`

## Run

```powershell
python my_project\orthogonal_source_baseline\run_solver.py --check
python my_project\orthogonal_source_baseline\run_solver.py --suite-id demo_v1
```

## Interpretation

The comparison is not "symbolic vs XGBoost". It tests whether source governance
helps fixed downstream models by reducing proxy/redundancy pressure and exposing
mechanistic source objects before prediction.

Open-tabular scenario keys:

- `open:california_housing`: sklearn real regression, low-dimensional continuous features.
- `open:diamonds_price`: OpenML diamonds real regression, target `price`, mixed categorical/continuous features.
- `open:covtype_numeric`: large pressure test only; Covertype is originally classification and is treated as a numeric target.
