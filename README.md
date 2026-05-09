# mlblack

For AI / coding agents: read `AGENTS.md` first.

`mlblack` is a surrogate-model assembly framework designed to plug into optimization pipelines.

Core purpose: **compose function families and capabilities without locking into one model form**.

Related project: [`nsgablack`](https://github.com/sorrowoMan/nsgablack) owns outer optimization orchestration. `mlblack` stays focused on inner evaluation proxies, fitting, artifacts, and audit/report surfaces.

## What It Provides
- Semantic-to-numeric data path: `schema -> numericizer -> pipeline -> trainer -> artifact`
- Pluggable trainer registry (`ridge`, `xgboost`, `mlp_torch`, `symbolic_*`)
- Symbolic stagewise auto-nesting (`manual|auto|hybrid`) with depth/beam/budget controls
- Declarative flow capabilities (`JSON -> capability registry -> instantiated lifecycle stack`)
- Experiment tracker capability (`experiment_tracker`) for run/event/metric sqlite logging
- Runtime state plane (`ContextStore + SnapshotStore`) with `*_ref`-based flow tracing
- Pluggable state backends (`memory` / `sqlite`) for runtime context & snapshots
- Explicit `ModelSpec` subspace training (choose feature/target subset per model)
- Multi-source data compose (`csv/sqlite/sql`) with merge lineage in flow state
- Unified training workflow with report/artifact output
- Project scaffold for three-entry style setup (`config.py`, `assembly.py`, `run_train.py`)
- `symbolic_stagewise` config supports both flat and grouped `trainer_params` styles

## Quick Start
1. Check unified CLI:
```powershell
python -m mlblack --help
```
2. Initialize a scaffold project:
```powershell
python -m mlblack scaffold init --path C:\path\to\my_mlblack_project
```
3. Fill `configs/train_config.json` in the generated project.
   For grouped `symbolic_stagewise`, scaffold also generates `configs/train_config_stagewise_grouped.json`.
4. Run **inside the generated project directory**:
```powershell
python run_train.py --config configs\train_config.json
```

## Config Style Note
- `train.trainer_params` for `trainer_key="symbolic_stagewise"` now officially supports two equivalent styles:
  - flat: `search_max_added_terms`, `search_topk_features`, `search_inner_opt_enabled`
  - grouped: `search_core.max_added_terms`, `search_core.topk_features`, `search_inner_opt.enabled`
- Flat syntax remains fully compatible.
- Grouped syntax is recommended for readability and scaffold projects.
- If flat and grouped keys appear in the same `trainer_params` object, flat keys win.
- Full examples: [docs/GETTING_STARTED.md](/C:/Users/hp/Desktop/mlblack/docs/GETTING_STARTED.md)

## Development
- Current capability snapshot: `docs/CURRENT_STATE.md`
- Getting-started workflows: `docs/GETTING_STARTED.md`
- Architecture purpose and lifecycle intent: `docs/ARCHITECTURE_PURPOSE.md`
- AI collaboration entry rules: `AGENTS.md`
- AI development guidelines: `docs/AI_DEVELOPMENT_GUIDELINES.md`
- Experiment dashboard v0: `docs/EXPERIMENT_DASHBOARD_V0.md`
- Public/deprecated surface contract: `docs/PUBLIC_SURFACE.md`
- Catalog UI explorer: `docs/CATALOG_UI.md`
- Catalog DB protocol: `docs/CATALOG_DB_PROTOCOL.md`
- Framework notes: `docs/mlblack_framework_logic.md`

## Catalog
```powershell
python -m mlblack catalog summary --profile framework-core
python -m mlblack catalog list --profile default --kind trainer
python -m mlblack catalog list --profile framework-core --kind preset --field family=neural
python -m mlblack catalog list --profile framework-core --kind component --field component_surface=runtime_mechanism
python -m mlblack catalog list --profile framework-core --kind plugin --field plugin_surface=flow_plugin
python -m mlblack catalog schema --profile framework-core --kind preset
python -m mlblack catalog snapshot --profile framework-core --kind preset --field family=neural
python -m mlblack catalog search symbolic --profile framework-core --limit 20
```

Persist the structured catalog into a SQL catalog store:

```powershell
python -m mlblack catalog db materialize --db-path runs\catalog.sqlite3 --profile framework-core
python -m mlblack catalog db materialize --db-path postgresql://postgres:YOUR_PASSWORD@127.0.0.1:5432/mlblack_catalog --profile framework-core
python -m mlblack catalog db materialize --db-path mysql://root:YOUR_PASSWORD@127.0.0.1:33306/mlblack_catalog --profile framework-core
python -m mlblack catalog db target
python -m mlblack catalog db summary --db-path runs\catalog.sqlite3 --profile framework-core
python -m mlblack catalog db show preset:mlp_torch --db-path runs\catalog.sqlite3 --profile framework-core
python -m mlblack catalog db list --db-path runs\catalog.sqlite3 --profile framework-core --kind preset --field family=neural --format json
python -m mlblack catalog db search gradient_norm --db-path runs\catalog.sqlite3 --profile framework-core --kind preset --format json
python -m mlblack catalog db facets --db-path runs\catalog.sqlite3 --profile framework-core --kind preset --field family=neural --facet-field runtime_backend
python -m mlblack catalog db snapshot --db-path runs\catalog.sqlite3 --profile framework-core --kind preset --field family=neural --selected preset:mlp_torch
python -m mlblack catalog source --profile framework-core
```

## Catalog UI
```powershell
python -m mlblack catalog ui --profile framework-core --kind preset
python -m mlblack catalog ui --profile framework-core --kind preset --db-path postgresql://postgres:YOUR_PASSWORD@127.0.0.1:5432/mlblack_catalog
python -m mlblack catalog ui --profile framework-core --kind preset --db-path mysql://root:YOUR_PASSWORD@127.0.0.1:33306/mlblack_catalog
```

Or route the page through `catalog/db.toml` / env mode:

```powershell
$env:MLBLACK_CATALOG_DB_URL = "postgresql://postgres:YOUR_PASSWORD@127.0.0.1:5432/mlblack_catalog"
$env:MLBLACK_CATALOG_DB_MODE = "only"
python -m mlblack catalog ui --profile framework-core --kind preset
```

- Structured standalone page layout:
  - left rail: field filters
  - center: result list / table with row selection
  - right rail: detail + relation jump
  - top switch: `family / preset / head / component / provider / plugin`
- Compatibility wrapper remains available:
```powershell
streamlit run examples/run_catalog_dashboard.py -- --profile framework-core --kind preset
```

## Experiment UI
```powershell
python -m mlblack experiment summary --db runs\experiments.sqlite3
python -m mlblack experiment list-runs --db runs\experiments.sqlite3 --has-fold-summary
python -m mlblack experiment list-artifacts --db runs\experiments.sqlite3 --head-task interval
python -m mlblack experiment ui --db runs\experiments.sqlite3
```

- Deprecated compatibility wrapper remains available:
```powershell
streamlit run examples/run_experiment_dashboard.py -- --db runs/experiments.sqlite3
```

## Doctor
```powershell
python -m mlblack doctor --path . --strict --format problem
```
- Optional external rule plugins:
```powershell
python -m mlblack doctor --path . --rules-dir .\my_doctor_rules --only-rule custom_rule_id
```

## Checkpoint / Replay
- Enable checkpoint in flow spec:
  - `TrainFlowSpec(save_checkpoint=True, checkpoint_dir="runs/my_checkpoint")`
- Replay from checkpoint:
  - `TrainFlowSpec(replay_from_checkpoint="runs/my_checkpoint")`
- Torch epoch-level resume:
  - `TorchMLPTrainerConfig(checkpoint_dir="runs/torch_ckpt", checkpoint_every_epochs=5)`
  - `TorchMLPTrainerConfig(resume_training_from="runs/torch_ckpt/latest.pt", ...)`
