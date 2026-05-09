# mlblack Standard Scaffold

This module supports two scaffold actions:

1. Initialize a standard project skeleton.
2. Run surrogate training from one scaffold config.

## 1) Initialize Project Skeleton

```powershell
python examples\init_project_scaffold.py --path C:\Users\hp\Desktop\my_mlblack_project
```

Generated layout:
- `schema/`, `numericizer/`, `pipeline/`, `bias/`
- `trainer/`, `workflow/`, `catalog/`
- `data/`, `configs/`, `runs/`, `docs/`, `scripts/`
- `config.py`, `assembly.py`, `run_train.py`, `START_HERE.md`
- `configs/train_config.json`
- `configs/train_config_stagewise_grouped.json`

Three-file entry split (nsgablack-style):
- `config.py`: registration defaults (presets / policy)
- `assembly.py`: payload resolve + scaffold spec assembly
- `run_train.py`: CLI execution only

## 2) Run Scaffold Training

```powershell
python run_train.py --config configs\train_config.json
```

or from repo examples:

```powershell
python examples\run_project_scaffold.py --config examples\configs\work_ci_xgboost.json
```

## Semantic Decoupling (nsgablack-style)

Assembly chain:
- `data reader` -> `schema/view` -> `numericizer` -> `pipeline` -> `bias` -> `trainer` -> `artifact`

Boundary rule in semantic flow:
- numericization options must be in `assembly.numericizer.params`
- they must not be hidden inside `assembly.trainer.trainer_params`
- enforced by `validate_flow_assembly(...)`

## Config Sections

- `data`: where to read table data and how to split train/test.
- `train`: preset key + runtime overrides.

### `train.trainer_params` Style

For `trainer_key="symbolic_stagewise"`, `train.trainer_params` officially supports:

- flat style:
  - `force_linear_base`
  - `search_max_added_terms`
  - `search_inner_opt_enabled`
- grouped style:
  - `strategy.force_linear_base`
  - `search_core.max_added_terms`
  - `search_inner_opt.enabled`

Rule:

- flat and grouped can coexist in the same object
- flat keys override grouped keys when both point to the same final config field
- grouped style is recommended for scaffold projects because it reads closer to module boundaries

Minimal grouped example:

```json
{
  "train": {
    "preset_key": "baseline_stagewise",
    "trainer_params": {
      "strategy": {
        "force_linear_base": "auto",
        "keep_search_trace": true
      },
      "search_core": {
        "max_added_terms": 12,
        "topk_features": 8
      },
      "search_inner_opt": {
        "enabled": true,
        "method": "adam"
      }
    }
  }
}
```

Core implementation:
- `project/scaffold.py`
- `config/assembly.py`
- `core/workflow.py`

## Trainer 能力描述出口

如果脚手架、UI 或外部配置面板需要读取 trainer 的训练能力，不要手写 if/else 猜测：

- 用 `config.describe_trainers()[trainer_key]["contract"]`
- 或用 `config.describe_registered()["trainers"][*]["metadata"]["trainer_contract"]`

这两个出口会给出同一套稳定结构，重点包括：

- `training_modes.fresh/resume/warm_start/incremental/recalibrate`
- `trainer_state.enabled/save_load`
- `supports`
- `artifacts`
- `runtime`
