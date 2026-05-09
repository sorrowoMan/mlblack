# my_project (MLBLACK Scaffold Template)

A standard MLBLACK scaffold template aligned with modular assembly.

## Layout

- `config/` unified config dataclasses + config loader
- `problem/` problem definition and contracts
- `features/` feature engineering assembly
- `model/` model training/inference assembly
- `reporting/` summary/report output assembly
- `runtime/` runtime parser + workflow orchestration
- `build_runtime.py` assembly factory
- `run_runtime.py` CLI entry

## Quick check

```powershell
python C:\Users\hp\Desktop\mlblack\my_project\run_runtime.py --check
```

## Example run

```powershell
python C:\Users\hp\Desktop\mlblack\my_project\run_runtime.py --run-id demo
```

## Stagewise Config Example

For `symbolic_stagewise`, `train.trainer_params` supports both `flat` and `grouped` styles.

Grouped style example:

```json
{
  "train": {
    "trainer_key": "symbolic_stagewise",
    "trainer_params": {
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
        "unary_ops": ["square", "sin", "cos", "tanh"]
      },
      "search_online_beam": {
        "enabled": true,
        "width": 6
      }
    }
  }
}
```

Rule:

- flat style remains compatible
- grouped style is recommended for new scaffold configs
- if flat and grouped write the same field together, flat wins
