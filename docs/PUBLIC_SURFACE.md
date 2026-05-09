# PUBLIC_SURFACE

This document fixes which framework surfaces are formal, which ones are deprecated,
and what migration path new callers should follow.

## 1. Formal framework entrypoints

Preferred long-term surfaces:

- `python -m mlblack catalog ...`
- `python -m mlblack doctor ...`
- `python -m mlblack scaffold ...`
- `python -m mlblack experiment ...`

For symbolic training presets, the formal preset entry is:

- `trainer_key="symbolic"`

New symbolic callers should express variation through family fields:

- `parameter_backend`
- `task`
- `structure_engine`
- future `search_policy`

## 2. Deprecated compatibility surfaces

The following surfaces remain for compatibility only and should not be used for new work:

- `trainer_key="symbolic_stagewise"`
- `trainer_key="symbolic_torch"`
- `trainer_key="symbolic_torch_interval"`
- `streamlit run examples/run_experiment_dashboard.py -- --db ...`

These continue to exist as legacy facades or shims so old flows do not break immediately.

## 3. Migration targets

Deprecated symbolic presets map to the same formal target:

- `symbolic_stagewise` -> `symbolic`
- `symbolic_torch` -> `symbolic`
- `symbolic_torch_interval` -> `symbolic`

Deprecated experiment dashboard wrapper maps to:

- `examples/run_experiment_dashboard.py` -> `python -m mlblack experiment ui --db ...`

## 4. Contract rule

When adding or cleaning framework surfaces:

1. formal surfaces must have stable CLI/module contracts
2. deprecated surfaces must point to a single explicit migration target
3. catalog should expose whether a surface is `formal` or `deprecated`
4. compatibility shims should be thin wrappers, not the main implementation home
