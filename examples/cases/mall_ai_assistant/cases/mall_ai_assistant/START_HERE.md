# START_HERE

## 1. Health Baseline

```powershell
python -m mlblack project doctor --path . --strict
```

## 2. Core Layers

- `problem/`: evaluate model and produce feedback
- `pipeline/`: prepare data and hold model-state encode/decode helpers when needed
- `adapter/`: propose/update or fitting strategy
- `plugins/`: audit, tracking, checkpoint, report side effects

Assembly lives in `build_solver.py`. Do not add case-level `assembly/` or `representation/` entrypoints.

## 3. Assembly

`build_solver.py` is canonical. `build_trainer.py`, when present, is an alias.

## 4. Run

```powershell
python run_solver.py
python -m compileall -q .
```
