# START_HERE

## 1. Health Baseline

```powershell
python -m mlblack project doctor --path . --strict
```

## 2. Cross-Framework Boundary

- Standard Cases communicate through payload/result contracts.
- `ResourceContext` comes from the shared Project L0 substrate.
- Large objects pass by Artifact/Snapshot refs.
- `nsgablack` contributes optimization/search semantics when needed.
- `mlblack` contributes ML semantics and artifacts.

Assembly lives in `build_solver.py`. Do not add case-level `assembly/` or `representation/` entrypoints.

## 3. Assembly

`build_solver.py` is canonical. `build_trainer.py`, when present, is an alias.

## 4. Run

```powershell
python run_solver.py
python -m compileall -q .
```
