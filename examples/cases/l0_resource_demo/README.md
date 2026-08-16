# Project Scaffold

Multi-case project. Each subdirectory under `cases/` is one independent solver/trainer case.

## Structure

```text
project_config.py
run_project.py
cases/
  <case_name>/
    build_solver.py        # canonical assembly entry
    run_solver.py          # canonical CLI entry
    build_trainer.py       # compatibility alias
    run_trainer.py         # compatibility alias
    config.py
    problem/
    pipeline/
      main.py
      operators/
    adapter/
    bias/
    plugins/
```

## Key rules

- Every case uses `build_solver.py` / `run_solver.py` as canonical entries;
  `.case kind` only describes semantics.
- One case has one pipeline primary entry (`pipeline/main.py` recommended).
- Fine-grained pipeline logic goes to `pipeline/operators/*`.
- Formal orchestration starts at `run_project.py`.
