# mlblack Example Projects

`examples/cases/` is the formal example namespace. Each direct child is a Project wrapper, not a loose script folder.

```text
examples/cases/<project>/
  project_config.py
  run_project.py
  README.md
  START_HERE.md
  cases/
    <case>/
      build_solver.py
      run_solver.py
      build_trainer.py    # alias only, when present
      run_trainer.py      # alias only, when present
      problem/
      pipeline/
      plugins/
      runtime/
```

Formal runs start at the Project root:

```powershell
python examples/cases/<project>/run_project.py --check
python examples/cases/<project>/run_project.py --check --build-check
```

Case-local `run_solver.py` is a debug entry only. Project orchestration, multi-case order, and L0 `ResourceContext` grants belong to `project_config.py` and `run_project.py`.

`mlblack` examples use the same substrate as `nsgablack`; the difference is semantic content, not runner ownership. `mlblack` cases may be outer or inner, and multi-trainer or nested projects should compose Cases through the Project runner rather than creating private schedulers.
