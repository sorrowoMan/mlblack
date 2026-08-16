# Cases

Each subdirectory is an independent solver/trainer Case: a complete standard
scaffold.

- Add a new case: `python -m nsgablack project add-case <name> --type solver`
- Run all cases: `python run_project.py` from the Project root.
- Validate Project L0 and assembly: `python run_project.py --check --build-check`.
- Debug one Case only: `cd cases/<name> && python run_solver.py --check`.
- Pipeline rule: keep one pipeline entry (`pipeline/main.py`), and place
  fine-grained operators under `pipeline/operators/*`.

Case-local `run_solver.py` is not the formal orchestration entry. Stage order,
multi-Case fanout, and resource grants belong to the Project root.
