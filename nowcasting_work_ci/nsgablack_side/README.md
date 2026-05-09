# nsgablack_side scaffold

Standard scaffold shape in this folder:

- `problem/config.py`
- `pipeline/config.py`
- `adapter/config.py`
- `evaluation/config.py`
- `plugins/config.py`
- `build_solver.py`
- `run_solver.py`

This side only hosts outer solver wiring concerns.
Model-specific runtime stays in `../mlblack_side/`.
