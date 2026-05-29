# orthogonal_point_demo

mlblack scaffold (inner-trainer project layout).

## Quickstart
1. python -m mlblack project doctor --path .
2. python run_solver.py
3. python build_solver.py

## Structure
- build_solver.py: canonical assembly entry
- problem/, pipeline/, pipeline/representation/
- adapter/, bias/, plugins/
- assembly/, catalog/entries.toml

## Notes
- Multi-stage/group/event orchestration belongs to nsgablack.
- This scaffold provides a single inner trainer.
- Use project doctor to validate contracts early.
