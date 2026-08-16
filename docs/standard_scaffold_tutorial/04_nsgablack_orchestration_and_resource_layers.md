# 04. Shared Orchestration And Resource Layers

This file name is kept for compatibility, but the architecture statement is unified:

- Orchestration belongs to shared Project substrate.
- Resource grants belong to shared Project L0 substrate.
- `nsgablack` and `mlblack` are semantic layers on the same substrate.

## 1) Ownership

| Concern | Owner |
| --- | --- |
| stage/group/fanout scheduling | shared Project substrate |
| global resource authorization | shared Project L0 |
| search semantics | `nsgablack` |
| ML semantics | `mlblack` |

## 2) Entry Resolution

By `.case kind`:

- `solver` -> `build_solver.py` / `run_solver.py`
- `trainer` -> `build_trainer.py` / `run_trainer.py`

No dual primary entry.

## 3) Resource Context Rule

- Project grants `ResourceContext`.
- Case consumes and audits it.
- Nested child cases consume parent-derived grants.
- Case should not privately allocate global resources.
