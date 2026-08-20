# 04. 共享编排与资源层

三仓共享同一套编排和资源口径：

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

无论 `.case kind` 是什么：

- `solver` -> `build_solver.py` / `run_solver.py`
- `trainer` -> `build_solver.py` / `run_solver.py`

Trainer 命名文件只能作为薄别名，不能形成第二主入口。

## 3) Resource Context Rule

- Project grants `ResourceContext`.
- Case consumes and audits it.
- Nested child cases consume parent-derived grants.
- Case should not privately allocate global resources.
