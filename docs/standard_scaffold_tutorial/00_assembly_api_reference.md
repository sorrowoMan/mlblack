# 00. Assembly API Reference

`mlblack` follows the same Project/Case assembly substrate.

## 1) Case primary entry by kind

| `.case kind` | Primary assembly | Primary run |
| --- | --- | --- |
| `solver` | `build_solver.py:build_solver()` | `run_solver.py` |
| `trainer` | `build_trainer.py:build_trainer()` | `run_trainer.py` |

No dual primary entries.

## 2) Pipeline contract

- Keep one case-level pipeline entry (`pipeline/main.py` or legacy `pipeline.py`).
- For `mlblack`, the entry assembles data/model semantics (DataView transforms, codec/head wiring, etc.).
- Fine-grained logic lives in operator modules and is assembled by the pipeline entry.

## 3) Standard signatures

```python
def build_solver(config=None, *, resource_context=None, component_overrides=None): ...
def build_trainer(config=None, *, resource_context=None, component_overrides=None): ...
```

The same signatures are expected in `nsgablack` tutorial and templates.

## 4) Slot Kernel 桥接

两框架都采用“一个 pipeline 主入口 + slot kernel 编排”。

Shared mode family:

- `serial`
- `parallel`
- `router`

详见：

- `nsgablack/docs/standard_scaffold_tutorial/08_slot_kernel_minimal_spec.md`
- `mlblack/docs/standard_scaffold_tutorial/09_slot_kernel_minimal_spec.md`

## 5) Resource rule

- Project L0 grants `ResourceContext`.
- Cases consume and audit grants.
- Cases do not allocate global resources.
