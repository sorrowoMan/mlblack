# 00. 统一装配 API 参考

`mlblack` follows the same Project/Case assembly substrate.

## 1）唯一规范入口

| `.case kind` | 规范装配入口 | 规范运行入口 |
| --- | --- | --- |
| `solver` | `build_solver.py:build_solver()` | `run_solver.py` |
| `trainer` | `build_solver.py:build_solver()` | `run_solver.py` |

`kind` 只表达语义与结果投影，不改变入口解析。`build_trainer.py` 必须是
`build_solver` 的薄别名，`run_trainer.py` 必须只转发到 `run_solver.main`。

## 2) Pipeline contract

- Keep one case-level pipeline entry (`pipeline/main.py` or legacy `pipeline.py`).
- For `mlblack`, the entry assembles data/model semantics (DataView transforms, codec/head wiring, etc.).
- Fine-grained logic lives in operator modules and is assembled by the pipeline entry.

## 3) Standard signatures

```python
def build_solver(config=None, *, resource_context=None, component_overrides=None): ...
def build_solver(config=None, *, resource_context=None, component_overrides=None): ...
```

Trainer 别名不得维护第二份签名或装配逻辑。

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
