# cross_framework

这是保留的跨框架边界实验兼容 Case。

当前标准：

- 两边都应该暴露标准 Case surface。
- `build_solver.py` 是 canonical；`build_trainer.py` 如存在只能是 alias。
- request/result 通过 JSON-compatible payload、Artifact/Snapshot refs 和 `ResourceContext` 传递。
- 装配入口只放在 `build_solver.py`；不要新增 case-level `assembly/` 或 `representation/` 入口。

共享 Project / Case / L0 substrate 负责编排和资源 grant。`nsgablack` 在需要时提供优化搜索语义；`mlblack` 提供 ML 语义和 artifact。
