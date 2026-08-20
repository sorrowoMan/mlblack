# cross_framework

这是正式的跨框架父 Case。

当前标准：

- 两边都应该暴露标准 Case surface。
- `build_solver.py` 是 canonical；`build_trainer.py` 如存在只能是 alias。
- `NestedTrainerProblem.evaluate()` 通过 `CaseRunRequest` 调用完整内层 Trainer Case。
- request/result 通过版本化 payload、Artifact refs 和派生 `ResourceContext` 传递。
- 装配入口只放在 `build_solver.py`；不要新增 case-level `assembly/` 或 `representation/` 入口。

外层 `GaussianSearchAdapter` 只搜索学习率；内层 `LearningSolver` 负责数据、
模型、loss 与训练结果。共享 Project / Case / L0 substrate 负责编排和资源 grant。
