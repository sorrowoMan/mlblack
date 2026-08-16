# mall_ai_assistant

这是保留的 assistant-style 实验兼容 Case。

当前标准：

- `build_solver.py` 是 canonical assembly entry。
- `run_solver.py` 如存在，是 Case CLI/debug entry。
- `problem/`、`pipeline/`、`adapter/`、`bias/`、`plugins/`、`evaluation/`、`runtime/`、`solver/` 是标准 Case 目录。
- 装配入口只放在 `build_solver.py`；不要新增 case-level `assembly/` 或 `representation/` 入口。

assistant workflow 编排属于共享 Project / Case / L0 substrate 或外部 operator layer，不应放进 `mlblack` 私有 runner。
