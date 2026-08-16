# benchmarks

这是保留的 benchmark 兼容 Case。

当前标准：

- `build_solver.py` 是 canonical assembly entry。
- `run_solver.py` 如存在，是 Case CLI/debug entry。
- `build_trainer.py` 如存在，只能 alias 到 `build_solver.py`。
- `problem/`、`pipeline/`、`adapter/`、`bias/`、`plugins/`、`evaluation/`、`runtime/`、`solver/` 是标准 Case 目录。
- 如需模型状态编码，放在 `pipeline/` 内；不要新增 case-level `assembly/` 或 `representation/` 入口。

多阶段顺序、fanout、artifact handoff 和资源 grant 属于共享 Project / Case / L0 substrate。需要搜索 benchmark 选择时，使用优化搜索 Case。
