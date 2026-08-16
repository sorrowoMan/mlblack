# orthogonal_point_demo

这是一个小型 `mlblack` Case，用于展示 orthogonal linear point prediction。

当前标准：

- `build_solver.py` 是 canonical assembly entry。
- `run_solver.py` 如存在，是 Case CLI/debug entry。
- `build_trainer.py` 如存在，只能 thin alias 到 `build_solver.py`。
- `problem/`、`pipeline/`、`adapter/`、`bias/`、`plugins/`、`evaluation/`、`runtime/`、`solver/` 是标准 Case 目录。

装配入口只放在 `build_solver.py`；模型状态编码如需要应放在 `pipeline/` 内，不新增 case-level `assembly/` 或 `representation/`。
