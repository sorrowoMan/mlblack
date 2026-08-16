# Blackbase substrate 演示 Case

该 Case 用一个小型线性回归任务演示统一框架栈的正式边界：

- `build_solver.py` 是唯一装配入口，返回 mlblack `Trainer`。
- `run_solver.py` 是唯一 CLI 入口；trainer 命名文件仅保留薄别名。
- `problem/` 负责回归目标，`pipeline/` 负责候选状态编解码，`adapter/` 负责搜索策略。
- Project 层发放 `ResourceContext`，Case 只消费 grant，不自行分配全局资源。

可先执行 `python run_solver.py --check` 检查装配，再执行 `python run_solver.py --steps 3` 跑通生命周期。
