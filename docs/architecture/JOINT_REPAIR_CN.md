# ML Case 的共同修复接入

2026-10-09。优化控制仍由 nsgablack.ComposableSolver 提供，ML 侧只适配候选语义。

在标准 Case 的 build_solver.py 中，向 LearningSolver 或 build_learning_solver 传入：

```python
solver = build_learning_solver(
    problem=problem, representation=representation, adapter=adapter,
    problem_validator=validate_final_state,
    joint_repair_policy=JointRepairPolicy(max_rounds=1, on_rejection="stop"),
    resource_context=resource_context,
)
```

`JointRepairPolicy` 从 `nsgablack.representation` 导入。`validate_final_state(state, context) -> bool` 检查最终 UnknownState，包括 UnitRepair 改动后的数值和 metadata。context 沿用 ML bridge 的上下文注入。它是送评前的显式声明检查，不能隐式训练、评估模型或消耗评估预算。

既有 ModelRepresentation.repair 仍处理修复；bridge.problem_repair 仅在显式提供验证函数时可用。不存在默认恒真 validator，也不根据模型中碰巧同名的方法推测可行性。未开启共同修复的 Case 保持既有行为。

UnitRepair 绑定实际叶 Adapter。默认仅 ProblemRepair 的路径与共同修复路径均捕获修复前局部参与者状态；成功送评后零接纳恢复该状态，再通知普通 disposition，包含 GradientOptimizer 的参数与相关对账改动，即使兄弟策略提交也不能留下被拒绝的修复。已执行评估的成本和证据仍保留。

部分接纳且绑定 UnitRepair 时要求组件实现 `project_admission(baseline, draft, evidence, disposition, context) -> Unit`，在 Adapter disposition **之后**执行。baseline 是完整修复前 Unit；draft 已整理为接纳坐标，evidence/disposition 仍用原提案索引，禁止二次筛选 draft 或改写 admission_unit_fields。缺少投影方法明确失败。ML 层不复制这套接纳或事务实现。

on_rejection 默认 stop；retry 显式允许重试；advance 只沿失败所有者路径规划串行阶段，所有分支可推进才在独立事务中整组执行，失败或取消整组回滚。退出阶段的清理在决定生效之后进行，失败保留为可恢复重试队列。失败报告保留所有者路径、索引和 token。取消、deadline、资源授权、子 Case 调用仍经过现有 Project/Case/L0。

恢复兼容性：nsgablack 的 trajectory/poll/parameters/local_region Unit schema 已升 v2，候选关联反馈包含显式依赖版本。旧 v1 Unit checkpoint 不自动恢复或重新求值；需匹配旧版本软件或显式重新开始。GradientOptimizer 的历史评估指标仍是历史，不冒充更新后参数的当前目标。

验证：`tests/test_joint_repair_bridge.py`；跨进程串行/并行/DAG 与 Solver→Trainer 嵌套组合在 nsgablack 的 `tests/test_joint_repair_project.py` 中覆盖。
