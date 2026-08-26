# MLBlack

`mlblack` 是统一框架栈中的机器学习语义扩展包。它把数据、模型结构、参数状态、输出语义、训练问题、求值 Provider 和模型 Artifact 映射到 `nsgablack` 的统一 Solver 控制面，而不是维护第二套 Trainer 循环或 Adapter 体系。

当前版本：`0.4.22`，依赖 `blackbase>=0.3.24,<0.4.0` 与 `nsgablack>=0.3.26,<0.4.0`。

本版将最终模型、best state、history 与 report 纳入同一个 Artifact finalization session：
teardown 成功后只冻结暂存引用，全部严格 `on_solver_finalization_prepare` 参与者通过后，
由 BlackBase 在 cleanup、序列化和物理内容复核成功后原子 seal；`on_solver_finalized` 只观察
带 PublicationReceipt 的 TrainerResult 与 Artifact refs，观察器异常只能形成诊断。任一暂存、发布前校验
或 commit 失败都会 abort 整组正式引用；直接手工调用最终发布入口也会
被拒绝，避免绕过统一 Solver 生命周期制造伪成功产物。

本版把 ML 最终产物发布移出 teardown 前的 finishing 阶段：`best_model`、大型 best state、
history、最终 report 与 `TrainerResult` 只会在 Provider、Adapter 和 backend session 全部
成功关闭后发布。teardown 失败的运行不会再在 ArtifactStore 留下看似成功的正式产物。

本版的 Torch Evaluation Provider 已纳入 BlackBase 首次使用自动 COW 认证：Adam/AdamW
发布新 optimizer slot 后，旧 slot 必须仍可通过正式 materialization surface 读取，认证
结果按 Provider 与 method 留存审计，避免后端实现回退为破坏事务回滚的原地覆盖。

## 统一视图

```text
DataView / DatasetSchema
    -> ModelRepresentation + Codec
    -> UnknownState / StateRef
    -> LearningProblem
    -> Evaluation Provider
    -> OptimizationFeedbackBatch
    -> nsgablack Adapter
    -> LearningSolver
    -> Typed Model Artifact
```

在这个视图中：

- “模型训练”是对模型语义状态的求解。
- `Codec` 定义候选状态怎样编码为算法可处理的数值视图，以及怎样重建模型。
- `LearningProblem` 定义 loss、metric、constraint、数据切分和输出语义。
- `Evaluation Provider` 实现 NumPy、Torch、JAX、TensorFlow、sklearn 或第三方 estimator 的实际求值/状态转移。
- `nsgablack Adapter` 决定 Adam、梯度法、进化、多目标、嵌套或组合搜索策略。
- `LearningSolver` 是面向 ML 用户的 facade，底层仍使用同一个 Solver 生命周期。

闭式拟合和第三方 `estimator.fit()` 仍然属于 Provider 的求值机制；它们不会产生第二套控制平面。

`CompletionPolicy` 消费 nsgablack 的可恢复 `RunProgressState`：checkpoint 恢复后继续同一次逻辑训练运行的累计 step、elapsed/deadline 和自定义 policy state，不会从零重新计数。`LearningSolver.set_state()` 只能在 post-setup restore transaction 中修改运行态。Torch Provider 的活状态保留按 `(scope_id, trajectory_id)` 分区；输入 `StateRef` 会先物化，随后 retention 与整批新引用在一个可回滚事务中发布。optimizer slot transition 采用 copy-on-write：被拒绝的梯度尝试通过 `StateReleaseRequest.state_ids` 只释放新 slot 并恢复前驱引用，成功提交才回收前驱 slot。

Gradient Adapter 是其 optimizer slot 的直接清理所有者：提交后按旧 state ID 精确回收，失败引用进入可重试队列，teardown 再完成定向释放。`LearningSolver` 的整 trajectory release 只作为幂等兜底。模型结果与最终 checkpoint 在 Provider、Adapter 和 backend session 全部 teardown 成功后才发布。

ML 最佳结果只从 canonical `EvaluationEventEnvelope` 的 CandidateBatch、OptimizationFeedbackBatch 与 candidate token 建立绑定；私有运行缓存只保留不可序列化 model handle。Provider/Plugin 短路评估不会再形成第二本反馈账，缺少 token 对应证据时会 fail-closed。
这些 Event 现在由共享 `EvaluationEvidenceJournal` 建立可恢复索引；ML 层不另建训练日志，也不会在恢复时隐式重跑具有副作用的 Provider 求值。

## 资源与 backend

GPU、CPU、线程、显存和设备 token 由 Project L0 授权。Provider 声明 required / preferred capability，并在注入的 `ResourceContext` 内选择可用 backend。

Adapter 不拥有 GPU，backend 也不自行申请全局资源：

```text
Project L0 grant
    -> ResourceContext
    -> LearningProblem request
    -> EvaluationProviderSpec capability match
    -> backend session
```

## 稳定装配

用户可以继续使用 ML 词汇：

```python
from mlblack import build_trainer

trainer = build_trainer(
    problem=problem,
    model=model,
    method="gradient.adam",
)
```

`build_trainer` 是语义装配 facade：Catalog 把稳定方法标识解析为 nsgablack Adapter、mlblack Provider、数据调度和共享 runtime。它不是私有 Trainer 基类的工厂。

标准 Case 仍使用统一入口：

- `build_solver.py`：唯一规范装配入口。
- `run_solver.py`：唯一规范运行入口。
- `.case kind=trainer`：只表达语义与 Catalog 分类。
- `build_trainer.py` / `run_trainer.py`：可选薄别名。

每个 Project 和 Case 只维护一个 `README.md`。

## 核心公共面

- `LearningSolver` / `build_learning_solver`
- `TrainerAssemblySpec` / `build_trainer`
- `ModelRepresentation`、Codec 与 Head
- `TrainingTask` / `TrainingContract`
- Evaluation Provider 与 backend session
- `ArtifactBuilder`、`TypedModelArtifact` 与 `TrainerResult`

## 快速开始

```powershell
python -m pip install -e .[all]
python -m pytest -q
python -m mlblack catalog list
```

标准跨框架项目由 nsgablack CLI 创建：

```powershell
python -m nsgablack project new demo_project
Set-Location demo_project
python -m nsgablack project add-case fit_model --type trainer --framework mlblack
python -m nsgablack project doctor --path . --build --strict
```

## 文档

- [文档入口](docs/README.md)
- [统一优化视图](docs/architecture/UNIFIED_OPTIMIZATION_VIEW.md)
- [线性模型完整切片](docs/architecture/LINEAR_MODEL_OPTIMIZATION_SLICE_CN.md)
- [统一梯度路径](docs/architecture/UNIFIED_GRADIENT_PATH.md)
- [神经网络 Provider / backend 边界](docs/neural_graph_backend_architecture/README.md)
- [标准脚手架教程](docs/standard_scaffold_tutorial/README.md)
- [学习材料](docs/learning/ML_TERMINOLOGY_MAP_CN.md)
- [示例项目](examples/cases/README.md)

历史实现总结和旧的 ML Adapter/Trainer 控制面已删除；需要考古时使用 Git 历史。
