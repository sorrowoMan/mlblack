# AGENTS.md

## 0) 使用方式

这份文档面向协作 Agent，目标是保证 `mlblack` 按新的 optimization-first 架构继续演进，而不是把旧 `mlblack` 的 family/trainer 耦合直接搬回来。

推荐阅读顺序：

1. 先看 `1/2/3`，确认项目定位、核心分层和运行数据流。
2. 再看 `4/5/6/7`，确认 state、resource、artifact、scaffold 边界。
3. 最后按 `10/11` 做实现和最小检查。

硬规则：

- 只复用旧机制思想，不复用旧耦合边界。
- `mlblack` 是 `nsgablack` 标准脚手架的 ML 特化层，不是第二套优化/编排框架。
- `nsgablack` 已有成熟的 solver、adapter 编排、group、serial、event、parallel、runtime/backend、L0 resource 能力；`mlblack` 不重复实现这些底座。
- `mlblack` 的正式复杂编排入口必须复用 `nsgablack`。低层 ML 组件可以保持轻量/可独立导入，但不能因此自建 workflow/runtime/L0。
- 不保留兼容层：发现 `runtime/workflow/resource_request/resource allocator` 配置时应报错，而不是静默降级。
- `mlblack` 只新增 ML 特有组件：representation、codec/decoder、head、problem/evaluation、inner parameter fitting、artifact/report、symbolic engine。
- 大对象进入 snapshot/artifact，context 只保留轻量字段或引用。
- 新增 demo/case 如果涉及 stage/group/event/parallel/backend/resource lease，必须通过 `nsgablack` 标准脚手架组装；`mlblack` 侧只暴露 inner training/evaluation surface。

## 1) 项目定位

`mlblack` 把机器学习视为优化问题：

```text
UnknownState
  -> ModelRepresentation.decode(...)
  -> model/function/spec
  -> LearningProblem.evaluate(...)
  -> Feedback(objectives, constraints, gradients, residuals)
  -> OptimizerAdapter.update(...)
```

核心对应关系：

| nsgablack | mlblack | 职责 |
| --- | --- | --- |
| Solver | Trainer | 控制平面、生命周期、状态、评估入口 |
| Adapter | OptimizerAdapter | 优化策略，如 GD、random search、torch backprop |
| Representation | ModelRepresentation + Codec + Head | 未知数编码/解码、模型输出语义 |
| Problem | LearningProblem | 唯一稳定吃数据的评估层 |
| Plugin | Capability | checkpoint、tracking、resource audit、report |
| Bias | OptimizationBias | 软偏好，不替代硬约束 |
| L0 Resource | injected ResourceContext / audit | 资源授权、调度、lease 属于 nsgablack；mlblack 只读取和审计 |

## 2) 架构边界

### Trainer

负责：

- `fit/step` 生命周期
- `evaluate_individual/evaluate_population`
- context/snapshot/artifact 状态边界
- adapter/representation/problem/capability/bias 装配

不负责：具体优化算法、模型结构、业务 objective、数据清洗细节、跨 trainer 编排、并行调度、资源授权。

### Adapter

负责：`propose(...)`、`update(...)`、adapter state/resume。

不负责：直接读训练数据、直接构造业务 artifact、接管 trainer 生命周期。

### Representation / Codec / Head

负责：`init`、`encode/decode`、`repair`、head 输出组合。

说明：

- `head` 是 decoder 输出语义的一部分。
- point/interval/probability 不应写成特殊 trainer。
- conditional/piecewise 用 router + branch representation/composer，不回到旧 family 耦合。
- 正交当前属于 representation/codec/feature map；正交评估可以作为 Problem/Capability 扩展，正交输出才属于 Head。

### Problem

负责消费数据、调用 model/spec 评估、返回 `Feedback`。

Problem 是唯一稳定吃数据的位置。Adapter 不直接吃数据。

### Capability

负责 checkpoint、experiment tracking、resource audit、report writer、lifecycle side effects。

Capability 不应改变优化语义；如果要影响优化方向，用 `OptimizationBias` 或 adapter。

## 3) 标准数据流

```text
adapter.propose
  -> representation.repair
  -> trainer.evaluate_population
  -> problem.evaluate
  -> bias.adjust_feedback
  -> adapter.update
  -> snapshot/context/report
```

说明：

- 单个 inner trainer 的 batch evaluation 可以保留为最小顺序语义。
- 并行、分组、stream/batch runtime、设备选择、资源 lease、跨 trainer portfolio 不属于 `mlblack` 主干，应交给 `nsgablack` 外层编排。

生命周期钩子：

```text
on_fit_start
on_step_start
on_evaluate_start
on_evaluate_end
on_step_end
on_fit_end
on_error
```

## 4) Context / Snapshot / State

Context 只放轻量信息：

- `run_name`
- `step`
- `resource.*`
- `last_population_snapshot`
- `pipeline` summary
- `signal.*` / reason / small metrics

不要长期把以下对象放进 context：

- full population
- full history
- full trace
- model object
- fitted estimator
- large arrays

这些应该进入：`SnapshotStore`、`ArtifactBundle`、`TrainerState`。

### 4.1 Context Contract

组件契约必须对齐 `nsgablack` 风格，使用普通字符串字段声明，不引入强制 `CTX.xxx` 常量层：

```python
context_requires = ("feedback.gradients", "candidate.unknown_state")
context_optional = ()
context_provides = ("population.candidates",)
context_mutates = ("adapter.current_state",)
context_cache = ()
requires_metrics = ()
metrics_fallback = "strict"
context_notes = "Reads gradients and current state; proposes next candidates."
```

规则：

- key 必须能被 `mlblack.core.context_keys` registry 校验。
- `doctor` 会扫描组件 class attrs 并校验 unknown context key、unknown metric key 和 invalid fallback。
- `catalog` 会动态解析 import path，并把统一 contract 注入 `CatalogEntry.contract`。
- `ComponentContract` 只是序列化兼容桥，组件源码中以 `context_*` class attrs 为主。

## 5) L0 Resource

`mlblack` 不拥有 L0 resource allocator。

资源第一原则：

- `nsgablack` owns resource authorization、lease、parallel scheduling、backend selection、solver fanout。
- `mlblack` owns passive `ResourceContext` consumption and audit only。
- `mlblack` 不实现新的 `LocalResourceAllocator`、`SQLiteLeaseStore`、GPU lease manager、thread scheduler。
- 不保留旧 allocator/lease-store 兼容类型。

嵌套关系：

```text
nsgablack outer allocator
  -> ResourceLease
  -> ResourceContext JSON
  -> mlblack inner training/evaluation task
```

`mlblack` 内部必须遵守外部注入的 device/thread/context，不允许私下写死 `cuda:0`、线程数或 backend。

## 6) Artifact / Replay

三类边界必须区分：

- `ModelArtifact`：训练产物
- `TrainerStateArtifact`：恢复/回放状态
- `RunReport`：审计报告

Typed artifact 可细化：

- `TreeEnsembleArtifact`
- `XGBoostArtifact`
- `SklearnMLPArtifact`
- `TorchModelArtifact`
- symbolic artifact 后续再接

不要把 artifact persistence 写进 adapter。

## 7) Assembly / Inner Training

标准 ML 组件入口：

```python
build_trainer(spec, data)
```

`TrainerAssemblySpec` 只负责单个 inner trainer 的 ML 组件装配：

- preset
- params
- biases
- capabilities
- component_overrides

`InnerTrainingAssemblySpec` 只允许表示 pipeline + single trainer 的声明数据。

禁止继续提供 `build_flow / MLFlow` 执行入口。

禁止把以下能力放进 `mlblack` assembly：

- trainer group / portfolio
- serial stage / workflow runner
- event router
- parallel runtime
- backend selection
- resource allocator / lease

这些必须由 `nsgablack` 标准脚手架负责。`mlblack` 侧应该暴露 training proxy、problem bridge、inner fitter、artifact builder 和 audit/report surface，供 `nsgablack` 调用。

### 7.1 Canonical Package Layout

当前实现主干已收敛到这些 canonical namespace：

- `core/`：Trainer、Adapter、Problem、Representation、Head、contracts、passive ResourceContext、state、artifact、store。
- `adapters/`：GD、random search、estimator search、torch backprop。
- `representations/`：representation、codec、model-space 解码。
- `representations/heads/`：point、interval、probability、piecewise head。
- `problems/`：监督/分类/条件 problem、跨框架 proxy/bridge、`problems/training/` task/result/contract。
- `pipeline/`：data view、pipeline components、feature_space、`pipeline/numericizer/`、`pipeline/conditional/`。
- `assembly/`：build_trainer、single-trainer assembly spec、schema/config。不要新增 `assembly/workflow` 机制。
- `catalog/`：registry/query/dashboard、`catalog/experiment/`。
- `capabilities/`、`bias/`、`models/`、`presets/`：保持独立主职责。

旧顶层包 `heads/`、`numericizer/`、`conditional/`、`workflow/`、`training/`、`problem/`、`experiment/`、`schema/`、`config/`、`data/` 已迁走。新增实现必须使用 canonical namespace，不要重新创建这些顶层包。

`assembly/workflow` 与 `core/runtime` 不属于 mlblack 主干；`core/resources` 只允许 passive `ResourceContext` 和 audit。

## 8) 当前已迁移能力

已落位：

- linear / orthogonal linear
- point / interval head
- logistic / softmax / probability calibration / piecewise head
- tree / xgboost estimator spec
- tree/boosting mechanism metadata: splitter / sampling / pruning / warm_start / continuation / early_stopping
- sklearn MLP spec
- numpy MLP + torch backprop with optimizer state / batching lifecycle / device policy
- supervised regression / interval regression / classification with AUC/F1/PR metrics
- piecewise representation / piecewise head / piecewise regression problem
- schema/config/scaffold
- training contract / problem bridge / proxy
- numericizer / feature_space
- conditional primitives / composer / branch model composition
- bias: noop / objective weight / state L2 / L2 scale / objective policy / branch policy / dynamic pool
- capability: checkpoint / experiment tracker / resource audit
- artifact bundle + typed model artifact + estimator state summary
- catalog / doctor / query/facet/deep-link / lightweight dashboard export
- experiment sqlite query/facet / lightweight dashboard export
- cross-framework resource-context case under `examples/cross_framework/`

已移除/迁出到 nsgablack-owned 语义：

- `assembly/workflow/*`：trainer candidate/group/stage/event/portfolio result 属于 `nsgablack` 编排层。
- `core/runtime.py`：serial/thread/batch/stream/facade backend 属于 `nsgablack` 外层 runtime/backend 管理。
- `core/resources.py`：allocator、lease store、heartbeat 等 L0 资源授权属于 `nsgablack`；`mlblack` 只保留 passive context/audit。
- `build_flow / MLFlow.workflow`：不再作为 mlblack 入口。

Symbolic first pass now exists:

- `models/symbolic.py`：fixed expression tree, `ParameterSpec`, numpy evaluator, expression stringification。
- `representations/codecs/symbolic.py`：fixed symbolic expression codec and multi-expression codec。
- `representations/symbolic.py`：fixed expression representation and fixed basis-set representation。
- `representations/heads/symbolic.py`：multi-symbol / basis-set head。
- `problems/symbolic.py`：fixed symbolic regression and orthogonal basis evaluation。
- `pipeline/symbolic/`：primitive registry, function-space objects, function-pool pipeline。
- `models/symbolic_gradient.py`：symbolic derivative expressions, chain-rule derivative values, parameter Jacobians, residual-gradient signals。
- `pipeline/symbolic/dynamic_pool.py`：residual/gradient/gate expansion and budget/redundancy pruning。
- `integrations/nsgablack_symbolic/orthogonal_problem.py`：Stage 1 outer basis problem; this optional integration may import `nsgablack` and delegates inner parameter fitting to mlblack。
- `pipeline/symbolic/grammar.py`：full primitive grammar, recursive unary/pair expansion, conditional lowering, dynamic activation config, family budgets。
- `integrations/nsgablack_symbolic/artifacts.py`：Stage 1 basis artifact, Stage 2 task artifact, lightweight symbolic artifact schema。
- `integrations/nsgablack_symbolic/task_symbolic_problem.py`：Stage 2 basis-conditioned outer task problem; outer selects function-pool terms over basis atoms, inner fits symbolic parameters。
- `integrations/nsgablack_symbolic/search_space.py`：index-coded function-pool search-space adapter。
- `integrations/nsgablack_symbolic/builders.py`：`build_symbolic_orthogonal_suite(...)` exposes a problem bundle for nsgablack-facing stages; it must not become an mlblack-owned workflow runner。

## 9) 实现规则

- 新增 ML 方法时，先判断属于哪一层：representation、codec、head、problem、adapter、capability、bias、artifact。
- 新增优化逻辑优先放 adapter。
- 新增模型输出形态优先放 head/model wrapper。
- 新增模型输出形态优先放 `representations/heads` 或 `models`。
- 新增数据准备优先放 `pipeline`、`pipeline/numericizer` 或 `pipeline/conditional`。
- 新增运行副作用优先放 capability。
- 新增软引导优先放 bias。
- 新增资源控制、并行调度、backend selection、group/stage/event 编排必须放到 `nsgablack` 或 `mlblack.integrations/nsgablack_*` 的 nsgablack-facing surface，不放到 `mlblack` 主干。
- 新增可复现产物优先放 artifact/state。

归属判断：

- 如果机制回答“怎么调度多个 trainer / 多个候选 / 多个资源 / 多个阶段”，它属于 `nsgablack`。
- 如果机制回答“一个 unknown state 怎么解码成 ML 模型/公式/输出 head”，它属于 `mlblack` representation/codec/head。
- 如果机制回答“这个模型怎么吃数据并产生 feedback”，它属于 `mlblack` problem/evaluation。
- 如果机制回答“外层怎么调用内层训练”，它属于 bridge/proxy/integration surface，编排权仍在 `nsgablack`。

## 10) 常用命令

```powershell
Set-Location "C:\Users\hp\Desktop\新建文件夹 (2)"

python -m compileall -q mlblack
python examples\orthogonal_point_demo.py
```

Doctor：

```powershell
python -c "from mlblack.project import run_project_doctor, format_doctor_report; print(format_doctor_report(run_project_doctor('.', strict=True)))"
```

## 11) 最小检查清单

- [ ] 是否保持 Trainer / Adapter / Representation / Problem / Capability 边界
- [ ] 是否避免 adapter 直接读数据
- [ ] 是否避免大对象写 context
- [ ] 是否只读取/审计外部注入的 `ResourceContext`，没有自建资源授权
- [ ] 是否能通过 `build_trainer` 或 nsgablack-facing proxy 装配
- [ ] 是否没有新增 mlblack-owned workflow/runtime/L0 编排
- [ ] 是否提供 `describe()` 或 contract/report surface
- [ ] 是否至少跑过 compileall 和一个 smoke
