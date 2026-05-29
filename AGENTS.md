# AGENTS.md

## 0) 使用方式

这份文档面向协作 Agent，目标是保证 `mlblack` 按新的 optimization-first 架构继续演进，而不是把旧 `mlblack` 的 family/trainer 耦合直接搬回来。

推荐阅读顺序：

1. 先看 `1/2/3`，确认项目定位、双层编排模型和运行数据流。
2. 再看 `4/5/6/7`，确认 state、resource、artifact、scaffold 边界。
3. 最后按 `10/11` 做实现和最小检查。

硬规则：

- 只复用旧机制思想，不复用旧耦合边界。
- `mlblack` 是 `nsgablack` 标准脚手架的 ML 特化层，不是第二套优化/编排框架。
- `nsgablack` 已有成熟的 solver group、并行调度、事件路由、resource lease 等外层编排能力；`mlblack` 不重复实现这些外层底座。
- `mlblack` 支持内层编排（`SerialTrainer`、`ModelConditionedTarget`、`DataPipeline`），用于单次训练任务内的顺序多阶段/多模型组合。跨 solver 的并行/分组/资源调度仍必须走 `nsgablack`。
- 不保留旧兼容层：发现 `runtime/workflow/resource_request/resource allocator` 等旧外层编排配置时应报错，而不是静默降级。
- `mlblack` 只新增 ML 特有组件：representation、codec/decoder、head、problem/evaluation、inner parameter fitting、artifact/report、symbolic engine、inner stage orchestration。
- 大对象进入 snapshot/artifact，context 只保留轻量字段或引用。
- 新增 demo/case 如果涉及 group/event/parallel/backend/resource lease，必须通过 `nsgablack` 标准脚手架组装；`mlblack` 侧只暴露 inner training/evaluation surface 和内层多 stage 串联。

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

## 2) 双层编排模型：内层 mlblack / 外层 nsgablack

核心区分：

| | 内层编排 (mlblack) | 外层编排 (nsgablack) |
|---|---|---|
| 粒度 | 单次训练任务内部 | 多个训练/评估任务之间 |
| 机制 | SerialTrainer, ModelConditionedTarget, DataPipeline, IntegratedPredictionModel | SolverGroup, SerialStageSolver, EventRouter, ParallelRuntime |
| 并发 | 单线程顺序 | 多线程/多进程/分布式 |
| 资源 | 被动消费一个 ResourceContext | 主动分配、租约、调度 |
| artifact | ArtifactRef 流（snapshot 引用，不过 trainer 边界） | 跨 solver 产物传递、portfolio 聚合 |

**判断归属**：一个机制如果回答“一个训练任务内部怎么做多阶段/多模型组合”，它属于 mlblack 内层编排。如果回答“多个训练/评估任务之间怎么调度、并行、分配资源”，它属于 nsgablack 外层编排。

### mlblack 内层编排（当前已实现）

#### SerialTrainer（`core/trainer_stage.py`）

顺序串联多个 Trainer，artifact 在 stage 间流动：

```python
stages = [
    StageSpec(name="pretrain", factory=lambda: pretrain_trainer, output_artifacts=["init_state"]),
    StageSpec(name="finetune", factory=lambda: finetune_trainer, input_artifacts={"init_state": "pretrain.init_state"}),
]
result = SerialTrainer(stages).fit()
```

- 每个 stage 是独立的 Trainer（有自己的 Representation/Problem/Adapter）
- artifact 通过 `ArtifactRef` 共享注册表流转（大对象走 snapshot_store，小对象 inline）
- 返回聚合 `StageResult` 历史
- 本质是**单线程顺序执行**，不涉及并行、组调度、资源分配

#### ModelConditionedTargetComponent（`pipeline/model_conditioning.py`）

数据层面的阶段化：用已训练模型的预测结果变换下一阶段的 target（如 `y' = y - model.predict(X)`），实现残差学习/boosting 风格的多阶段训练。不是特殊 Trainer，是数据 pipeline 内的 target 变换。

#### IntegratedPredictionModel（`models/composition.py`）

模型层面的多模型组合：将已训练的多个 component model 在推理时按权重/策略集成。**不是 workflow runner**，是模型语义边界。

#### DataPipeline（`pipeline/base.py`）

有序数据变换链：fit → transform 顺序执行，为 Trainer 准备数据。pipeline 内的 component 可以是有状态的（如 numericizer）。

### 外层编排（nsgablack 专属）

以下能力 mlblack **不支持**，必须走 nsgablack：

- trainer **group** / portfolio：同时管理多个 Trainer 实例
- **parallel** runtime：多线程/多进程执行
- **event** router：异步事件驱动编排
- **resource** allocator / lease：GPU 分配、线程池管理
- **backend** selection：自动选择计算后端
- 跨 solver 的 **stage** 编排：用 nsgablack 的 `SerialStageSolver`

三层嵌套关系：

```text
nsgablack outer orchestration (groups, parallel, events, resources)
  ├── Solver A
  │     └── Problem (evaluate calls mlblack TrainingProxy)
  │           ├── mlblack SerialTrainer (inner multi-stage)
  │           │     ├── Stage 1: ComposableTrainer
  │           │     └── Stage 2: ComposableTrainer
  │           └── artifact flow within SerialTrainer
  └── Solver B
        └── ...
```

### 为什么这样分？

如果把内层编排（SerialTrainer）删除，每个 stage 必须用一个 nsgablack Solver 包裹——这对“预训练 → 微调”这种顺序组合来说是过度工程。SerialTrainer 提供轻量的顺序串联，无需外层调度器介入。

反过来，如果把并行/分组/资源调度放进 mlblack，等于在 mlblack 内部重建一个 nsgablack，两个框架会各自为政。

## 3) 架构边界

### Trainer

负责：

- `fit/step` 生命周期
- `evaluate_individual/evaluate_population`
- context/snapshot/artifact 状态边界
- adapter/representation/problem/capability/bias 装配
- **内层多 stage 串联**（`SerialTrainer`）

不负责：具体优化算法、模型结构、业务 objective、数据清洗细节、**并行调度、资源授权、跨 solver 编排**。

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

### Plugin

Owns checkpointing, experiment tracking, resource audit, report writing, and other lifecycle side effects. Legacy Capability semantics are folded into the Plugin vocabulary; new cases must not create case-level `capabilities/` directories.

Plugin must not change optimization semantics. If a component needs to change optimization direction, use `OptimizationBias` or an adapter.


## 4) 标准数据流

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

## 5) Context / Snapshot / State

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

## 6) L0 Resource

`mlblack` 不拥有 L0 resource allocator。

资源第一原则：

- `nsgablack` owns resource authorization、lease、parallel scheduling、backend selection、solver fanout。
- `mlblack` owns passive `ResourceContext` consumption and audit only。
- `mlblack` 拥有 `ComputeBackendSession` 和 `backends/`（torch/numpy/jax/tensorflow）作为 **intra-evaluation compute layer**——负责单次评估内的张量运算、自动微分、参数优化。不管理跨 solver 的资源调度。
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

## 6.1 Project / Case / Scaffold Directory Rule

Use the same three-layer structure as nsgablack: Project -> Case -> Standard Scaffold. Solver and Trainer are the same abstraction level. The directory template is identical; catalog kind and ML semantics are the only difference.

### 6.1.1 Three Layers

1. Project
   - Owns cross-case orchestration, ResourceContext injection, and one project entrypoint.
   - Typical shape: project_config.py, run_project.py, cases/.

2. Case
   - Owns one independently discoverable and testable Solver/Trainer unit.
   - Typical location: cases/<case_name>/ or examples/cases/<case_name>/.

3. Standard Scaffold
   - Owns problem, pipeline, adapter, plugins, runtime surface, and local config.
   - build_solver.py is canonical. build_trainer.py is only an alias.

### 6.1.2 Unified Case Template

```text
<case_name>/
  __init__.py
  build_solver.py           # canonical assembly entry
  build_trainer.py          # alias: from .build_solver import build_solver as build_trainer
  run_solver.py             # canonical CLI entry
  run_trainer.py            # alias: from .run_solver import main
  config.py                 # component registry aggregator
  problem/
  pipeline/                 # data pipeline + representation/codec sublayer
    representation/
  adapter/
  bias/
  plugins/                  # lifecycle capabilities; replaces legacy capabilities/
  evaluation/
  runtime/
  solver/
```

### 6.1.3 Hard Rules

- build_solver.py is the canonical assembly entry; build_trainer.py must be a thin alias only.
- run_solver.py is the canonical CLI entry; run_trainer.py must be a thin alias only.
- Case-level capabilities/ is forbidden; use plugins/.
- Case-level representation/ is forbidden; use pipeline/representation/.
- assembly/scaffold.json is forbidden; assembly logic belongs in build_solver.py.
- assembly/ may temporarily hold preset registry docs, but not the runtime truth source.
- Cross-case dependencies must flow through Artifact, SnapshotStore, ResourceContext, or result payloads.
- New or migrated examples must make their real problem / pipeline / adapter / plugins visible from build_solver.py or an equivalent --check path.

## 7) Artifact / Replay

Keep these boundaries separate:

- ModelArtifact: fitted model output.
- TrainerStateArtifact: resume/replay state.
- RunReport: audit/report payload.

Do not put artifact persistence into adapters.

## 8) Assembly / Inner Training

Standard ML component entry remains:

```python
build_trainer(spec, data)
```

TrainerAssemblySpec only builds one inner trainer:

- preset
- params
- biases
- plugins / legacy capabilities
- component_overrides

InnerTrainingAssemblySpec may describe pipeline + single trainer only.

Forbidden in mlblack assembly:

- trainer group / portfolio
- nsgablack SerialStageSolver-level cross-solver stage
- event router
- parallel runtime
- backend selection
- resource allocator / lease

Allowed mlblack inner orchestration:

- SerialTrainer: sequential trainer stages with artifact flow
- DataPipeline: ordered fit/transform data chain
- ModelConditionedTargetComponent: target transformation using prior model output

Cross-solver orchestration belongs to nsgablack. mlblack should expose training proxy, problem bridge, inner fitter, artifact builder, and audit/report surface.

### 8.1 Canonical Package Layout

- core/: Trainer, Adapter, Problem, Representation, Head, contracts, passive ResourceContext, state, artifacts, stores.
- adapters/: GD, random search, estimator search, torch backprop.
- representations/: representation, codec, model-space decode.
- representations/heads/: point, interval, probability, piecewise, symbolic heads.
- problems/: supervised/classification/conditional problems, bridge/proxy, training task/result/contract.
- pipeline/: data views, pipeline components, feature space, numericizer, conditional, data_views.
- assembly/: build_trainer, single-trainer assembly spec, schema/config. Do not add assembly/workflow.
- catalog/: registry/query/dashboard and experiment catalog.
- plugins/: case-level lifecycle side effects.
- capabilities/: top-level legacy/backend capability namespace only; do not create case-level capabilities/.
- bias/, models/, presets/: keep their independent responsibilities.

## 9) Current Migrated Capabilities

Already present:

- linear / orthogonal linear
- point / interval head
- logistic / softmax / probability calibration / piecewise head
- tree / xgboost estimator spec
- sklearn MLP spec
- numpy MLP + torch backprop
- supervised regression / interval regression / classification metrics
- piecewise representation / piecewise head / piecewise regression problem
- schema/config/scaffold
- training contract / problem bridge / proxy
- numericizer / feature_space
- conditional primitives / composer / branch model composition
- bias: noop / objective weight / state L2 / L2 scale / objective policy / branch policy / dynamic pool
- plugin: checkpoint / experiment tracker / resource audit
- artifact bundle + typed model artifact + estimator state summary
- catalog / doctor / query/facet/deep-link / dashboard export
- time-series DataView, temporal neural presets, ARIMA/SARIMAX provider route
- symbolic model/codec/head/problem/pipeline and nsgablack-facing symbolic integrations

Moved to nsgablack-owned semantics:

- assembly/workflow/* group/stage/event/portfolio orchestration
- core/runtime.py serial/thread/batch/stream runtime backend
- resource allocator / lease store / heartbeat
- build_flow / MLFlow.workflow

## 10) Implementation Rules

- Before adding a capability, classify it as representation, codec, head, problem, adapter, plugin, bias, artifact, backend provider, or nsgablack orchestration.
- Put optimization logic in adapters.
- Put model output semantics in heads or model wrappers.
- Put data preparation in pipeline, pipeline/data_views, numericizer, or conditional.
- Put lifecycle side effects in plugins.
- Put soft guidance in bias.
- Put resource control, parallel scheduling, backend selection, group/stage/event orchestration in nsgablack or an nsgablack-facing integration surface.
- Put reproducible outputs in artifact/state/report surfaces.

Ownership decision:

- Multiple trainers/candidates/resources/stages scheduling -> nsgablack.
- One training task internal sequence/composition -> mlblack inner orchestration.
- Unknown state to ML model/formula/head -> mlblack representation/codec/head.
- Data + model to feedback -> mlblack problem/evaluation.
- Outer layer calling inner training -> bridge/proxy/integration, with orchestration authority still in nsgablack.

## 11) Common Commands

```powershell
Set-Location "C:\Users\hp\Desktop\mlblack"

python -m compileall -q project examples\cases
python -c "from mlblack.project.doctor import run_project_doctor, format_doctor_report; print(format_doctor_report(run_project_doctor('.', strict=True)))"
```

## 12) Minimum Checklist

- [ ] Keep Trainer / Adapter / Representation / Problem / Plugin boundaries.
- [ ] Avoid adapters reading data directly.
- [ ] Avoid writing large objects to context.
- [ ] Only read/audit injected ResourceContext; do not create a resource allocator.
- [ ] Provide canonical build_solver and alias build_trainer when this is a case scaffold.
- [ ] Do not add mlblack-owned workflow/runtime/L0 orchestration, except inner SerialTrainer/DataPipeline/ModelConditionedTarget.
- [ ] Provide describe(), contract, artifact, or report surface where appropriate.
- [ ] Run compileall and at least one smoke/doctor check.
