# 机器学习模型族统一接入指南

这份文档用于判断一个新的模型族应该接到 `mlblack`、`nsgablack`，还是外部 domain backend。它不把两边看成互相调用的脚本，而是看成一个统一框架栈中的不同语义层。

核心规则：

> `nsgablack` 和 `mlblack` 共享统一的 Project / Case / Scaffold / L0 substrate。`nsgablack` 是优化搜索语义层，`mlblack` 是机器学习语义层。编排和资源授权属于 substrate，不属于任一语义层的私有能力。

## 1. 先判断落层

| 需求 | 正确落点 |
| --- | --- |
| 新输入结构、数据视图、schema | `mlblack` DataView / pipeline |
| 新模型状态、结构规格 | `mlblack` Spec / Representation / Codec |
| 新输出语义 | `mlblack` Head |
| 新 loss、metric、residual、训练反馈 | `mlblack` Problem |
| 新拟合/推理后端能力 | `mlblack` Trainer / Provider / backend capability |
| 新模型、报告、checkpoint 产物 | `mlblack` Artifact |
| 搜结构、超参、策略、预算、tradeoff | 优化搜索 Case，通常用 `nsgablack` 语义实现 |
| 多 Case 顺序、并行、重试、artifact handoff、资源 grant | 共享 Project / L0 substrate |
| 数值求解器、仿真器、数据库、向量索引、对象存储、云运行时 | provider / bridge / runtime surface |

不要为每个模型族新建 `mlblack` 私有 runner、私有 scheduler、私有资源分配器或私有编排栈。

## 2. 标准接入形态

```text
Project substrate
  -> 选择 Case 顺序并发放 ResourceContext

outer Case（可选）
  -> 搜索或选择 component_overrides
  -> 传入结构化 request payload

mlblack Case
  -> DataView / pipeline
  -> Spec / Codec / Head
  -> Problem / Trainer / Provider
  -> Artifact / Report / audit
  -> 返回 metrics、artifact refs、resource audit
```

外层 Case 可以是 `nsgablack`，可以是 `mlblack`，也可以是其他标准 Case。关键不是仓库名，而是是否遵守 Case 边界。

## 3. 先复用已有 primitive

新增模型族前，先看它是不是已有 primitive 的组合：

| 表面模型族 | 通常由什么组合 |
| --- | --- |
| 时序预测 | DataView + window/lag/horizon spec + forecast head + backtest problem |
| Transformer | NeuralGraphSpec + codec/backend lowering + LM/classification/embedding head + backprop trainer |
| PINN | neural function spec + physics residual problem + numerical solver bridge |
| Neural ODE | dynamics model + ODE solver bridge + trajectory problem |
| 多模态 | 多个 DataView + branch model + PredictionIOContract + fusion head/model |
| 符号学习 | symbolic spec / feature space + constant fitter + expression artifact |
| 推荐 / 检索 | user-item DataView + embedding/retrieval head + ranking problem + index artifact/provider |
| 不确定性 / 区间 | interval/distribution head + calibration problem + report artifact |

如果缺的是一个局部语义对象，就补这个对象。  
如果缺的是阶段顺序、并行、资源授权或嵌套评估，就交给共享 substrate。

## 4. 资源规则

Project L0 发放 `ResourceContext`。`mlblack` Case 只消费并审计这个 grant：

```python
def build_solver(config=None, *, resource_context=None, component_overrides=None):
    runtime = RuntimeProfile.from_context(resource_context)
    trainer = build_trainer_from_config(config, runtime=runtime)
    trainer.add_plugin(build_resource_audit_plugin(runtime))
    return trainer
```

文档和配置中使用逻辑资源 token。不要在 Trainer 或 Provider 内部写死本机设备名，也不要创建第二套全局 lease 系统。

## 5. Payload 与 Artifact 契约

层间 payload 应该是 JSON-compatible，并通过引用传递大对象：

```text
request:
  candidate_id
  component_overrides
  artifact_refs
  budget
  resource_context

result:
  objectives
  violations
  metrics
  artifact_refs
  audit
```

拟合后的模型、权重、tensor、history、trace、graph cache 等大对象进入 Artifact 或 Snapshot。

## 6. 禁止回退模式

不要新增：

```text
TimeSeriesPrivateRunner
PINNPrivateRunner
MultiModalRuntime
HybridTrainer
StageRunner
PrivateResourceAllocator
TransformerRuntimeBackend as orchestration
case-local global GPU lease
adapter directly reading X/y
codec silently switching backend
```

应该使用：

```text
DataView / Spec / Codec / Head / Problem / Trainer / Provider / Artifact
Project / Case / Scaffold / L0 substrate
standard component_overrides
ResourceContext
Artifact/Snapshot refs
```

## 7. 新模型族 PR 检查

- [ ] 是否拆成了明确的 ML 语义组件。
- [ ] 如需搜索，是否表达成优化搜索 Case。
- [ ] 多阶段或嵌套执行是否使用 Project / Case / Scaffold。
- [ ] `build_solver.py` 是否仍是 canonical entry。
- [ ] `build_trainer.py` 是否只是 alias。
- [ ] 资源是否来自注入的 `ResourceContext`。
- [ ] Artifact 是否描述 model family、head、problem、backend、metrics 和 resource audit。
- [ ] 是否没有新增私有编排或私有 L0。
