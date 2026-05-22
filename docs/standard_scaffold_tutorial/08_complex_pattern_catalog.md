# 08. 复杂组合模式目录

这一章把“理论上可以畅爽组合”的模式系统列出来。每个模式都说明应该怎么落层，避免一兴奋就写出新的 `HybridTrainer` 或 `Workflow`。

## 1. 读法

每个模式拆成四列：

```text
结构意图:
  想表达什么模型/训练过程。

nsgablack owns:
  阶段、并行、资源、outer search。

mlblack owns:
  模型语义、数据/目标变换、inner trainer、artifact。

标准组件:
  当前可用或应扩展的位置。
```

## 2. Baseline + Residual

```text
final = base + residual
```

| 归属 | 内容 |
| --- | --- |
| nsgablack | stage1 train base, stage2 train residual, stage3 compose |
| mlblack | `ModelConditionedTargetComponent`, `IntegratedPredictionModel` |
| artifact | component refs, additive spec, residual metrics |

适合：主趋势明显、误差结构可学习。

## 3. Multi-round Residual / Boosting-like

```text
final = model_0 + model_1 + ... + model_k
model_i fits residual from previous integrated model
```

| 归属 | 内容 |
| --- | --- |
| nsgablack | serial stages, early stop, round budget |
| mlblack | residual target builder, additive integration |
| outer search | round count, learner type, weights |

适合：逐轮修正误差。

## 4. Stacking

```text
base models -> predictions -> meta learner
```

| 归属 | 内容 |
| --- | --- |
| nsgablack | base learner group, meta stage |
| mlblack | append prediction feature, meta trainer |
| extension | sequential prediction wrapper for inference-time feature generation |

适合：多个基础模型互补。

## 5. Weighted Late Fusion

```text
final = w1*m1 + w2*m2 + ...
```

| 归属 | 内容 |
| --- | --- |
| nsgablack | search weights / choose component subset |
| mlblack | `PredictionIntegrationComponent.additive(weights=...)` |
| contract | all outputs point vector and row-aligned |

适合：多模型投票、不同模态分数融合。

## 6. Multi-modal Fusion

```text
text branch + image branch + tabular branch -> fused score
```

| branch | input contract |
| --- | --- |
| text | token ids, `ndim=2` |
| image | NCHW image, `ndim=4` |
| tabular | numeric matrix, `ndim=2`, `n_features=k` |

落层：

```text
nsgablack group:
  train/evaluate branches independently or with scheduled dependencies

mlblack composition:
  PredictionIOContract routes inputs by key
```

## 7. Main Model + Local Corrector

```text
main_model handles global trend
corrector handles local bias or rare region
final = main + alpha * corrector
```

| 归属 | 内容 |
| --- | --- |
| nsgablack | choose corrector region, alpha, budget |
| mlblack | correction data transform, additive integration |
| possible extension | region mask / router head |

## 8. Expert Ensemble

```text
experts = {expert_0, expert_1, ...}
final = aggregate(experts)
```

当前可做：weighted mean/additive。

后续可扩展：

```text
GatedIntegratedPredictionModel:
  gate_model predicts row-wise expert weights
```

归属：gate 是 model semantic；训练 gate 的阶段由 nsgablack 编排。

## 9. Router + Branch Model

```text
router(X) -> branch id / soft weights
branch_model_i(X) -> prediction
```

当前已有相关能力：piecewise / conditional primitives。

| 归属 | 内容 |
| --- | --- |
| mlblack | router model, piecewise head, branch model composition |
| nsgablack | branch structure search, staged refinement |

## 10. Symbolic + Neural Hybrid

```text
symbolic_model captures interpretable law
neural_model captures residual/high-frequency error
final = symbolic + neural_residual
```

| 归属 | 内容 |
| --- | --- |
| nsgablack | symbolic structure stage, neural residual stage |
| mlblack | symbolic representation/problem, neural graph trainer, additive integration |
| artifact | symbolic canonical schema + neural graph artifact + integrated artifact |

## 11. Tree + Neural Hybrid

```text
tree_model handles tabular discontinuities
neural_model handles smooth residual
```

落层：

```text
stage1: tree_estimator_search
stage2: residual target
stage3: numpy_mlp_torch_backprop or neural graph
stage4: integrated model
```

## 12. Linear + Symbolic + Tree + Neural

夸张组合也能表达：

```text
linear baseline
+ symbolic law
+ tree correction
+ neural residual
```

I/O contract 可以要求它们都吃同一个 numeric X，或者不同组件吃不同 feature subset。

```python
final = PredictionIntegrationComponent.additive(
    component_order=("linear", "symbolic", "tree", "neural"),
    weights={"linear": 1.0, "symbolic": 1.0, "tree": 0.5, "neural": 0.2},
).compose({...})
```

## 13. Pretrained Model + Local Adapter

```text
pretrained_model frozen
local adapter / LoRA / small correction trained locally
```

| 归属 | 内容 |
| --- | --- |
| mlblack | pretrained bridge, adapter model, artifact summary |
| nsgablack | prompt/RAG/tool/adapter search, resource allocation |

## 14. Prompt / RAG / Tool Policy Search

LLM 不一定要训练。可以把它作为 black-box evaluator/generator。

```text
outer candidate:
  prompt template
  RAG top_k
  retrieval filter
  tool plan
  decoding params

inner evaluation:
  call LLM bridge
  score quality/cost/latency
```

归属：

```text
nsgablack:
  searches configuration

mlblack:
  exposes evaluation proxy/artifact/report if ML semantics are needed
```

## 15. Multi-head Model

```text
shared backbone
  -> classification head
  -> ranking head
  -> embedding head
  -> preference head
```

| 归属 | 内容 |
| --- | --- |
| mlblack | neural graph head specs, problem metrics |
| nsgablack | outer head selection/search, multi-objective tradeoff |

## 16. Cascaded Model

```text
cheap_model first
if uncertain -> expensive_model
```

落层：

```text
mlblack:
  cascaded prediction wrapper / uncertainty output

nsgablack:
  searches threshold, schedules expensive evaluation budget
```

## 17. Distillation

```text
teacher_model predicts soft target
student_model trains on teacher target
```

| 归属 | 内容 |
| --- | --- |
| mlblack | model-conditioned target mode = prediction |
| nsgablack | teacher/student stage order and budget |

## 18. Active Learning Loop

```text
model trains
uncertainty selects new samples
model retrains
```

归属：

```text
nsgablack:
  loop/stage/data acquisition scheduling

mlblack:
  uncertainty metrics/problem/report/data view updates
```

不要在 mlblack 主干写 active learning workflow。

## 19. Curriculum / Progressive Training

```text
stage 1 easy data
stage 2 harder data
stage 3 full data
```

| 归属 | 内容 |
| --- | --- |
| nsgablack | stage schedule |
| mlblack | data pipeline filters/views, trainer specs |

## 20. Constraint-aware Model Search

```text
optimize loss, complexity, latency, memory, stability
```

| 归属 | 内容 |
| --- | --- |
| nsgablack | multi-objective outer optimization |
| mlblack | metrics/artifacts to project objectives |

## 21. Robustness Ensemble

```text
train same model under perturbations/seeds
aggregate stable predictors
```

| 归属 | 内容 |
| --- | --- |
| nsgablack | seed/resource/fanout |
| mlblack | trainer result, integrated mean model |

## 22. Architecture Search

```text
outer searches NeuralGraphSpec
inner trains fixed spec
```

| 归属 | 内容 |
| --- | --- |
| nsgablack | search hidden_dim/layers/heads/ffn/norm/lora |
| mlblack | NeuralGraphSpec/Codec/Problem/Artifact |

## 23. Symbolic Grammar Search

```text
outer searches grammar/primitive activation
inner fits parameterized expression
```

| 归属 | 内容 |
| --- | --- |
| nsgablack | primitive family budgets, graph/path search |
| mlblack | symbolic grammar, normalization, recovery scoring |

## 24. Pattern 选择建议

| 数据/目标 | 优先模式 |
| --- | --- |
| 主趋势简单、误差复杂 | baseline + residual |
| 多模型互补 | stacking / weighted fusion |
| 多数据源 | multi-modal I/O contract fusion |
| 需要可解释主项 | symbolic + residual |
| 表格非线性强 | tree + neural residual |
| 大模型不能训练 | prompt/RAG/tool policy search |
| 需要多目标权衡 | nsgablack outer multi-objective |
| 资源昂贵 | cascade / budgeted evaluation |

## 25. 总结

所有复杂组合都落到同一个原则：

```text
mlblack:
  定义单模型、数据变换、模型整合、artifact。

nsgablack:
  定义多个训练/评估任务的顺序、并行、资源、搜索。
```

只要不把“训练过程编排”塞回 `mlblack`，这些模式可以无限组合，而且每个部分都能被测试、审计和替换。
