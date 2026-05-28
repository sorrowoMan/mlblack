# 02. 组件配置完整拆解

这一章按真实分层讲配置。判断一个组件放哪里，只问一个问题：它回答的是“模型语义”、“数据评估”、“参数更新”、“运行副作用”，还是“外层编排”？

## 1. 总览

```text
Data / Pipeline
  -> Representation / Codec / Head
  -> Problem
  -> Adapter
  -> Backend
  -> Capability / Bias
  -> Artifact / Report
```

对应职责：

| 层 | 输入 | 输出 | 典型组件 |
| --- | --- | --- | --- |
| Data | raw arrays / rows | `NumericDataView` | numericizer, tokenizer, split |
| Pipeline | data view | transformed data view | zscore, columns, model-conditioned target |
| Representation | `UnknownState` | model/spec/expression | linear, estimator, neural graph, symbolic |
| Codec | flat values | typed model | linear codec, neural codec, symbolic codec |
| Head | base model output | semantic output | point, interval, probability, piecewise |
| Problem | model + data | `Feedback` | regression, classification, LM, DPO |
| Adapter | feedback + state | next state | GD, random, estimator search, backprop |
| Backend | tensor/model op | executable op | torch, jax, tensorflow, numpy |
| Capability | lifecycle event | side effect/report | checkpoint, tracking, resource audit |
| Artifact | result/model/report | reproducible payload | model artifact, integrated model artifact |

## 2. Data 与 NumericDataView

`NumericDataView` 是普通监督任务的稳定边界：

```python
from mlblack.pipeline.data_views import NumericDataView

data = NumericDataView(
    X_train=X_train,
    y_train=y_train,
    X_valid=X_valid,
    y_valid=y_valid,
    feature_names=("x0", "x1"),
    target_name="price",
)
```

约束：

```text
X_train: 2D numeric array
y_train: 1D numeric vector
X_valid/y_valid: 要么都给，要么都不给
feature_names: 长度必须等于 X_train columns
```

其他数据视图：

| data view | 用途 |
| --- | --- |
| `ImageDataView` | CNN image classification，NCHW |
| `GraphDataView` | 小图分类 smoke |
| `PreferencePairDataView` | DPO/preference |
| `ImageContrastivePairDataView` | retrieval/contrastive |

## 3. Pipeline

Pipeline 只做数据和目标变换，不做优化。

```python
pipeline = build_pipeline({
    "components": [
        {"name": "select_columns", "params": {"columns": [0, 2, 4]}},
        {"name": "zscore"},
        {"name": "feature_space"},
    ]
})
```

### 3.1 标准数据变换

| 组件 | 场景 |
| --- | --- |
| `identity` | 占位或测试 |
| `select_columns` | 手动选择特征 |
| `zscore` | 稳定梯度和距离度量 |
| `feature_space` | 记录特征语义，供 artifact/report 审计 |
| `conditional_primitives` | gate、hinge、one-hot 等条件特征 |

### 3.2 模型条件化目标

```python
from mlblack.pipeline import ModelConditionedTargetComponent

residual_data = ModelConditionedTargetComponent().build(
    data,
    reference_model=main_model,
)
```

可表达：

| 模式 | 公式 | 场景 |
| --- | --- | --- |
| residual | `y_next = y - model.predict(X)` | 残差/boosting |
| identity target + prediction feature | `X_next = [X, pred]` | stacking |
| prediction target | `y_next = pred` | distillation / teacher signal |

这个组件不会训练 reference model；reference model 由前一阶段训练得到，前后阶段由 `nsgablack` 编排。

## 4. Representation / Codec / Head

### 4.1 线性

```text
UnknownState(theta)
  -> LinearPointCodec
  -> LinearPointModel.predict(X)
```

适合：baseline、残差修正、小样本可解释模型。

### 4.2 正交线性

```text
X -> OrthogonalFeatureMap.transform(X) -> Q
UnknownState(theta) -> intercept + Q @ weights
```

适合：你关心特征解耦、稳定性和可解释 residual 的场景。

### 4.3 Estimator spec

```text
UnknownState
  -> EstimatorSpecModel(family, route, params, mechanisms)
  -> Problem fits estimator
  -> FittedEstimatorModel artifact
```

适合：tree、random forest、xgboost、sklearn MLP 这类外部 estimator。

### 4.4 Neural graph

```text
NeuralGraphSpec
  -> NeuralGraphCodec
  -> backend lowering
  -> executable model
```

当前路线：

| route | backend | task |
| --- | --- | --- |
| MLP | numpy/jax/tensorflow/torch surface | regression smoke |
| tiny Transformer | torch | classification / LM / DPO |
| tiny CNN | torch | image classification / retrieval |
| tiny GNN | torch | graph classification |

### 4.5 Symbolic

```text
Expression genome
  -> SymbolicExpressionCodec
  -> SymbolicExpressionModel
  -> symbolic problem / gradient / artifact
```

适合：表达式搜索、函数池、正交 basis、truth recovery。

### 4.6 Head

| head | 输出 | problem |
| --- | --- | --- |
| point | `y_hat` | regression |
| interval | `lower, upper` | coverage/width |
| binary logistic | `p(y=1)` | binary classification |
| softmax | class probabilities | multiclass |
| piecewise | branch-conditioned model | routed problem |
| symbolic basis | expression set | orthogonality / basis task |
| LM / preference / embedding | neural graph heads | LM/DPO/retrieval |

## 5. Problem

Problem 是唯一稳定吃数据的位置。

```python
feedback = problem.evaluate(model, state, context)
```

`Feedback` 字段：

| 字段 | 含义 |
| --- | --- |
| `objectives` | 优化目标，越小越好 |
| `constraints` | 约束违反量 |
| `loss` | 单标量训练损失 |
| `gradients` | adapter 可用梯度 |
| `residuals` | residual/dynamic pool/symbolic signal |
| `metrics` | 可读指标 |
| `signals` | 附加结构化信号 |

常见 problem：

| problem | 用途 |
| --- | --- |
| `SupervisedRegressionProblem` | 点回归 |
| `SupervisedEstimatorFitRegressionProblem` | 拟合 decoded estimator spec |
| `SupervisedClassificationProblem` | 分类/概率 |
| `TinyTransformerLanguageModelProblem` | LM next-token |
| `TinyTransformerDPOPreferenceProblem` | DPO/preference |
| `FixedSymbolicRegressionProblem` | 固定表达式参数拟合 |
| `OrthogonalBasisEvaluationProblem` | basis 正交性评估 |

## 6. Adapter

Adapter 是优化策略面。

| adapter | 消费 | 更新 | 注意 |
| --- | --- | --- | --- |
| `GradientDescentAdapter` | `feedback.gradients` | flat state | 不读数据 |
| `RandomSearchAdapter` | objective score | population | black-box |
| `EstimatorSpecSearchAdapter` | fitted estimator feedback | estimator spec state | 外部 estimator |
| `TorchBackpropAdapter` | torch loss/backward | flat state + optimizer state | torch route |
| `NeuralGraphBackpropAdapter` | backend neural graph loss | graph parameters | torch neural graph |
| `FunctionalBackpropAdapter` | problem-owned functional gradient | flat state | jax/tensorflow style |

错误边界：

```text
adapter 直接读 X/y
adapter 保存 artifact
adapter 决定多阶段顺序
adapter 分配 GPU lease
```

## 7. Backend

Backend 是执行系统，不是编排系统。

| backend | 擅长 | 不应伪装 |
| --- | --- | --- |
| numpy | CPU、无梯度、轻量 smoke | autograd / GPU |
| jax | functional gradient、JIT 潜力 | torch-style module backward |
| tensorflow | GradientTape functional gradient | torch optimizer.step |
| torch | stateful module、backward、optimizer | outer resource scheduler |

组件通过 `backend_requires` 声明能力。Trainer setup 会 fail-fast。

## 8. Bias

Bias 是软偏好。

| bias | 作用 |
| --- | --- |
| `ObjectiveWeightBias` | reweight objectives |
| `StateL2Bias` | 参数 L2 soft preference |
| `L2ScaleBias` | scale-aware L2 |
| `ObjectivePolicyBias` | context-aware objective policy |
| `BranchPolicyBias` | branch preference |
| `DynamicPoolBias` | symbolic/dynamic pool hint |

硬约束必须进 Problem/constraints，不能用 Bias 偷偷替代。

## 9. Capability

Capability 处理生命周期副作用：

| capability | 作用 |
| --- | --- |
| `CheckpointCapability` | state snapshot |
| `ExperimentTrackerCapability` | SQLite/experiment records |
| `ResourceAuditCapability` | audit injected ResourceContext |

Capability 不改变 candidate 选择方向。如果要影响优化，写 Adapter 或 Bias。

## 10. Artifact

Artifact 是产品边界。

| artifact | 场景 |
| --- | --- |
| `ModelArtifact` | 通用模型 |
| `IntegratedModelArtifact` | 多模型组合产物 |
| `NeuralGraphArtifact` | neural graph spec/layout/audit |
| `TreeEnsembleArtifact` | tree/forest |
| `XGBoostArtifact` | xgboost |
| `SymbolicModelArtifact` | symbolic expression |
| `TrainerStateArtifact` | resume/replay |
| `RunReport` | metrics/components/resources |

## 11. Context contract

组件源码应声明：

```python
context_requires = ("candidate.model", "data.X_train", "data.y_train")
context_optional = ("data.X_valid", "data.y_valid")
context_provides = ("feedback.objectives", "feedback.metrics")
context_mutates = ()
context_cache = ()
requires_metrics = ()
metrics_fallback = "strict"
```

Doctor/catalog 会读取这些字段。不要新增第二套强制 key 系统。

## 12. 配置判断表

| 你想做 | 应放位置 |
| --- | --- |
| 新模型结构 | representation / codec / neural graph spec |
| 新输出类型 | head / model wrapper |
| 新 loss/metric | problem / backend losses |
| 新参数更新策略 | adapter |
| 新 tensor/autograd 系统 | backend |
| 新数据/target 变换 | pipeline |
| 新模型融合方式 | `models.composition` |
| 新 artifact schema | core artifacts / catalog viewer |
| 多阶段训练 | nsgablack case |
| 并行调度/资源 lease | nsgablack L0 |
