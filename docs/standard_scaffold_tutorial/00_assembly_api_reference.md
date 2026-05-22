# 00. 架构地图与装配 API 速查

这一章只做两件事：固定架构边界，列出标准装配 API。后续章节所有例子都按这里的口径展开。

## 1. 分层地图

```text
NumericDataView / DataPipeline
  -> ModelRepresentation / Codec / Head
  -> LearningProblem
  -> OptimizerAdapter
  -> Trainer
  -> Capability / Artifact / Report
```

外层复杂编排不在这条链里，它属于 `nsgablack`：

```text
nsgablack Solver / Adapter / Representation / Plugin / L0 Resource
  -> calls mlblack inner trainer / problem / proxy / artifact surface
```

## 2. 核心对象职责

| 对象 | 负责 | 不负责 |
| --- | --- | --- |
| `NumericDataView` | 稳定数据视图 | 优化、调度、artifact |
| `DataPipeline` | 数据转换、特征处理、目标转换 | 搜索候选、分配资源 |
| `ModelRepresentation` | init/repair/encode/decode unknown state | 计算 objective |
| `Codec` | 参数向量与模型/spec/表达式互转 | 读取训练数据 |
| `Head` | 输出语义，如 point/interval/probability | 训练循环 |
| `LearningProblem` | 消费数据和模型，返回 `Feedback` | 调度多个 trainer |
| `OptimizerAdapter` | propose/update 参数或候选 | 直接读数据、保存 artifact |
| `Trainer` | 单 inner trainer 生命周期 | workflow、parallel、resource lease |
| `Capability` | checkpoint/tracking/resource audit/report | 改变优化语义 |
| `Artifact` | 可复现产物 | 临时 context 大对象 |
| `IntegratedPredictionModel` | 合并已训练模型预测 | 决定训练顺序 |

## 3. 标准单 trainer API

| API | 用法 | 说明 |
| --- | --- | --- |
| `build_pipeline(spec)` | `pipeline = build_pipeline({...})` | 构造数据 pipeline |
| `pipeline.fit_transform(data)` | `prepared = pipeline.fit_transform(raw)` | 输出新 `NumericDataView` |
| `build_trainer(spec, data)` | `trainer = build_trainer(spec, data)` | 构造一个 inner trainer |
| `trainer.fit(max_steps=n)` | `result = trainer.fit(max_steps=50)` | 单 trainer 优化 |
| `trainer.evaluate_individual(state, ctx)` | 返回 `Feedback` | 单候选评估 |
| `trainer.evaluate_population(states, ctx)` | 返回 feedback list | 当前是简单顺序语义，不负责外层并行 |
| `trainer.write_snapshot(payload, context_key=...)` | 写大对象 ref | context 只放轻量引用 |
| `trainer.build_report()` | 输出 run report | 包含 component contract/resource/adapter state |
| `ArtifactBuilder().build(trainer, result)` | 输出 bundle | model artifact + trainer state + run report |

## 4. TrainerAssemblySpec

`build_trainer` 接收 dict 或 `TrainerAssemblySpec`：

```python
spec = {
    "preset": "orthogonal_linear_point",
    "run_name": "demo_inner",
    "params": {
        "learning_rate": 0.05,
        "l2": 0.001,
    },
    "resource_context": {
        "device": "cpu",
        "threads": 1,
        "namespace": "manual.demo",
    },
    "capabilities": [
        "resource_audit",
        {"name": "checkpoint", "params": {"interval": 5}},
    ],
    "biases": [
        {"name": "state_l2", "params": {"weight": 0.001}},
    ],
    "metadata": {
        "case": "tutorial",
    },
}
```

字段解释：

| 字段 | 归属 | 说明 |
| --- | --- | --- |
| `preset` | mlblack assembly | 单 trainer 预置组合 |
| `params` | preset builder | 学习率、模型维度、搜索规模等 |
| `run_name` | trainer report | run id |
| `resource_context` | 外层注入 | `mlblack` 被动读取和审计 |
| `capabilities` | lifecycle side effect | checkpoint、tracking、audit |
| `biases` | soft preference | objective weight、L2、policy |
| `metadata` | audit | 不参与优化语义 |

禁止字段：

```text
workflow
flow
runtime
orchestration
resource_request
parallel_backend
stage_group
```

这些字段意味着你在 `mlblack` 里创建第二套编排系统。应迁到 `nsgablack` outer config。

## 5. 当前 preset 速查

| preset | 类型 | adapter | problem | 适用 |
| --- | --- | --- | --- | --- |
| `orthogonal_linear_point` | 正交线性回归 | gradient descent | supervised regression | 快速 baseline |
| `orthogonal_linear_interval` | 区间输出 | random search | interval regression | coverage/width |
| `orthogonal_logistic_classification` | 二分类概率 | random search | classification | binary probability |
| `orthogonal_softmax_classification` | 多分类概率 | random search | classification | multiclass |
| `tree_estimator_search` | tree estimator spec | estimator search | fit regression | 树结构/参数搜索 |
| `tree_boosting_estimator_search` | boosting/xgboost spec | estimator search | fit regression | boosting/xgboost |
| `numpy_mlp_torch_backprop` | MLP 参数向量 | torch backprop | regression | 小神经网络 |
| `sklearn_mlp_estimator_search` | sklearn MLP spec | estimator search | fit regression | 外部 estimator |
| `tiny_transformer_classification` | tiny Transformer | neural graph backprop | classification | Transformer smoke |
| `tiny_transformer_lm` | tiny Transformer LM | neural graph backprop | LM problem | next-token smoke |
| `tiny_transformer_dpo` | tiny Transformer preference | neural graph backprop | DPO problem | preference smoke |
| `tiny_cnn_image_classification` | CNN | neural graph backprop | image classification | vision smoke |
| `tiny_gnn_graph_classification` | GNN | neural graph backprop | graph classification | graph smoke |
| `tiny_cnn_image_contrastive` | CNN retrieval | neural graph backprop | triplet/contrastive | retrieval smoke |

## 6. Pipeline API

```python
from mlblack.assembly import build_pipeline

pipeline = build_pipeline({
    "name": "tabular_prep",
    "components": [
        {"name": "select_columns", "params": {"columns": [0, 1]}},
        {"name": "zscore"},
        {"name": "feature_space", "params": {"tags": ["baseline"]}},
    ],
})
prepared = pipeline.fit_transform(data)
```

常用组件：

| component | 输入 | 输出 | 用途 |
| --- | --- | --- | --- |
| `identity` | `NumericDataView` | 原数据 | 占位 |
| `zscore` | numeric X | 标准化 X | 稳定训练 |
| `select_columns` | numeric X | 子列 | 特征选择 |
| `feature_space` | data metadata | feature metadata | 审计 |
| `conditional_primitives` | data | 派生条件特征 | branch/piecewise |
| `ModelConditionedTargetComponent` | data + reference model | 新 target / 可选新 feature | residual/stacking |

## 7. 组合模型 API

```python
from mlblack.models import (
    PredictionIOContract,
    PredictionInputSpec,
    PredictionIntegrationComponent,
)

io_contract = PredictionIOContract.by_component({
    "tabular": PredictionInputSpec(key="tabular", ndim=2, n_features=16),
    "image": PredictionInputSpec(key="image", ndim=4),
})

model = PredictionIntegrationComponent.additive(
    component_order=("tabular", "image"),
    weights={"tabular": 0.7, "image": 0.3},
    io_contract=io_contract,
).compose({
    "tabular": tabular_model,
    "image": image_model,
})

pred = model.predict({"tabular": X_tabular, "image": X_image})
```

`IntegratedPredictionModel` 是最终模型语义，不训练 component，也不决定 stage 顺序。

## 8. Backend API

单 trainer 通过 `compute_backend` 指定后端：

```python
from mlblack.core import ComputeBackendSpec, Trainer

trainer = Trainer(
    problem=problem,
    representation=representation,
    adapter=adapter,
    compute_backend=ComputeBackendSpec(name="torch", device="cpu"),
)
```

后端 capability 必须满足组件需求。不满足要报错：

```text
FunctionalBackpropAdapter requires:
  autograd.functional.grad
  autograd.gradients.flat_export
  optimizer.sgd_step

Torch neural graph route requires:
  autograd.backward
  optimizer.step
  parameters.flat_export
```

## 9. nsgablack-facing surface

`mlblack` 可以暴露给 `nsgablack` 调用的 surface：

```text
OuterProblem bridge
TrainingProxy
search-space adapter
artifact builder
audit producer
model/data transform component
```

但不要暴露：

```text
WorkflowRunner
StageScheduler
ParallelRuntime
ResourceAllocator
```

## 10. 最小检查

```powershell
python -m pytest -q tests
python -c "from mlblack.project import run_project_doctor, format_doctor_report; print(format_doctor_report(run_project_doctor('.', strict=True)))"
```
