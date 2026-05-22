# 03. 复杂模型组合与 I/O Contract

这一章是当前教程的核心增强：`mlblack` 现在可以严谨表达多模型组合，但不把组合训练顺序写成新的 workflow。组合模型负责 inference/evaluation 语义；训练顺序、并行和资源仍由 `nsgablack` 外层编排。

## 1. 核心原则

```text
复杂模型不是复杂 trainer。
复杂模型 = 多个 component model + 显式 I/O contract + prediction integration spec。
```

不要写：

```text
HybridTrainer
ResidualWorkflow
MultiModalTrainer
StackingRuntime
BoostingFlow
```

应该写：

```text
ModelConditionedTargetComponent:
  用已训练模型生成下一阶段数据/目标。

IntegratedPredictionModel:
  用显式 I/O contract 路由输入，并组合多个已训练模型的输出。

nsgablack stage/group/serial:
  决定这些 component model 如何训练、何时训练、并行还是串行。
```

## 2. 两个核心组件

### 2.1 ModelConditionedTargetComponent

位置：

```text
mlblack.pipeline.model_conditioning
```

用途：训练下一阶段前，调用已经训练好的模型。

残差：

```text
y_next = y - main_model.predict(X)
```

stacking：

```text
X_next = [X, main_model.predict(X)]
y_next = y
```

distillation：

```text
y_next = teacher_model.predict(X)
```

### 2.2 IntegratedPredictionModel

位置：

```text
mlblack.models.composition
```

用途：把多个已训练模型组合成最终模型。

```text
final.predict(inputs)
  -> route each component input
  -> component.predict(component_input)
  -> validate output shape
  -> integrate predictions
```

当前支持：

```text
additive / sum / residual_sum
mean / average
```

后续可以扩展：learned linear fusion、gated fusion、router fusion、probability calibration fusion、rank fusion。

## 3. I/O Contract 为什么必要

不能假设所有 component 都吃同一个 `X`。

| 场景 | component 输入 |
| --- | --- |
| 残差 | main 和 residual 都吃同一个 numeric X |
| stacking | stage2 吃 `[X, stage1_pred]` |
| 多模态 | text 吃 token ids，image 吃 NCHW tensor，tabular 吃 numeric matrix |
| 主线 + 修正器 | main 吃全局特征，correction 吃局部特征 |
| 专家模型 | 每个 expert 吃自己的特征子集或模态 |

所以组合模型必须显式声明：

```text
component name -> input key
input kind
input ndim
input n_features
output kind
row alignment requirement
```

## 4. PredictionInputSpec

```python
from mlblack.models import PredictionInputSpec

PredictionInputSpec(
    key="tabular",
    kind="numeric_array",
    ndim=2,
    n_features=16,
    required=True,
)
```

字段：

| 字段 | 含义 |
| --- | --- |
| `key` | 从 `predict(inputs)` 的 mapping 里取哪个 key |
| `kind` | `numeric_array` / `array` / `tensor_like` / `any` |
| `ndim` | 维度要求，例如 tabular=2、image=4、tokens=2 |
| `n_features` | 2D numeric 输入的 feature 数 |
| `required` | 缺失时是否报错 |

## 5. PredictionOutputSpec

当前 integration 消费的是 point vector：

```text
允许:
  shape=(n,)
  shape=(n, 1)

拒绝:
  shape=(n, k), k>1
  row count 不一致
```

这保证 additive / mean 不会误把多维 logits、embedding 或 interval 当作 scalar prediction 直接相加。

## 6. 同输入残差模型

训练阶段：

```python
from mlblack.pipeline import ModelConditionedTargetComponent
from mlblack.presets import build_orthogonal_linear_point_trainer

# stage 1 已经由某个 trainer 得到 main_model
residual_data = ModelConditionedTargetComponent().build(
    data,
    reference_model=main_model,
)

residual_trainer = build_orthogonal_linear_point_trainer(
    residual_data,
    learning_rate=0.2,
    energy_threshold=None,
)
residual_result = residual_trainer.fit(max_steps=120)
```

整合阶段：

```python
from mlblack.models import PredictionIntegrationComponent

final_model = PredictionIntegrationComponent.additive(
    component_order=("main", "residual"),
).compose(
    {"main": main_model, "residual": residual_result.best_model},
)

prediction = final_model.predict(X_numeric)
```

解释：

```text
非 mapping 输入会作为 shared input 给每个 component。
final = main.predict(X) + residual.predict(X)
```

## 7. 不同输入多模态模型

```python
from mlblack.models import (
    PredictionIOContract,
    PredictionInputSpec,
    PredictionIntegrationComponent,
)

io_contract = PredictionIOContract.by_component({
    "tabular": PredictionInputSpec(key="tabular", ndim=2, n_features=12),
    "image": PredictionInputSpec(key="image", ndim=4),
    "text": PredictionInputSpec(key="input_ids", ndim=2),
})

final_model = PredictionIntegrationComponent.additive(
    component_order=("tabular", "image", "text"),
    weights={"tabular": 0.4, "image": 0.3, "text": 0.3},
    io_contract=io_contract,
).compose({
    "tabular": tabular_model,
    "image": image_model,
    "text": text_model,
})

prediction = final_model.predict({
    "tabular": X_tabular,
    "image": X_image,
    "input_ids": X_tokens,
})
```

这个模型不关心三个 component 是怎么训练出来的。可能是：

```text
tabular_model: orthogonal linear / tree / MLP
image_model: tiny CNN / pretrained wrapper
text_model: tiny Transformer / pretrained wrapper
```

训练顺序由 `nsgablack` 控制。

## 8. Stacking

Stage 1：训练 base model。

```python
base_result = base_trainer.fit(max_steps=50)
base_model = base_result.best_model
```

Stage 2：把 base prediction 追加成 feature。

```python
from mlblack.pipeline import ModelConditionedTargetComponent, ModelConditionedTargetConfig

stack_data = ModelConditionedTargetComponent(
    reference_model=base_model,
    config=ModelConditionedTargetConfig(
        mode="identity",
        reference_name="base",
        append_prediction_feature=True,
        prediction_feature_name="base_pred",
    ),
).build(data)

meta_trainer = build_orthogonal_linear_point_trainer(stack_data)
meta_result = meta_trainer.fit(max_steps=80)
```

最终模型有两种方式：

```text
方式 A:
  只使用 meta_model，并在推理前重复同样的 feature transform。

方式 B:
  写一个 composition wrapper，先调用 base_model 生成 base_pred，再调用 meta_model。
```

当前已有能力覆盖方式 A。方式 B 后续可作为 `SequentialPredictionModel` 扩展，仍然是 model semantic，不是 workflow。

## 9. Boosting-like 多轮残差

概念：

```text
model_0 fits y
model_1 fits y - model_0(X)
model_2 fits y - model_0(X) - model_1(X)
...
final = sum_i model_i
```

落层：

```text
nsgablack serial stages:
  stage i decides whether to train another residual learner and with what budget

mlblack:
  ModelConditionedTargetComponent builds residual target
  IntegratedPredictionModel additive-composes learned models
```

伪代码：

```python
components = {}
current_integrated = None
current_data = data

for i in range(num_rounds):
    trainer = build_trainer(stage_specs[i], current_data)
    result = trainer.fit(max_steps=stage_steps[i])
    components[f"round_{i}"] = result.best_model

    current_integrated = PredictionIntegrationComponent.additive(
        component_order=tuple(components),
    ).compose(components)

    current_data = ModelConditionedTargetComponent().build(
        data,
        reference_model=current_integrated,
    )
```

注意：这个循环如果是正式工程，不应在 `mlblack` 主干里做成 workflow；应在 `nsgablack` case 的 stage 编排里做。

## 10. 主线 + 修正器

```text
main_model:
  学全局趋势，例如低频、线性、物理主项。

correction_model:
  学局部误差，例如非线性、小区域、异常模式。

final:
  main + alpha * correction
```

```python
final_model = PredictionIntegrationComponent.additive(
    component_order=("main", "correction"),
    weights={"main": 1.0, "correction": 0.3},
).compose({
    "main": main_model,
    "correction": correction_model,
})
```

`alpha` 可以固定，也可以由 nsgablack outer search 搜索。

## 11. 专家模型 + Late Fusion

```text
expert_a: handles small x range
expert_b: handles large x range
expert_c: handles sparse or high-noise region
fusion: weighted mean or learned gate
```

当前可做：

```text
weighted additive / mean fusion
```

后续扩展：

```text
GatedIntegratedPredictionModel:
  gate_model.predict(X) -> weights per row
  final = sum_i gate_i(X) * expert_i(X)
```

这仍然属于 `mlblack.models.composition`，不是 workflow。

## 12. Contract fail-fast 示例

```python
io_contract = PredictionIOContract.by_component({
    "image": PredictionInputSpec(key="image", ndim=4),
})

model.predict({"image": X_2d})
# raises: component 'image' input must be 4D
```

```python
class BadModel:
    def predict(self, X):
        return np.ones((len(X), 3))

# additive expects point vector, so this fails.
```

Fail-fast 是必须的。组合模型如果默默广播或 reshape，会让 outer search 学到错误反馈。

## 13. Artifact

组合模型会输出 `integrated_model` artifact：

```python
from mlblack.core import ArtifactBuilder

bundle = ArtifactBuilder().build(trainer_like, result_like)
assert bundle.model_artifact.describe()["artifact_type"] == "integrated_model"
```

推荐 metadata：

```text
component model names
component artifact refs
integration kind
weights
I/O contract
orchestration_owner = nsgablack
source stage ids
```

## 14. 什么时候要新增新组件

| 需求 | 新增位置 |
| --- | --- |
| 新融合公式 | `PredictionIntegrationSpec` / composition model |
| 行级动态 gate | composition model + gate model |
| 多输出融合 | `PredictionOutputSpec` 扩展 |
| 顺序调用模型 | sequential model wrapper |
| 训练阶段编排 | nsgablack case，不进 mlblack |
| 多设备分配 | nsgablack L0，不进 mlblack |
