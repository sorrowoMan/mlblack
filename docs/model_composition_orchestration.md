# Model Composition And Orchestration Boundary

这份文档记录当前结论：复杂机器学习结构不应该在 `mlblack` 里新增 workflow/runtime，而应该拆成可组合的 ML 语义组件，再交给 `nsgablack` 做阶段、组、串行和资源编排。

## 1. 核心判断

```text
复杂模型 = 多个已训练模型 + 一个模型整合语义
多阶段训练 = 外层编排调用多个 inner trainer
残差学习 = 下一阶段 target transform 的一种策略
多模态融合 = 多个 component model 各自吃自己的输入，然后在整合层融合输出
```

不要新增：

```text
HybridTrainer
ResidualWorkflow
MultiModalWorkflow
mlblack SerialRuntime
```

应该新增和复用：

```text
IntegratedPredictionModel
PredictionIntegrationComponent
PredictionIOContract
ModelConditionedTargetComponent
```

## 2. 为什么需要 I/O contract

模型整合不能假设所有 component 都吃同一个 `X`。

典型场景：

```text
残差模型:
  main_model.predict(X_numeric)
  residual_model.predict(X_numeric)

stacking:
  stage2_model.predict([X_numeric, stage1_prediction])

多模态:
  text_model.predict(input_ids)
  image_model.predict(image_tensor)
  tabular_model.predict(tabular_features)

主线 + 修正器:
  main_model.predict(global_features)
  correction_model.predict(local_features)
```

所以整合层必须显式声明：

```text
每个 component 从输入 mapping 的哪个 key 取数据
该输入应该是什么 kind / ndim / feature count
component prediction 应该是什么输出形态
不同 component prediction 是否必须 row-aligned
最终 integration 如何组合 prediction
```

## 3. 当前 I/O contract

位置：

```text
mlblack.models.composition
```

核心对象：

```text
PredictionInputSpec:
  key: 从输入 mapping 中取哪个字段
  kind: numeric_array / array / tensor_like / any
  ndim: 输入维度要求，例如 tabular=2, image=4
  n_features: 二维 tabular 的特征数要求
  required: 缺失时是否报错

PredictionOutputSpec:
  kind: point_vector
  requires_aligned_rows: component 输出行数是否必须一致

PredictionIOContract:
  component_inputs: 每个 component 的输入要求
  shared_input_key: mapping 中的共享输入 key，默认 shared
  output: 统一输出要求
```

当前 additive/mean integration 消费的是 `point_vector`：

```text
允许输出:
  shape = (n,)
  shape = (n, 1)

拒绝输出:
  shape = (n, k), k > 1
  未对齐 row count
```

## 4. 同输入残差模型

```python
from mlblack.models import PredictionIntegrationComponent
from mlblack.pipeline import ModelConditionedTargetComponent

residual_data = ModelConditionedTargetComponent().build(
    data,
    reference_model=main_model,
)

residual_result = residual_trainer.fit(max_steps=120)

final_model = PredictionIntegrationComponent.additive(
    component_order=("main", "residual"),
).compose(
    {"main": main_model, "residual": residual_result.best_model},
)

# 非 mapping 输入会作为 shared input 传给每个 component
prediction = final_model.predict(X_numeric)
```

## 5. 不同输入多分支模型

```python
from mlblack.models import (
    PredictionIOContract,
    PredictionInputSpec,
    PredictionIntegrationComponent,
)

io_contract = PredictionIOContract.by_component(
    {
        "tabular": PredictionInputSpec(key="tabular", ndim=2, n_features=16),
        "image": PredictionInputSpec(key="image", ndim=4),
        "text": PredictionInputSpec(key="input_ids", ndim=2),
    }
)

final_model = PredictionIntegrationComponent.additive(
    component_order=("tabular", "image", "text"),
    weights={"tabular": 0.4, "image": 0.3, "text": 0.3},
    io_contract=io_contract,
).compose(
    {
        "tabular": tabular_model,
        "image": image_model,
        "text": text_model,
    }
)

prediction = final_model.predict(
    {
        "tabular": X_tabular,
        "image": X_image,
        "input_ids": X_tokens,
    }
)
```

这个模型只定义 inference/evaluation 的组合语义。哪个分支先训练、并行训练还是串行训练、用什么资源，仍然归 `nsgablack` 编排。

## 6. ModelConditionedTargetComponent

位置：

```text
mlblack.pipeline.model_conditioning
```

职责：

```text
读取 NumericDataView
调用 reference_model.predict(X)
生成下一阶段训练用的 y
可选把 reference prediction 追加成新 feature
```

残差场景：

```text
y_next = y - main_model.predict(X)
```

stacking 场景：

```text
X_next = [X, main_model.predict(X)]
y_next = y
```

它不是 trainer，也不是 workflow。它只是数据/目标变换组件。

## 7. IntegratedPredictionModel

位置：

```text
mlblack.models.composition
```

职责：

```text
保存多个已训练 component models
按 PredictionIOContract 路由输入
调用每个 component.predict(...)
校验输出 shape
按 PredictionIntegrationSpec 组合输出
```

当前支持的 integration：

```text
additive / sum / residual_sum
mean / average
```

后续可以扩展：

```text
learned linear fusion
router/gated fusion
rank fusion
probability calibration fusion
multi-output fusion
```

扩展时仍然只新增 integration spec/model 语义，不新增 mlblack workflow。

## 8. 当前验收

对应测试：

```powershell
python -m pytest -q tests\test_model_integration.py
```

覆盖：

```text
ModelConditionedTargetComponent 生成 residual target
普通 mlblack trainer 训练 residual target
IntegratedPredictionModel 合成 main + residual
IntegratedPredictionModel 路由不同 component 输入
输入 key / ndim / n_features / output shape contract 失败时 fail-fast
ArtifactBuilder 输出 integrated_model artifact
```
