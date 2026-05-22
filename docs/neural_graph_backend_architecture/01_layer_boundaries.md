# 01. Layer Boundaries

## 核心边界

| 层 | 代表对象 | 负责什么 | 不负责什么 |
| --- | --- | --- | --- |
| Structure semantics | `NeuralGraphSpec` | 描述模型结构、block、head、参数化方式。 | 不执行 tensor 运算。 |
| Codec / Decoder | `NeuralGraphCodec` | 把 flat unknown state 和 spec 解码成模型对象。 | 不决定使用 torch/jax/tf。 |
| Representation | `NeuralGraphRepresentation` | 对接 Trainer 的 `init/repair/decode/describe`。 | 不自己选择 backend。 |
| Problem | `LearningProblem` | 吃数据，调用模型，返回 loss/metrics/feedback。 | 不拥有优化算法，不私自建 backend。 |
| Adapter | `OptimizerAdapter` | 根据 feedback 或 backend-native loss 更新 unknown state。 | 不直接吃业务数据。 |
| Backend | `mlblack.backends.*` | tensor、lowering、loss、autograd、optimizer、artifact。 | 不定义搜索语义，不做 Trainer 编排。 |
| Trainer/L0 context | `Trainer.compute_backend_session` | 本次 run 的 compute backend 选择和能力预检。 | 不实现具体 tensor kernel。 |
| nsgablack outer | solver/group/serial/resource | 外层结构搜索、并行、资源授权、阶段编排。 | 不写死 mlblack 内部 trainer 后端细节。 |

## 正确依赖方向

```text
Trainer
  owns compute_backend_session

Representation / Problem / Adapter
  declare backend_requires
  consume context["backend.session"]

Backend
  exposes capability contracts
  implements execution details

Codec
  only dispatches through current backend session
```

禁止反向依赖：

```text
Codec -> hardcoded torch
Problem -> get_backend("torch")
Adapter -> direct torch import outside backend capability
RepresentationConfig -> backend="torch"
```

## Codec 和 Backend 的区别

`Codec` 回答：

```text
这个 unknown state 如何解释成一个模型？
这个 NeuralGraphSpec 的参数 layout 是什么？
这个 flat vector 怎么塞进模型结构？
```

`Backend` 回答：

```text
参数 layout 怎么按本后端实现？
模型怎么 lower 成可执行对象？
loss 怎么算？
梯度怎么拿？
optimizer 怎么更新？
artifact 怎么审计？
```

同一个 spec 可以通过不同 backend 执行：

```text
NeuralGraphSpec.mlp(...)
  -> numpy backend -> NumpyMLPPointModel
  -> jax backend   -> JaxMLPPointModel
```

但不是所有 backend 都必须支持所有 spec：

```text
NeuralGraphSpec.tiny_transformer(...)
  -> torch backend 支持
  -> numpy backend 不支持
  -> jax backend 当前不支持
```

## Problem 和 Adapter 的分工

`Problem.evaluate(...)` 是稳定 no-backward 评估路径：

```text
model + state + context
  -> Feedback(objectives, metrics, gradients optional)
```

对于 torch neural graph，backward 不放在 `evaluate()` 里：

```text
Problem.compute_backend_loss(...)
  -> backend-native loss object
  -> NeuralGraphBackpropAdapter calls backend.autograd.backward(...)
```

原因是：

```text
Problem 负责“这个模型好不好”。
Adapter 负责“怎么更新参数”。
```

对于 JAX / TensorFlow MLP 当前路线：

```text
Problem.compute_functional_gradient(...)
  -> backend.autograd.functional.grad
  -> FunctionalBackpropAdapter updates flat state
```

这说明函数式后端走 functional grad，不伪装成 torch-style backward。

## Setup 生命周期

当前神经图 backend 解析发生在 Trainer setup：

```text
ComposableTrainer.setup()
  -> collect backend_requires
  -> require_compute_backend(...)
  -> representation.setup(trainer, context)
  -> adapter.setup(trainer)
```

`NeuralGraphRepresentation.setup(...)` 会：

```text
context["backend.session"]
  -> codec.parameter_layout(context)
  -> self.layout / self.dimension / self.base_dimension
```

这样 dimension 是当前 backend session 的结果，不是构造期偷算出来的结果。

## Fail-fast 规则

如果某组件需要：

```text
autograd.backward
optimizer.step
loss.cross_entropy
```

但当前 backend 是 `jax` 或 `numpy`，setup 或调用点必须直接报错。

这不是缺点，是架构边界正确：

```text
backend 能力不等价。
contract 必须明确暴露差异。
不能通过隐式 fallback 掩盖差异。
```
