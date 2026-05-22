# 02. Capability Contracts

## 为什么不用一个全局大接口

不同执行系统的能力不等价：

```text
numpy:
  ndarray, CPU, no autograd graph

jax:
  functional arrays, grad/value_and_grad, JIT/XLA potential

torch:
  stateful modules, loss.backward(), optimizer.step(), CUDA modules
```

如果强行要求所有 backend 实现一个巨大接口，会出现两种坏结果：

```text
1. 简单 backend 被迫伪造能力。
2. 复杂 backend 的特殊能力被抹平。
```

所以当前采用：

```text
小 capability key + backend catalog + fail-fast
```

## Capability component

一个 backend 由多个 capability component 组成：

```text
mlblack/backends/<backend_name>/
  backend.py
  capabilities/
    tensor.py
    neural_lowering.py
    losses.py
    autograd.py
    optimizers.py
    artifacts.py
```

每个 component 暴露一个 `BackendCapabilityContract`：

```text
backend
capability
provides
methods
tensor_kinds
model_kinds
routes
heads
supports_autograd
supports_stateful_module
supports_functional_params
supports_gpu
supports_resume
notes
```

## 常用 capability key

Tensor：

```text
tensor
tensor.from_numpy
tensor.token_ids
tensor.class_labels
tensor.float_tensor
tensor.device
tensor.to_device
```

Lowering：

```text
neural.lowering
neural.lowering.mlp
neural.lowering.transformer
neural.lowering.cnn
neural.lowering.gnn
parameters.layout
parameters.init
parameters.flat_import
```

Autograd：

```text
autograd.backward
autograd.zero_grad
autograd.gradients.flat_export
autograd.functional.grad
autograd.value_and_grad
```

Optimizer：

```text
optimizer.build
optimizer.step
optimizer.zero_grad
optimizer.state
optimizer.sgd_step
```

Loss / metrics：

```text
loss.mse
loss.cross_entropy
loss.lm_next_token
loss.dpo
loss.triplet
metrics.regression
metrics.classification
```

Artifact：

```text
artifact.parameters.summary
artifact.neural_graph.audit
artifact.numpy_model.describe
artifact.jax_model.describe
artifact.tensorflow_model.describe
artifact.torch_model.describe
```

## 组件如何声明需求

Representation：

```python
backend_requires = ("parameters.layout", "parameters.init", "neural.lowering")
```

Torch neural backprop adapter：

```python
backend_requires = (
    "autograd.mode.train",
    "autograd.backward",
    "autograd.zero_grad",
    "autograd.gradients.flat_export",
    "optimizer.build",
    "optimizer.step",
    "parameters.flat_export",
    "parameters.state_json",
)
```

JAX functional route 不应该声明 `autograd.backward`：

```text
它应该声明或消费:
  autograd.functional.grad
  optimizer.sgd_step

而不是伪装:
  autograd.backward
  optimizer.step
```

## 查询 backend 能力

```python
from mlblack.backends import (
    list_backends,
    list_backend_capabilities,
    explain_backend_requirements,
)

list_backends()
list_backend_capabilities("torch")

explain_backend_requirements(
    "jax",
    ("neural.lowering.mlp", "autograd.functional.grad", "optimizer.sgd_step"),
)

explain_backend_requirements(
    "jax",
    ("autograd.backward", "optimizer.step"),
)
```

预期语义：

```text
jax + functional grad:
  ok = True

jax + torch-style backward:
  ok = False
  missing = ("autograd.backward", "optimizer.step")
```

## Contract 的作用

contract 不是文档装饰，而是执行边界：

```text
Trainer.setup:
  收集 representation/problem/adapter backend_requires
  调用 compute_backend_session.ensure(...)
  缺能力就 fail-fast

Codec/Problem/Adapter:
  从 context["backend.session"] 获取 backend
  不允许 ad-hoc backend.name fallback
```

这保证：

```text
配置写的是 numpy，就不会偷偷变成 torch。
配置写的是 jax，就不会偷偷调用 torch backward。
配置写的是 torch，就可以使用完整 torch neural graph route。
```

## 正确错误是有价值的

例如：

```text
compute_backend="numpy"
adapter=NeuralGraphBackpropAdapter
```

应报错：

```text
backend 'numpy' is missing required capabilities:
autograd.backward, optimizer.step, ...
```

这说明系统没有隐式降级，契约有效。
