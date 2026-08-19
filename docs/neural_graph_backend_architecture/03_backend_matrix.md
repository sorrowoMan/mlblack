# 03. Backend Matrix

> 下面保留旧 Adapter 矩阵用于兼容性判断。Torch 神经预设的规范入口已经是
> `gradient.* -> GradientOptimizerAdapter -> TorchEvaluationProvider`；JAX/TensorFlow
> functional 路线尚未迁入同一 EvaluationProvider 状态协议。

## 总览

| backend | tensor | lowering | loss | gradient | optimizer | artifact | 当前定位 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `numpy` | `np.ndarray` | MLP | MSE | 无 backend autograd | 无 | 参数摘要 | 最小 CPU 后端 |
| `jax` | `jax.Array` | MLP | MSE | functional grad | SGD helper | 参数摘要 | 函数式参数后端 |
| `tensorflow` | `tf.Tensor` | MLP | MSE | GradientTape functional grad | SGD helper | 参数摘要 | TensorFlow GradientTape 后端 |
| `torch` | `torch.Tensor` | MLP/Tiny Transformer/CNN/GNN | CE/LM/DPO/triplet | backward | AdamW 等 | audit + 参数摘要 | 完整 neural graph 后端 |

## numpy backend

目录：

```text
mlblack/backends/numpy_neural/
  backend.py
  capabilities/
    tensor.py
    neural_lowering.py
    losses.py
    artifacts.py
```

支持：

```text
tensor.from_numpy
tensor.float_tensor
tensor.device
tensor.to_device
neural.lowering.mlp
parameters.layout
parameters.init
loss.mse
metrics.regression
artifact.parameters.summary
artifact.numpy_model.describe
```

不支持：

```text
autograd.backward
autograd.functional.grad
optimizer.step
optimizer.sgd_step
neural.lowering.transformer
neural.lowering.cnn
neural.lowering.gnn
GPU
```

适合：

```text
MLP decode smoke
CPU-only lightweight prediction
artifact/contract examples
```

不适合：

```text
Transformer training
CNN/GNN training
backend-native gradient training
```

## jax backend

目录：

```text
mlblack/backends/jax_neural/
  backend.py
  capabilities/
    tensor.py
    neural_lowering.py
    losses.py
    autograd.py
    optimizers.py
    artifacts.py
```

支持：

```text
tensor.from_numpy
tensor.float_tensor
tensor.device
tensor.to_device
neural.lowering.mlp
parameters.layout
parameters.init
loss.mse
metrics.regression
autograd.functional.grad
autograd.value_and_grad
autograd.gradients.flat_export
optimizer.sgd_step
artifact.parameters.summary
artifact.jax_model.describe
```

不支持：

```text
autograd.backward
optimizer.build
optimizer.step
optimizer.zero_grad
optimizer.state
neural.lowering.transformer
neural.lowering.cnn
neural.lowering.gnn
```

适合：

```text
函数式参数实验
JAX MLP gradient smoke
和 FunctionalGradientLearningProblem + GradientOptimizerAdapter 组合
后续扩展 jit / vmap / optax
```

当前已验证路线：

```text
NeuralGraphSpec.mlp
  -> NeuralGraphCodec with backend.session=jax
  -> JaxMLPPointModel
  -> Problem.compute_functional_gradient(...)
  -> backend.autograd.mse_parameter_gradient(...)
  -> FunctionalGradientLearningProblem emits Feedback.gradients
  -> GradientOptimizerAdapter updates flat state
```

关键说明：

```text
JAX backend 不伪装 torch-style backward。
如果 Problem/Provider 请求 torch-style backward，JAX 会因缺 autograd.backward / optimizer.step 报错。
```

## tensorflow backend

目录：

```text
mlblack/backends/tensorflow_neural/
  backend.py
  capabilities/
    tensor.py
    neural_lowering.py
    losses.py
    autograd.py
    optimizers.py
    artifacts.py
```

支持：

```text
tensor.from_numpy
tensor.float_tensor
tensor.device
tensor.to_device
neural.lowering.mlp
parameters.layout
parameters.init
loss.mse
metrics.regression
autograd.functional.grad
autograd.value_and_grad
autograd.gradients.flat_export
optimizer.sgd_step
artifact.parameters.summary
artifact.tensorflow_model.describe
```

不支持：

```text
autograd.backward
optimizer.build
optimizer.step
optimizer.zero_grad
optimizer.state
neural.lowering.transformer
neural.lowering.cnn
neural.lowering.gnn
```

适合：

```text
GradientTape 风格验证
TensorFlow MLP functional gradient smoke
和 FunctionalGradientLearningProblem + GradientOptimizerAdapter 组合
```

## torch backend

目录：

```text
mlblack/backends/torch_neural/
  backend.py
  transformer.py
  vision.py
  graph.py
  capabilities/
    tensor.py
    neural_lowering.py
    autograd.py
    optimizers.py
    losses.py
    artifacts.py
```

支持：

```text
tensor.from_numpy
tensor.token_ids
tensor.class_labels
tensor.float_tensor
tensor.device
tensor.to_device
neural.lowering
neural.lowering.transformer
neural.lowering.cnn
neural.lowering.gnn
parameters.layout
parameters.init
parameters.flat_import
autograd.mode.train
autograd.mode.eval
autograd.no_grad
autograd.backward
autograd.zero_grad
autograd.gradients.flat_export
parameters.flat_export
parameters.summary
parameters.state_to_cpu
parameters.state_to_device
parameters.state_json
optimizer.build
optimizer.step
optimizer.zero_grad
optimizer.state
loss.cross_entropy
loss.lm_next_token
loss.dpo
loss.triplet
metrics.classification
artifact.neural_graph.audit
artifact.parameters.summary
artifact.torch_model.describe
```

适合：

```text
Tiny Transformer classification / LM / DPO
Tiny CNN image classification / retrieval
Tiny GNN graph classification
GradientOptimizerAdapter + TorchEvaluationProvider
attention/FFN audit artifact
```

## 当前 route 支持

| route | numpy | jax | tensorflow | torch |
| --- | --- | --- | --- | --- |
| MLP | yes | yes | yes | local MLP path / backend optional |
| Tiny Transformer | no | no | no | yes |
| Tiny CNN | no | no | no | yes |
| Tiny GNN | no | no | no | yes |
| LoRA tiny transformer | no | no | no | yes |
| QLoRA surface | no | no | no | yes tiny surface |

## 当前统一策略匹配

| adapter | numpy | jax | tensorflow | torch |
| --- | --- | --- | --- | --- |
| `nsgablack GradientOptimizerAdapter` + ML Problem/Provider | 解析梯度 | functional gradient | functional gradient（需安装 TensorFlow） | TorchEvaluationProvider |
| `nsgablack GaussianSearchAdapter` | yes | yes | yes | yes |
| `nsgablack FixedCandidateAdapter` | yes | yes | yes | yes |

## 快速检查命令

```powershell
python -c "from mlblack.backends import list_backends; print(list_backends())"
```

预期：

```text
('jax', 'numpy', 'tensorflow', 'torch')
```

```powershell
python -c "from mlblack.backends import explain_backend_requirements; print(explain_backend_requirements('jax', ('autograd.backward','optimizer.step')))"
```

预期：

```text
ok = False
missing = ('autograd.backward', 'optimizer.step')
```
