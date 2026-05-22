# 04. NeuralGraph Decoder 当前架构

这篇现在只保留 Transformer 学习路线下最重要的架构结论。完整 backend / capability / 多后端细节已经拆到：

```text
docs/neural_graph_backend_architecture/
```

推荐配套阅读：

- [../neural_graph_backend_architecture/01_layer_boundaries.md](../neural_graph_backend_architecture/01_layer_boundaries.md)
- [../neural_graph_backend_architecture/02_capability_contracts.md](../neural_graph_backend_architecture/02_capability_contracts.md)
- [../neural_graph_backend_architecture/03_backend_matrix.md](../neural_graph_backend_architecture/03_backend_matrix.md)
- [../neural_graph_backend_architecture/04_add_backend_guide.md](../neural_graph_backend_architecture/04_add_backend_guide.md)

## 1. 核心判断

```text
神经网络不是优化器。
神经网络是可参数化的数据变换图。
Transformer 是 NeuralGraph 的一个 preset。
```

所以它在 `mlblack` 里的位置不是新 workflow，而是：

```text
NeuralGraphSpec
  -> NeuralGraphCodec
  -> NeuralGraphRepresentation
  -> LearningProblem
  -> OptimizerAdapter
  -> Artifact
```

## 2. 层级映射

| Transformer 概念 | mlblack 层级 | 说明 |
| --- | --- | --- |
| token embedding / attention / FFN / norm / residual | `NeuralGraphSpec` + backend lowering | 模型结构机制。 |
| 参数向量 / 权重表 | `UnknownState` | 被优化对象。 |
| 参数 layout / decode | `NeuralGraphCodec` | 把 flat state 解释进模型结构。 |
| classification / LM / embedding / preference head | head spec / problem head | 输出语义。 |
| cross entropy / LM next token / DPO / triplet | `LearningProblem` + backend losses | 评价方式。 |
| backward / optimizer.step | `Adapter` + backend autograd/optimizer | 参数更新。 |
| attention map / FFN activation / parameter summary | backend artifacts | 审计与产物。 |

## 3. 当前实现路线

```text
Trainer.compute_backend_session
  -> context["backend.session"]
  -> capability preflight
  -> NeuralGraphRepresentation.setup
  -> NeuralGraphCodec.parameter_layout(context)
  -> NeuralGraphCodec.init_values(context)
  -> NeuralGraphCodec.decode(values, context)
  -> Problem.evaluate / compute_backend_loss
  -> Adapter.update
  -> ArtifactBuilder
```

关键边界：

```text
Codec:
  定义 unknown state 如何变成模型。
  不私自选择 torch/jax/tensorflow/numpy。

Backend:
  定义 tensor/lowering/loss/autograd/optimizer/artifact 怎么执行。

Trainer:
  指定本次 run 使用哪个 backend。

Contract:
  判断当前 backend 是否满足组件要求。
```

## 4. 当前 backend 支持

| backend | 当前定位 | 支持路线 |
| --- | --- | --- |
| `numpy` | CPU ndarray 最小后端 | MLP lowering、MSE、artifact summary。 |
| `jax` | 函数式参数后端 | MLP lowering、functional gradient、SGD helper。 |
| `tensorflow` | GradientTape 后端 | MLP lowering、GradientTape functional gradient、SGD helper。 |
| `torch` | 完整 neural graph 后端 | Tiny Transformer/CNN/GNN、backward、optimizer、classification/LM/DPO/retrieval、audit artifact。 |

这说明当前框架已经不是“torch 专用设计”。但是也不能说所有 backend 能力等价：

```text
numpy:
  不支持 autograd.backward / optimizer.step。

jax:
  支持 functional grad，不支持 torch-style backward。

tensorflow:
  支持 GradientTape functional grad，不支持 torch-style backward。

torch:
  支持 stateful module backward 和 optimizer.step。
```

缺能力时必须报错，不允许静默换后端。

## 5. 当前 NeuralGraph preset

已覆盖：

```text
MLP:
  numpy / jax / local numpy route

Tiny Transformer:
  torch route
  classification
  language modeling
  generation / KV cache surface
  DPO preference
  LoRA / QLoRA tiny surface
  RMSNorm / RoPE / SwiGLU

Tiny CNN:
  torch route
  image classification
  contrastive / retrieval

Tiny GNN:
  torch route
  graph classification
```

## 6. 和 nsgablack 的关系

`mlblack` 不拥有外层编排：

```text
nsgablack:
  outer solver
  group / serial
  parallel scheduling
  L0 resource allocation
  outer structure search

mlblack:
  inner model representation
  codec / backend-facing lowering
  problem evaluation
  inner parameter fitting
  artifact/report
```

神经网络结构搜索应是：

```text
nsgablack outer:
  搜 NeuralGraphSpec 字段

mlblack inner:
  固定 spec
  初始化参数
  训练/评估
  返回 loss / complexity / audit metrics
```

这和符号学习 nested search 是同构的：

```text
outer searches structure
inner fits parameters
problem returns multi-objective feedback
```

## 7. 不再放在这篇里的内容

以下内容已经拆走：

```text
backend capability 细节
numpy/jax/torch 能力矩阵
新增 backend 指南
fail-fast contract 细节
```

请看：

```text
docs/neural_graph_backend_architecture/
```

## 8. 当前验收口径

基础验证：

```powershell
python -m pytest -q tests
python -c "from mlblack.project import run_project_doctor, format_doctor_report; print(format_doctor_report(run_project_doctor('.', strict=True)))"
```

神经图 smoke：

```powershell
python examples\cases\tiny_transformer_smoke\run_case.py --steps 1
python examples\benchmarks\neural_graph_benchmark_matrix.py --steps 1
```

backend 查询：

```powershell
python -c "from mlblack.backends import list_backends; print(list_backends())"
```

预期：

```text
('jax', 'numpy', 'tensorflow', 'torch')
```

## 9. 下一步方向

如果继续增强，优先级应该是：

```text
1. JAX MLP/CNN 更完整 lowering
   保持 functional style，不伪装 torch module。

2. TensorFlow CNN/Transformer route
   当前 TensorFlow 已验证 MLP + GradientTape functional style，后续再扩 route。

3. 更完整 benchmark/dashboard
   展示 backend capability、缺失能力、适配路线、artifact 和 experiment query。
```

不建议：

```text
在 Codec 里重新加入 backend 参数。
在 Problem 里直接 get_backend("torch")。
为了兼容让 JAX 声明 autograd.backward。
在 mlblack 里自建 workflow/runtime/L0。
```
