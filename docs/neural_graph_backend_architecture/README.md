# NeuralGraph Backend 架构

这组文档记录 `mlblack` 神经图语义与后端执行之间的稳定边界。

当前规则：

> Codec 定义模型是什么。Backend capability 定义模型能怎么运行。Project L0 发放资源。Case 消费 `ResourceContext` 并报告实际 backend。

## 阅读顺序

1. [01_layer_boundaries.md](01_layer_boundaries.md)：Codec、backend、problem、adapter、trainer 的边界。
2. [02_capability_contracts.md](02_capability_contracts.md)：capability key、catalog entry、fail-fast 规则。
3. [03_backend_matrix.md](03_backend_matrix.md)：当前 numpy / jax / tensorflow / torch 能力矩阵。
4. [04_add_backend_guide.md](04_add_backend_guide.md)：新增 backend 时如何不长出私有 runtime。

## 稳定执行链

```text
Case builder 接收 ResourceContext
  -> Trainer 构造 compute_backend_session
  -> backend capability preflight
  -> NeuralGraphRepresentation.setup(...)
  -> NeuralGraphCodec.parameter_layout(context)
  -> NeuralGraphCodec.init_values(context)
  -> NeuralGraphCodec.decode(values, context)
  -> LearningProblem.evaluate(...) or backend loss
  -> Adapter.update(...)
  -> ArtifactBuilder / backend artifacts
```

## 边界规则

- `NeuralGraphCodec` 不选择全局 backend。
- `NeuralGraphRepresentationConfig` 不拥有资源。
- `Problem` 不创建私有 backend。
- `Adapter` 不绕过 `backend.session`。
- backend capability 不足时必须 fail-fast，不允许静默切到另一个 backend。
- 资源授权属于共享 Project / L0 substrate。

## 当前 backend 定位

| backend | 定位 |
| --- | --- |
| `numpy` | CPU ndarray、小型 MLP lowering、MSE、artifact summary |
| `jax` | functional gradient 路线、MLP lowering、optimizer helper |
| `tensorflow` | GradientTape functional gradient 路线 |
| `torch` | neural graph training 路线，Transformer/CNN/GNN、backward、optimizer、artifact audit |

Transformer 学习文档解释机制；本组文档解释工程接入路径：

```text
spec -> codec -> backend lowering -> problem -> adapter -> artifact
```
