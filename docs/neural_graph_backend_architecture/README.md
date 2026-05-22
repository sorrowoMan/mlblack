# NeuralGraph Backend 架构

这组文档记录当前 `mlblack` 神经网络路线的稳定边界。重点不是解释 Transformer 细节，而是说明：

```text
NeuralGraphSpec / Codec / Representation
  和
Compute Backend / Capability Contract
```

到底怎么分工。

## 推荐阅读顺序

1. [01_layer_boundaries.md](01_layer_boundaries.md)：先明确 codec、backend、problem、adapter、trainer 的边界。
2. [02_capability_contracts.md](02_capability_contracts.md)：理解 capability key、catalog、fail-fast。
3. [03_backend_matrix.md](03_backend_matrix.md)：查看 numpy / jax / tensorflow / torch 当前能力矩阵。
4. [04_add_backend_guide.md](04_add_backend_guide.md)：后续新增更多 backend 时按这个流程做。

## 一句话结论

```text
Codec 定义模型是什么。
Backend 定义模型怎么执行。
Trainer/L0 指定本次运行用哪个 backend。
Capability contract 判断这个 backend 能不能跑。
```

## 当前统一链路

```text
Trainer.compute_backend_session
  -> context["backend.session"]
  -> backend capability preflight
  -> NeuralGraphRepresentation.setup(...)
  -> NeuralGraphCodec.parameter_layout(context)
  -> NeuralGraphCodec.init_values(context)
  -> NeuralGraphCodec.decode(values, context)
  -> LearningProblem.evaluate(...) or compute_backend_loss(...)
  -> Adapter.update(...)
  -> ArtifactBuilder / backend artifacts
```

关键规则：

- `NeuralGraphCodec.__init__` 不选择 backend。
- `NeuralGraphRepresentationConfig` 不保存 backend。
- `Problem` 不私自创建 backend。
- `Adapter` 不绕过 `backend.session`。
- 当前 backend 缺 capability 时直接报错，不静默换成 torch。

## 当前 backend

| backend | 当前定位 | 主要用途 |
| --- | --- | --- |
| `numpy` | CPU ndarray 简单后端 | MLP lowering、MSE、artifact summary。 |
| `jax` | 函数式参数后端 | MLP lowering、`jax.grad`、FunctionalBackpropAdapter。 |
| `tensorflow` | GradientTape 后端 | MLP lowering、GradientTape functional grad、FunctionalBackpropAdapter。 |
| `torch` | 完整神经网络训练后端 | Transformer/CNN/GNN、backward、optimizer、LM/classification/DPO/retrieval。 |

## 和 Transformer 学习文档的关系

Transformer 学习文档负责解释机制：

```text
tokenizer -> embedding -> attention -> FFN -> residual/norm -> head -> loss
```

本组文档负责解释工程接入：

```text
spec -> codec -> backend lowering -> problem -> adapter -> artifact
```
