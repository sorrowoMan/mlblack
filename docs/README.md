# mlblack 文档入口

这组文档按职责分成四类。阅读时优先按当前目标选择，不需要从头顺序读完。

## 架构主线

| 文档 | 用途 |
| --- | --- |
| [standard_scaffold_tutorial/](standard_scaffold_tutorial/README.md) | 标准脚手架、assembly、case、artifact、跨框架协调。 |
| [neural_graph_backend_architecture/](neural_graph_backend_architecture/README.md) | NeuralGraph、codec、backend、capability contract、多后端边界。 |
| [symbolic_learning_migration_inventory.md](symbolic_learning_migration_inventory.md) | 符号学习迁移盘点和策略归属。 |

## 学习资料

| 文档 | 用途 |
| --- | --- |
| [transformer_learning/](transformer_learning/README.md) | Transformer 机制学习，以及如何映射到 NeuralGraph。 |
| [ml_terminology_review/](ml_terminology_review/README.md) | ML/LLM 面试常见术语对照和复习。 |

## 当前核心结论

```text
mlblack 是 nsgablack 的 ML 特化层，不自建 workflow/runtime/L0。

Codec / Representation:
  定义模型语义和 unknown state 如何解码。

Backend:
  定义 tensor / lowering / autograd / optimizer / loss / artifact 的执行能力。

Trainer/L0:
  指定本次 run 使用哪个 compute backend。

Capability contract:
  判断当前 backend 是否能满足组件需求；不能满足就 fail-fast。
```

当前已注册 compute backend：

```text
numpy:
  CPU ndarray / MLP / MSE / artifact summary

jax:
  JAX array / MLP / functional gradient / SGD helper / artifact summary

tensorflow:
  TensorFlow tensor / MLP / GradientTape functional gradient / SGD helper / artifact summary

torch:
  torch Tensor / Transformer-CNN-GNN / backward / optimizer / neural artifact audit
```
