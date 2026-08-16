# mlblack 文档入口

`mlblack` 是 `nsgablack / mlblack` 统一框架栈中的机器学习语义层。

当前规则：

- `mlblack` 负责 ML 语义：DataView、Spec、Codec、Head、Problem、Trainer、Provider、Artifact、backend capability 和 ML report。
- `nsgablack` 负责优化搜索语义。
- Project / Case / Scaffold / L0 编排属于共享 substrate。
- `mlblack` Case 可以作为外层，也可以作为内层；它不应该创建私有编排或私有资源栈。

## 主要入口

| 文档 | 用途 |
| --- | --- |
| [standard_scaffold_tutorial/](standard_scaffold_tutorial/README.md) | 标准 Project / Case / Scaffold 教程 |
| [neural_graph_backend_architecture/](neural_graph_backend_architecture/README.md) | NeuralGraph、Codec、backend capability 边界 |
| [model_composition_orchestration.md](model_composition_orchestration.md) | 共享 substrate 下的模型组合边界 |
| [symbolic_learning_migration_inventory.md](symbolic_learning_migration_inventory.md) | 当前符号学习迁移地图 |
| [transformer_learning/](transformer_learning/README.md) | Transformer 与模型族学习笔记 |
| [ml_terminology_review/](ml_terminology_review/README.md) | ML 术语复盘 |

## 核心链路

```text
Data / Schema:
  Dataset schema, FeatureSpec, TargetSpec, DataView.

Codec / Representation:
  ML 语义状态如何 encode/decode。

Head / Problem:
  输出语义、目标、约束、metric。

Trainer / Provider:
  拟合、推理、backend capability、result payload。

Artifact:
  持久化模型、报告、lineage、部署 payload。

Case runtime:
  消费 Project L0 发放的 ResourceContext，并报告实际 backend。
```

Backend 选择必须来自 Case config 和注入的 `ResourceContext`，不能藏在 Trainer 私有逻辑里。
