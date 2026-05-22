# Transformer 学习笔记

这组文档不是标准脚手架教程，也不是 `mlblack` API 文档。它的目标是把 Transformer 作为一个机器学习机制拆开学习，然后再判断它怎样接入 `nsgablack + mlblack` 的统一优化架构。

适合你的原因很明确：你现在已经把 ML 理解成“unknown state 经过 decoder 变成模型，再由 problem/evaluation 返回反馈”的优化过程。Transformer 正好可以按这个视角拆解：

```text
architecture spec + parameters theta
  -> Transformer decoder / model builder
  -> sequence function
  -> problem.evaluate(...)
  -> loss / gradients / metrics
  -> adapter.update(...)
```

所以你不需要先把它当成神秘的 LLM。先把它看成一种复杂但很规则的“可参数化函数空间”。

## 推荐阅读顺序

1. [01_transformer_core.md](01_transformer_core.md)：Transformer 到底是什么，token、embedding、attention、block、head 分别负责什么。
2. [02_attention_and_orthogonality.md](02_attention_and_orthogonality.md)：attention 和你关心的正交性有什么相似和不同。
3. [03_framework_integration_notes.md](03_framework_integration_notes.md)：如果以后接入框架，应该放在哪些层，不应该重复实现什么。
4. [04_neural_graph_decoder_design.md](04_neural_graph_decoder_design.md)：NeuralGraph decoder 当前稳定架构，以及 Transformer/CNN/GNN/MLP 的统一口径。
5. [05_mlblack_neural_decoupling_summary.md](05_mlblack_neural_decoupling_summary.md)：mlblack 神经网络解耦现状、Transformer 解耦点和后续补强项。
6. [06_ml_model_family_integration_guide.md](06_ml_model_family_integration_guide.md)：常见机器学习模型族怎么接入 mlblack，每类组件如何设计。
7. [../neural_graph_backend_architecture/](../neural_graph_backend_architecture/README.md)：如果要看 backend、capability contract、多后端矩阵和新增后端指南，读这一组。

## 学习目标

学完这组文档，你应该能回答：

- Transformer 是模型结构，不是训练算法，这句话是什么意思。
- Attention 为什么本质上是关系矩阵和动态信息路由。
- Q/K/V、multi-head、residual、layer norm、FFN、head 各自解决什么问题。
- GPT、BERT、T5、LLM 和 Transformer 的关系。
- Transformer 和正交 basis / dynamic basis selection 的关系在哪里。
- 在 `mlblack` 里，Transformer 应该是 representation/codec/head/problem/adapter 的组合，而不是新框架。
- 在 `nsgablack` 里，可以搜索 Transformer 机制、prompt/RAG/agent policy，而不是从零训练大模型。

## 建议学习方式

不要一开始就追大模型训练。推荐路线：

```text
第一层：看懂 forward pass
第二层：看懂 attention 的矩阵计算
第三层：看懂 block 为什么这样堆
第四层：看懂不同 head 对应不同 problem
第五层：再考虑训练、LoRA、RAG、agent
```

你当前最值得学的是：

```text
Transformer 机制拆解 + attention 和正交/表示空间的关系
```

暂时不必优先学：

```text
从零预训练 LLM
大规模分布式训练
复杂 serving infra
```

这些对当前单机和当前框架阶段不是主线。

## 最小心智模型

```text
文本
  -> tokenizer
  -> token ids
  -> embedding vectors
  -> Transformer blocks
  -> hidden states
  -> output head
  -> task output
```

如果是语言模型：

```text
hidden states
  -> language modeling head
  -> next-token probability
```

如果是分类：

```text
hidden states
  -> classification head
  -> class probability
```

如果是 embedding：

```text
hidden states
  -> pooling / embedding head
  -> vector representation
```

## 和当前框架的最短映射

```text
TransformerRepresentation:
  定义 architecture spec、参数形状、decode 逻辑

TransformerHead:
  language modeling / classification / embedding / ranking

TransformerProblem:
  next-token loss / classification loss / retrieval metric / preference metric

TransformerAdapter:
  AdamW / backprop / LoRA / QLoRA

TransformerArtifact:
  config、weights ref、LoRA delta、token trace、attention summary、eval report

nsgablack outer:
  搜 architecture mechanism、head choice、LoRA rank、prompt/RAG/tool policy
```

## 当前工程边界

现在文档口径已经收敛为：

```text
Transformer 机制学习:
  仍放在 transformer_learning/

NeuralGraph / backend / capability 架构:
  放在 neural_graph_backend_architecture/
```

当前已注册 backend：

```text
numpy:
  CPU ndarray / MLP / MSE

jax:
  JAX array / MLP / functional gradient

torch:
  Transformer-CNN-GNN / backward / optimizer / neural artifact audit
```

