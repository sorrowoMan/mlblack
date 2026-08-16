# 03. Transformer 如何接入 nsgablack + mlblack

这篇不是实现计划，而是架构定位。目标是先把 Transformer 拆成可以接入现有框架的组件，避免一上来写出第二套 LLM workflow。

## 1. 第一原则

Transformer 不改变主线。

主线仍然是：

```text
unknown state / config
  -> representation.decode
  -> model / task object
  -> problem.evaluate
  -> feedback
  -> adapter.update
  -> artifact/report
```

Transformer 只是一个更复杂的 model family。

## 2. Transformer 在 mlblack 里是什么

最合理拆法：

| mlblack 层 | Transformer 对应物 |
| --- | --- |
| `pipeline` | tokenizer、text numericizer、dataset packing |
| `representation` | Transformer architecture spec / parameter layout |
| `codec` | theta / LoRA delta / adapter weights 的 encode/decode |
| `head` | LM head、classification head、embedding head、ranking head |
| `problem` | next-token loss、classification loss、retrieval loss、preference loss |
| `adapter` | AdamW/backprop、LoRA、QLoRA、frozen evaluation |
| `capability` | checkpoint、attention trace、resource audit、eval tracker |
| `artifact` | config、weights ref、LoRA delta、token trace、metrics |

## 3. Transformer 在 nsgablack 里是什么

`nsgablack` 不训练每个 token 的参数。它负责外层结构和配置搜索。

适合外层搜索的对象：

| 搜索对象 | 例子 |
| --- | --- |
| architecture | layers、heads、hidden dim、FFN ratio |
| attention mechanism | full/local/sparse/linear attention |
| regularization | dropout、orthogonality weight、head diversity |
| LoRA config | rank、alpha、target modules |
| RAG config | chunk size、top-k、retriever、reranker |
| prompt config | template、few-shot、system prompt |
| agent policy | tool order、stop rule、retry policy |

统一形式：

```text
nsgablack outer candidate
  -> TransformerSpec / PromptSpec / RAGSpec
  -> mlblack inner evaluation
  -> objectives: quality, cost, latency, memory, stability
```

## 4. 不要自建什么

不要在 `mlblack` 里新增：

```text
LLMPrivateFlow
AgentPrivateRunner
TransformerRuntimeBackend
GPULeaseAllocator
PromptStageRunner
```

这些属于共享 Project substrate 的编排、runtime、resource 语义；如果需要搜索语义，再由 `nsgablack` Case 提供。

`mlblack` 只应该暴露：

```text
TransformerRepresentation
TransformerProblem
TransformerAdapter
TransformerArtifact
nsgablack-facing problem/proxy
```

## 5. 从最小可实现面开始

如果后面要实现，推荐顺序：

### Step 1: Frozen Transformer evaluator

不训练，只调用现成模型。

```text
TextInput
  -> tokenizer
  -> frozen model
  -> output
  -> problem metric
```

适合：

```text
classification evaluation
embedding evaluation
prompt evaluation
RAG evaluation
```

组件：

```text
TransformerExternalModel
TransformerEvaluationProblem
TransformerArtifact
```

### Step 2: Prompt/RAG optimizer

不训练 LLM 参数，优化使用方式。

```text
nsgablack candidate
  -> prompt/RAG config
  -> call LLM / local model
  -> evaluate answer
```

目标：

```text
accuracy
faithfulness
cost
latency
format validity
```

这是当前最实际的方向。

### Step 3: Small Transformer training

用小模型/小数据验证机制。

```text
TransformerRepresentation
  + LMHead / ClassificationHead
  + TorchBackpropAdapter
```

只做小模型，不做 LLM 预训练。

### Step 4: LoRA / QLoRA adapter

冻结 base model，只训练低秩 delta。

```text
base_model_ref
  + LoRADeltaRepresentation
  + LoRAAdapter
  + task problem
```

这是有 GPU 后的训练路线。

### Step 5: Outer architecture search

用 `nsgablack` 搜机制组合：

```text
num_heads
head_dim
attention_type
orthogonality_weight
LoRA rank
```

内层 `mlblack` 只负责固定 spec 下的训练或评估。

## 6. TransformerSpec 草图

可以先设计 spec，不急着实现完整训练。

```python
TransformerSpec = {
    "architecture": {
        "kind": "decoder_only",
        "num_layers": 4,
        "hidden_dim": 256,
        "num_heads": 4,
        "ffn_ratio": 4.0,
    },
    "attention": {
        "kind": "causal_self_attention",
        "position": "rope",
        "dropout": 0.0,
    },
    "normalization": "rms_norm",
    "head": {
        "kind": "classification",
        "num_classes": 2,
    },
    "regularization": {
        "attention_head_orthogonality": 0.01,
    },
}
```

这就是“机制拆解”，比 `family='transformer'` 清楚。

## 7. Representation 设计草图

```text
TransformerRepresentation
  context_requires:
    candidate.unknown_state
    transformer.spec

  context_provides:
    candidate.model
    model.forward
    model.attention_hooks

  decode:
    unknown state -> model weights / LoRA delta / config-bound model
```

如果是 frozen model：

```text
unknown state 不一定是 weights
可能是 prompt config / routing config / adapter config
```

所以要分清：

| 类型 | unknown state 表示什么 |
| --- | --- |
| full training | 全部模型参数 |
| LoRA | 低秩 adapter 参数 |
| prompt search | prompt template/config |
| RAG search | retrieval config |
| architecture search | TransformerSpec |

## 8. Head 设计草图

| head | 输出 | problem |
| --- | --- | --- |
| `LanguageModelHead` | next-token logits | cross entropy |
| `ClassificationHead` | class logits/probability | CE/AUC/F1 |
| `EmbeddingHead` | vector | retrieval/contrastive |
| `RankingHead` | score | pairwise/listwise ranking |
| `PreferenceHead` | chosen/rejected score | preference loss |
| `AttentionAuditHead` | attention summaries | report/regularizer |

Head 仍然是输出语义，不是 trainer。

## 9. Problem 设计草图

```text
TransformerLanguageModelProblem
  reads data.token_ids
  evaluates next-token loss
  provides feedback.loss / gradients / perplexity

TransformerClassificationProblem
  reads text/token data + labels
  evaluates CE/AUC/F1

TransformerRetrievalProblem
  reads query/document pairs
  evaluates recall@k / MRR / nDCG

TransformerPromptProblem
  calls external LLM
  evaluates task metric / cost / latency
```

## 10. Adapter 设计草图

| adapter | 作用 | 需要资源 |
| --- | --- | --- |
| `FrozenTransformerEvalAdapter` | 不训练，只评估候选配置 | CPU/API/local inference |
| `TorchTransformerBackpropAdapter` | 小模型 backprop | GPU 可选 |
| `LoRAAdapter` | 只训练 LoRA delta | GPU 推荐 |
| `PromptSearchAdapter` | 可作为 nsgablack outer adapter，不放 mlblack 主干 | 无需梯度 |

注意：Prompt/RAG/agent policy 搜索更像 `nsgablack` outer adapter/problem，不是 `mlblack` inner adapter。

## 11. Artifact 设计草图

```text
TransformerArtifact
  model_config
  base_model_ref
  tokenizer_ref
  head_kind
  trained_delta_ref
  metrics
  resource_context
  attention_summary
  eval_examples
```

LoRA artifact：

```text
LoRAArtifact
  base_model_ref
  target_modules
  rank
  alpha
  delta_weights_ref
  task_metrics
```

Prompt/RAG artifact：

```text
LLMUsageArtifact
  prompt_template
  rag_config
  tool_plan
  model_name
  outputs
  judge_scores
  cost
  latency
```

## 12. 和符号正交嵌套的同构关系

符号嵌套：

```text
nsgablack outer searches symbolic structure
mlblack inner fits constants
problem returns RMSE/orthogonality/complexity
```

Transformer 机制搜索：

```text
nsgablack outer searches TransformerSpec / regularization / LoRA config
mlblack inner fits parameters or evaluates frozen model
problem returns loss/cost/latency/diversity
```

LLM prompt/RAG 搜索：

```text
nsgablack outer searches prompt/RAG/tool config
mlblack inner calls external model and evaluates outputs
problem returns quality/cost/faithfulness
```

结构完全一致。

## 13. 适不适合你学

适合，但学习顺序要控制。

你不适合一开始学：

```text
大规模 LLM 预训练工程
分布式并行细节
CUDA kernel 优化
serving infra
```

你非常适合先学：

```text
attention 的关系矩阵
multi-head 和正交/多样性
Transformer block 的机制分层
head/problem 的对应关系
LoRA 为什么是低秩参数优化
prompt/RAG 为什么是黑箱优化
```

因为这些和你已经建立的优化框架视角高度一致。

## 14. 如果要进入实现，建议先做哪个

建议先做一个很小的文档驱动 prototype，而不是直接接大模型：

```text
mini_transformer_classification_demo
```

范围：

```text
小 synthetic token dataset
小 Transformer encoder/decoder
classification head
TorchBackpropAdapter
attention head diversity metric
artifact report
```

这能验证：

```text
TransformerRepresentation 是否合理
Head/Problem 是否清楚
attention orthogonality metric 是否能接入 feedback
```

之后再做：

```text
prompt/RAG optimizer case
LoRA adapter
LLM external model bridge
```

## 15. 最终原则

Transformer 接入时不要问：

```text
我要不要做一个 Transformer 框架？
```

应该问：

```text
这个机制属于 representation、head、problem、adapter、artifact，还是 nsgablack outer orchestration？
```

只要这个问题分清，Transformer 就能自然接进现有架构。
