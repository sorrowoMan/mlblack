# 01. Transformer 核心机制

## 1. Transformer 是什么

Transformer 是一种神经网络结构，最早用于序列到序列任务，后来成为现代大语言模型的基础。

一句话定义：

```text
Transformer = token embedding + attention block 堆叠 + task head
```

它不是训练算法。训练算法通常是 AdamW、SGD、LoRA trainer、backprop 等。Transformer 是被训练的模型结构。

用 `mlblack` 的语言说：

```text
Transformer 是一种 representation/decoder/model family。
```

用统一优化语言说：

```text
unknown parameters theta
  -> Transformer architecture decoder
  -> sequence function f_theta(tokens)
  -> problem evaluates loss
  -> adapter updates theta
```

## 2. 为什么需要 Transformer

序列任务的核心困难是：一个位置的信息经常依赖另一个很远的位置。

例子：

```text
The animal didn't cross the street because it was tired.
```

`it` 指的是 `animal` 还是 `street`？模型需要在句子里建立远距离关系。

旧的 RNN/LSTM 是顺序读：

```text
token1 -> token2 -> token3 -> token4
```

Transformer 让所有 token 可以直接互相看：

```text
token1 <-> token2 <-> token3 <-> token4
```

这就是 self-attention 的基本价值。

## 3. Transformer 的数据流

以文本为例：

```text
raw text
  -> tokenizer
  -> token ids
  -> token embeddings
  -> positional information
  -> Transformer block 1
  -> Transformer block 2
  -> ...
  -> hidden states
  -> output head
  -> output
```

每一层都在把 token 表示变得更“上下文化”。

```text
初始 embedding：这个 token 自己的向量
中间 hidden state：这个 token 结合上下文后的向量
最终 hidden state：适合当前任务的表示
```

## 4. Tokenizer

Tokenizer 把文本切成模型能处理的整数 id。

```text
"hello world"
  -> [15339, 1917]
```

关键点：

| 概念 | 含义 |
| --- | --- |
| token | 文本片段，不一定是一个词 |
| vocabulary | 所有 token 的字典 |
| token id | token 在字典里的编号 |
| context length | 一次最多处理多少 token |

Tokenizer 不是 Transformer block 的一部分，但它决定输入空间。

在框架里它属于：

```text
pipeline / numericizer / feature encoder
```

## 5. Embedding

Embedding 把 token id 映射成向量。

```text
token id 15339
  -> embedding vector [0.12, -0.03, ...]
```

这是查表操作：

```text
EmbeddingMatrix[vocab_size, hidden_dim]
```

每个 token 对应一行向量。

在框架里它属于：

```text
representation / model parameter space
```

## 6. Position 信息

Attention 本身不天然知道顺序。下面两个序列如果没有 position 信息，attention 很难区分：

```text
A B C
C B A
```

所以需要 position encoding 或 position embedding。

常见形式：

| 类型 | 含义 |
| --- | --- |
| absolute position embedding | 第 0/1/2 个位置有自己的向量 |
| sinusoidal position encoding | 用 sin/cos 函数编码位置 |
| relative position bias | 关注两个 token 的相对距离 |
| RoPE | rotary positional embedding，现代 LLM 常用 |

在框架里它属于 Transformer 机制参数：

```text
mechanisms.position = absolute / sinusoidal / rope / relative_bias
```

## 7. Self-Attention

Self-attention 的作用：让每个 token 根据当前输入动态选择应该看哪些 token。

输入：

```text
X: [seq_len, hidden_dim]
```

通过三个线性映射得到：

```text
Q = X Wq
K = X Wk
V = X Wv
```

含义：

| 符号 | 直觉 |
| --- | --- |
| Q / Query | 我正在找什么信息 |
| K / Key | 我能提供什么匹配标签 |
| V / Value | 我真正携带的信息 |

计算：

```text
scores = Q K^T / sqrt(d)
weights = softmax(scores)
output = weights V
```

直觉：

```text
每个 token 用 Q 去问所有 token 的 K：谁和我相关？
然后用相关性权重聚合那些 token 的 V。
```

## 8. Causal Mask

GPT 这类 decoder-only 语言模型不能偷看未来 token。

训练目标是：

```text
给定 token_0...token_t，预测 token_{t+1}
```

所以 attention 需要 mask：

```text
token 3 可以看 token 0,1,2,3
但不能看 token 4,5,6
```

这叫 causal attention。

不同结构：

| 模型 | attention 方式 |
| --- | --- |
| BERT | 双向 attention，可以看左右上下文 |
| GPT | causal attention，只能看过去 |
| T5 | encoder 双向，decoder causal，并可 cross-attend encoder |

## 9. Multi-Head Attention

单个 attention 只能学一种关系模式。Multi-head attention 让模型同时学习多种关系。

```text
head_1: 可能关注语法关系
head_2: 可能关注实体关系
head_3: 可能关注位置关系
head_4: 可能关注局部短语
```

计算上就是把 hidden_dim 分成多个 head：

```text
hidden_dim = num_heads * head_dim
```

每个 head 独立做 attention，最后 concat 回来。

这和你关心的正交性有强连接：理想情况下，不同 head 应该捕捉不同信息，而不是全部重复。

## 10. Feed-Forward Network / MLP

Attention 负责 token 之间交换信息。FFN/MLP 负责对每个 token 自己做非线性变换。

```text
hidden -> linear up -> activation -> linear down -> hidden
```

常见激活：

| 激活 | 说明 |
| --- | --- |
| ReLU | 简单非线性 |
| GELU | Transformer 常用 |
| SwiGLU | 现代 LLM 常用变体 |

很多现代 LLM 的参数大头其实在 FFN 层。

## 11. Residual Connection

Residual 是：

```text
x_new = x + block(x)
```

作用：

```text
让深层网络更容易训练
保留原始信息
缓解梯度消失
```

没有 residual，几十层上百层的 Transformer 很难稳定训练。

## 12. LayerNorm

LayerNorm 稳定每层的数值分布。

常见结构：

```text
Pre-LN:
  x -> norm -> attention -> residual
  x -> norm -> ffn -> residual
```

现代大模型多用 Pre-LN 或 RMSNorm 变体。

框架里它属于 mechanism：

```text
mechanisms.normalization = layer_norm / rms_norm
```

## 13. Transformer Block

一个标准 decoder-only block：

```text
x
  -> norm
  -> causal self-attention
  -> residual add
  -> norm
  -> FFN / MLP
  -> residual add
  -> output
```

堆叠 N 层：

```text
block_1 -> block_2 -> ... -> block_N
```

## 14. Output Head

不同任务换不同 head。

| head | 输出 | problem |
| --- | --- | --- |
| language modeling head | next-token logits | cross entropy |
| classification head | class logits | classification loss |
| embedding head | vector | contrastive/retrieval loss |
| ranking head | score | pairwise/listwise loss |
| regression head | number | MSE/MAE |

这和我们当前 `mlblack` 的 head 概念完全一致：head 是输出语义，不是 trainer。

## 15. GPT / BERT / T5 / LLM 的关系

| 名词 | 本质 |
| --- | --- |
| Transformer | 基础结构 |
| BERT | encoder-only Transformer |
| GPT | decoder-only Transformer |
| T5 | encoder-decoder Transformer |
| LLM | 很大的 Transformer 语言模型 |
| ChatGPT | LLM + 对齐训练 + 工具/产品系统 |

不要把它们混成一个东西。

## 16. 训练目标

### Language Modeling

```text
输入：今天 天气 很
目标：好
```

模型学习预测下一个 token。

loss：

```text
cross entropy(next_token_distribution, true_next_token)
```

### Masked Language Modeling

BERT 用：

```text
今天 [MASK] 很好
```

预测被 mask 的 token。

### Classification

```text
输入文本 -> class label
```

### Contrastive / Embedding

```text
相似文本向量更近
不相似文本向量更远
```

## 17. 从零训练为什么不适合当前单机

从零训练 LLM 需要：

```text
海量 token
大量 GPU
长时间训练
复杂分布式系统
稳定数据管线
评估和安全对齐
```

当前更合理的路线是：

```text
先学机制
再调用现成模型
再做 prompt/RAG/agent 优化
最后有条件再做 LoRA/QLoRA
```

## 18. 最小总结

Transformer 的核心不是“很大的模型”，而是下面这个循环：

```text
token 表示
  -> attention 建立 token 间关系
  -> MLP 做非线性变换
  -> residual/norm 保持稳定
  -> 多层堆叠形成强表示
  -> head 产出任务输出
```

用框架话说：

```text
Transformer = 一种复杂 representation/decoder + 多种 head + 可微 problem + backprop adapter
```
