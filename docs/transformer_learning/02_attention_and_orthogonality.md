# 02. Attention 和正交性的关系

这篇专门回答一个问题：Transformer 的 attention 为什么会让人联想到正交性？它们到底哪里一样，哪里不一样？

结论先写清楚：

```text
正交性是表示空间的几何目标。
Attention 是表示空间里的动态路由机制。
```

它们共享“内积相似度”和“关系矩阵”这个数学核心，但职责不同。

## 1. 正交性在做什么

如果有一组 basis：

```text
b1(x), b2(x), b3(x), ...
```

正交性希望：

```text
b_i 和 b_j 尽量不重复
```

数学上常写成：

```text
<b_i, b_j> ≈ 0, i != j
```

如果把每个 basis 在数据集上的输出看成向量：

```text
basis_matrix = [b1_values, b2_values, b3_values]
```

那么正交性就看：

```text
basis_matrix^T basis_matrix
```

非对角线越小，basis 越不冗余。

## 2. Attention 在做什么

Attention 计算：

```text
scores = Q K^T / sqrt(d)
weights = softmax(scores)
output = weights V
```

其中 `QK^T` 也是内积矩阵。

如果有 token 表示：

```text
x1, x2, x3, ...
```

每个 token 会问：

```text
我应该从哪些 token 聚合信息？
```

所以 attention 的关系矩阵是：

```text
token_i 对 token_j 的关注权重
```

## 3. 相似点

### 3.1 都在算关系矩阵

正交性：

```text
basis_i 和 basis_j 的相似度
```

Attention：

```text
token_i 和 token_j 的相关性 / 路由权重
```

它们都可以写成矩阵：

```text
similarity_matrix[i, j]
```

### 3.2 都依赖内积

正交性：

```text
b_i dot b_j
```

Attention：

```text
q_i dot k_j
```

所以你觉得它们像，是有数学原因的。

### 3.3 都在改造表示空间

正交学习：

```text
raw features
  -> better basis set
```

Transformer：

```text
token embeddings
  -> contextual hidden states
```

都是：

```text
旧表示 -> 关系计算 -> 新表示
```

### 3.4 都可以追求多样性

正交 basis 追求 basis 之间不同。

Multi-head attention 也希望不同 head 学不同模式。

```text
head_1: 语法
head_2: 指代
head_3: 局部搭配
head_4: 长距离依赖
```

如果所有 head 都学同一个东西，就浪费容量。

## 4. 不同点

### 4.1 正交性是目标，attention 是机制

正交性回答：

```text
这组 basis 好不好？是否冗余？
```

Attention 回答：

```text
当前 token 应该从哪里拿信息？
```

所以：

```text
orthogonality = objective / regularizer / metric
attention = model mechanism / routing operator
```

### 4.2 正交性偏去冗余，attention 偏按需聚合

正交性希望不同 basis 尽量独立。

Attention 不一定希望 token 互相独立。它可能故意让相似 token 互相关注。

比如：

```text
"New" 和 "York" 应该强相关
```

这不是坏事。

### 4.3 正交性常是全局结构，attention 是输入相关结构

正交 basis 通常评估整个数据集上的结构。

Attention 每个样本、每一层、每个 head 都可能不同。

```text
orthogonal basis: global geometry
attention map: input-conditioned routing geometry
```

### 4.4 正交矩阵不等于 attention 矩阵

正交矩阵通常希望：

```text
A^T A = I
```

Attention 权重通常是 softmax 后的概率分布：

```text
sum_j weights[i, j] = 1
weights[i, j] >= 0
```

它不要求正交。

## 5. Multi-head attention 和正交 basis 的强连接

Multi-head attention 有多个 head：

```text
head_1_output
head_2_output
head_3_output
...
```

可以对它们加正交/多样性约束：

```text
corr(head_i_output, head_j_output) 尽量小
```

这会鼓励不同 head 捕捉不同信息。

对应你的正交符号学习：

```text
symbolic basis_i 和 basis_j 尽量不冗余
```

所以可以设计：

```text
OrthogonalAttentionRegularizer
  reads: model.attention_heads / hidden states
  provides: attention_head_corr / orthogonal_penalty
```

## 6. Attention 可以看成动态 basis 选择

Basis 展开：

```text
f(x) = sum_j alpha_j * basis_j(x)
```

Attention 输出：

```text
output_i = sum_j weight_ij(x) * value_j(x)
```

非常像，但有关键区别：

| 项 | basis expansion | attention |
| --- | --- | --- |
| 被组合对象 | basis function | value vector |
| 权重 | 参数或函数 | 由 Q/K 动态算出 |
| 是否输入相关 | 不一定 | 强输入相关 |
| 是否要求正交 | 可作为目标 | 默认不要求 |

所以 attention 可以理解为：

```text
input-conditioned dynamic basis routing
```

这和你的动态函数池、条件/分段建模很接近。

## 7. 从正交符号学习看 Transformer

你的符号正交学习：

```text
outer search:
  搜 basis 结构

inner fit:
  拟合 basis 参数

problem:
  评估 orthogonality / condition / rank
```

Transformer：

```text
representation:
  定义 attention block / heads / FFN / norm

adapter:
  backprop 优化参数

problem:
  next-token loss / classification loss

optional regularizer:
  head diversity / hidden-state orthogonality
```

两者可以统一成：

```text
构造表示空间
  -> 评估表示质量
  -> 优化结构/参数
```

## 8. 可以搜索哪些 Transformer 机制

如果用 `nsgablack` 做外层搜索，可以搜：

| 机制 | 示例 |
| --- | --- |
| number of heads | 4 / 8 / 16 |
| head dimension | 32 / 64 / 128 |
| attention type | full / sparse / local / linear |
| position encoding | absolute / RoPE / relative |
| FFN type | GELU / SwiGLU |
| normalization | LayerNorm / RMSNorm |
| orthogonality weight | 0.0 / 0.01 / 0.1 |
| LoRA rank | 4 / 8 / 16 / 32 |
| dropout | 0.0 / 0.1 |

内层 `mlblack` 固定这个 spec，然后用 backprop/LoRA 拟合参数。

这和符号嵌套很像：

```text
外层搜结构机制
内层拟合参数
外层看多目标反馈
```

## 9. 可以加哪些正交指标

| 指标 | 含义 |
| --- | --- |
| attention head correlation | 不同 head 输出是否重复 |
| Q projection correlation | Q 投影空间是否冗余 |
| K projection correlation | K 投影空间是否冗余 |
| value subspace diversity | V 子空间是否多样 |
| hidden-state rank | hidden states 是否塌缩 |
| token representation covariance | token 表示是否高度共线 |
| expert diversity | MoE expert 是否学重复功能 |

这些可以变成：

```text
problem metric
regularization loss
outer objective
capability report
```

## 10. 一个框架化草图

```text
nsgablack outer solver
  unknown candidate:
    num_heads
    head_dim
    attention_type
    orthogonality_weight
    lora_rank

  decode:
    TransformerSpec

  evaluate:
    mlblack inner trainer
      representation = TransformerRepresentation(spec)
      head = ClassificationHead / LMHead
      problem = TransformerProblem(data)
      adapter = TorchBackpropAdapter / LoRAAdapter

  objectives:
    validation loss
    latency
    parameter count
    attention_head_corr
    memory cost
```

## 11. 最重要的区别再强调一次

不要把 attention 误认为“就是正交”。

更准确是：

```text
Attention 使用相似度做动态信息路由。
正交性使用相似度评估表示冗余。
```

它们可以结合：

```text
用正交/多样性目标约束 attention heads
```

但它们不是同一个东西。

## 12. 你可以怎么学

你当前最适合从这个角度学：

```text
1. 先理解 attention 的矩阵关系
2. 再理解 multi-head 为什么像多 basis
3. 再理解 head diversity / orthogonal regularization
4. 最后再看 LoRA、RAG、LLM agent
```

这比直接读大模型训练工程更适合当前阶段。
