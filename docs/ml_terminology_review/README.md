# 优化统一视角下的机器学习术语复习

这份文档的目标不是把传统机器学习术语重新背一遍，而是把常见名词翻译成你当前更熟悉的 `optimization-first` 视角。

你的主线可以稳定记成：

```text
unknown state / spec
  -> decoder / codec
  -> model / function / neural graph
  -> head / problem / evaluation
  -> feedback / loss / metrics
  -> adapter / optimizer / search
  -> artifact / report / checkpoint
```

传统机器学习经常用“模型、训练、特征、loss、head、backbone、embedding、attention、temperature”等词。它们不是另一套世界，大多数都能映射到上面这条链路。

## 1. 总览：你的语言和传统语言怎么对齐

| 你的框架语言 | 传统 ML 名词 | 一句话解释 | 面试表达 |
| --- | --- | --- | --- |
| unknown state | 参数、权重、候选解、latent state | 当前还不知道、需要被优化的东西 | The learnable state is represented explicitly and decoded into a model. |
| representation | 表示、编码空间、搜索空间 | unknown state 的形状和合法范围 | The representation defines the candidate space. |
| codec / decoder | 解码器、model builder、architecture builder | 把 unknown state/spec 变成可执行模型 | The codec maps an encoded state or spec to an executable model. |
| model | 模型、函数、网络、公式 | 输入数据后能输出结果的对象 | A model is the executable function after decoding. |
| head | 输出头、任务头 | 决定模型输出语义，如回归、分类、概率、区间 | The head defines the output semantics and task-specific interpretation. |
| problem | 任务、数据集、objective wrapper | 吃数据，调用模型，计算 loss/metric/constraint | The problem evaluates a decoded model on data and returns feedback. |
| feedback | loss、metric、objective、gradient | 评估结果，告诉优化器好坏和方向 | Feedback is the signal used by the optimizer. |
| adapter | optimizer、trainer、search algorithm | 根据 feedback 更新 unknown state | The adapter owns the optimization strategy. |
| artifact | checkpoint、model card、trained model | 训练后的可复现产物 | The artifact stores the fitted model, metadata, and audit signals. |
| capability | plugin、callback、tracking、checkpoint | 不改变主优化语义的工程能力 | Capabilities add side effects such as checkpointing and tracking. |
| bias | inductive bias、regularization、prior | 软偏好，让搜索更偏向某类解 | Bias guides optimization without replacing constraints. |

注意：传统 ML 里经常把这些东西混着叫“模型”或“训练流程”。你的框架更细，因为你把“可优化对象、解码器、任务头、评估、优化策略、产物”拆开了。

## 2. 最小闭环：什么叫训练

传统说法：

```text
data -> model -> prediction -> loss -> optimizer -> updated model
```

你的说法：

```text
data
  -> problem.evaluate(decoded_model)
  -> feedback
  -> adapter.update(unknown_state)
  -> next unknown_state
```

更完整地说：

1. `representation` 产生或维护一个 unknown state。
2. `codec` 把 unknown state 解码成模型。
3. `problem` 把模型放到数据上评估。
4. `head` 决定输出怎么解释，比如标量、类别概率、区间、排序分数。
5. `loss / metric / constraints` 给出反馈。
6. `adapter` 用反馈更新 unknown state。
7. `artifact` 保存结果和审计信息。

所以“训练”不是神秘动作，本质是重复执行：

```text
decode -> evaluate -> feedback -> update
```

## 3. 数据相关术语

### 3.1 Data / Dataset

数据集是一组样本。传统里常写成：

```text
X: features
y: labels / targets
```

例如回归：

```text
X = 房屋面积、楼层、城市
y = 房价
```

你的视角：

```text
DataView / ProblemData
  -> problem consumes X, y
  -> model predicts y_hat
  -> feedback compares y_hat and y
```

### 3.2 Feature

Feature 是模型输入的维度或字段。

传统说法：

```text
x0, x1, x2
```

你的说法：

```text
原始输入变量 / 原子对象 / feature_space 的基础维度
```

符号学习里，feature 可以作为表达式的原子：

```text
sin(x0) + x1 * x2
```

Transformer 里，token 被 tokenizer 转成 token id，再由 embedding 变成向量。这个向量也可以被理解成特征表示。

### 3.3 Label / Target

Label 是监督学习里的正确答案。

例子：

```text
classification: 这张图是 cat
regression: 这个房子的价格是 300 万
language model: 下一个 token 是 "world"
```

你的视角：

```text
label 是 problem 评估时的 reference / ground truth
```

### 3.4 Numericizer

Numericizer 是把非数值对象变成数值输入的组件。

传统术语可能叫：

| 场景 | 传统叫法 | 你的叫法 |
| --- | --- | --- |
| 文本转 token id | tokenizer | numericizer / tokenizer |
| 类别转数字 | label encoder / one-hot encoder | numericizer |
| 图像转 tensor | image transform | data view / tensorizer |
| 图转 node/edge tensor | graph featurizer | graph data view |

重要结论：

tokenizer 本身通常不是“理解语言”的核心，它更多是把文本切成稳定的离散编号。真正学到语义关系的是 embedding、attention、FFN 等可训练映射。

## 4. 模型家族总表

| 模型家族 | 传统解释 | 你的统一解释 | 常见场景 |
| --- | --- | --- | --- |
| Linear Model | 线性组合 | 固定 decoder，优化线性参数 | 回归、分类 baseline |
| Logistic Regression | 线性模型 + sigmoid/softmax | linear decoder + classification/probability head | 二分类、多分类 |
| Tree | if/else 分裂 | piecewise routing model | 表格数据 |
| Random Forest | 多棵树集成 | tree ensemble artifact | 表格数据、鲁棒 baseline |
| Boosting / XGBoost | 逐步补残差的树集成 | residual-driven additive model | 表格数据强 baseline |
| MLP | 多层全连接神经网络 | NeuralGraphSpec 的基础路由 | 通用函数拟合 |
| CNN | 卷积神经网络 | 适合局部空间结构的 neural graph | 图像、局部模式 |
| GNN | 图神经网络 | 适合 node/edge/message passing 的 neural graph | 分子、社交网络、知识图 |
| Transformer | attention + FFN 堆叠 | 可配置 neural graph 的一种 decoder | 文本、序列、多模态 |
| Symbolic Regression | 搜表达式结构和常数 | 外层搜 decoder/structure，内层拟合参数 | 可解释公式发现 |

关键判断：

如果结构固定，只优化参数，它更像普通 ML 或神经网络训练。

如果结构也在变，比如公式结构、神经网络结构、函数池、Transformer spec，它就是“结构搜索 + 参数优化”的嵌套问题。

## 5. Head：为什么它很重要

Head 决定“模型输出到底是什么意思”。

同一个 backbone 或 decoder，换 head 后任务就会变。

| Head | 输出 | Loss / Metric | 典型任务 |
| --- | --- | --- | --- |
| Regression Head | 连续数值 | MSE、MAE、RMSE | 房价、温度、收益预测 |
| Classification Head | 类别 logits | Cross Entropy、Accuracy、F1 | 图像分类、文本分类 |
| Probability Head | 概率 | NLL、Brier、Calibration Error | 风险预测、概率分类 |
| Interval Head | 下界/上界 | coverage、width、pinball loss | 区间预测 |
| Ranking Head | 排序分数 | pairwise loss、NDCG | 搜索排序、推荐 |
| Retrieval Head | embedding 向量 | contrastive/triplet loss | 召回、相似度检索 |
| LM Head | 每个词表 token 的 logits | next-token cross entropy | 语言模型 |
| Preference Head | 偏好分数 | DPO、pairwise preference loss | 对齐、偏好学习 |
| Orthogonal Head | 一组基函数或向量 | orthogonality metrics | 正交基搜索 |

面试表达：

```text
The head is the task-specific output layer. It maps the shared representation or model output into the target space required by the problem.
```

你的表达：

```text
head 是 decoder 输出语义的一部分。它决定 problem 如何解释输出和计算反馈。
```

## 6. Loss、Metric、Objective、Constraint

这些词经常混用，但最好区分。

| 名词 | 作用 | 是否用于优化 | 例子 |
| --- | --- | --- | --- |
| Loss | 训练时最直接优化的误差 | 通常是 | cross entropy、MSE |
| Metric | 观察模型效果的指标 | 不一定 | accuracy、F1、AUC |
| Objective | 优化目标 | 是 | minimize RMSE + penalty |
| Constraint | 约束条件 | 是或作为过滤 | latency < 10ms、coverage > 0.9 |

你的框架里可以统一成：

```text
Feedback(
  objectives=...,
  metrics=...,
  constraints=...,
  gradients=...,
  residuals=...
)
```

### 6.1 Accuracy

分类预测对了多少。

问题：类别不平衡时可能很假。

例如 99% 都是负样本，模型全预测负样本也有 99% accuracy，但毫无价值。

### 6.2 Precision / Recall / F1

Precision：预测为正的里面有多少是真的。

Recall：真实为正的里面有多少被找出来。

F1：Precision 和 Recall 的调和平均。

面试常用解释：

```text
Precision cares about false positives, recall cares about false negatives, and F1 balances both.
```

### 6.3 AUC / ROC

AUC 衡量模型把正样本排在负样本前面的能力。

它不依赖某个固定阈值，所以适合比较概率/打分模型的排序能力。

### 6.4 Calibration

Calibration 关注“概率准不准”。

如果模型说 100 个样本都有 0.8 概率为正，那么理想情况下大约 80 个真的为正。

这和 accuracy 不一样。一个模型可以 accuracy 高但概率不准。

## 7. 训练和优化术语

### 7.1 Parameter / Weight

参数是训练要更新的数值。

例如线性模型：

```text
y = w0 + w1*x1 + w2*x2
```

`w0, w1, w2` 是参数。

神经网络里每层矩阵都是参数。

Embedding 表也是参数：

```text
token_id -> embedding vector
```

每个 token 对应的向量会随着训练更新。

### 7.2 Hyperparameter

超参数不是训练直接学出来的，一般由人或外层搜索设定。

例子：

```text
learning_rate
batch_size
num_layers
hidden_dim
dropout
tree_depth
```

你的视角：

```text
hyperparameter 可以进入 spec，由外层 nsgablack 搜。
parameter 由内层 mlblack 拟合。
```

### 7.3 Gradient

Gradient 是 loss 对参数的导数，表示“参数往哪个方向改，loss 会下降”。

数学上：

```text
gradient = d(loss) / d(parameter)
```

你的符号学习梯度拓池也类似：不是只看当前表达式效果，还看 residual/gradient 提示下一步该扩什么函数。

### 7.4 Backpropagation

反向传播是自动计算复杂嵌套函数梯度的方法。

神经网络是很多层函数嵌套：

```text
f(x) = layer_96(...layer_2(layer_1(x)))
```

只要每层可微，就能用链式法则把梯度从 loss 传回每一层参数。

面试表达：

```text
Backpropagation applies the chain rule to compute gradients of the loss with respect to all learnable parameters in a composed differentiable model.
```

你的表达：

```text
backprop 是对 decoder 产生的可微计算图做自动反馈传播，让 adapter 能更新 unknown state 里的参数。
```

### 7.5 Optimizer

Optimizer 决定怎么用梯度更新参数。

常见：

| Optimizer | 特点 |
| --- | --- |
| SGD | 简单，直接沿负梯度方向走 |
| Momentum | 加入惯性，减少抖动 |
| Adam | 自适应学习率，实践中常用 |
| AdamW | Adam + decoupled weight decay，大模型常用 |

你的视角：

```text
optimizer 是 adapter 的一种。
```

如果是梯度下降，adapter 用 gradient。

如果是随机搜索，adapter 用 objective 排名。

如果是 NSGA，adapter 用多目标反馈。

### 7.6 Learning Rate

学习率控制每次更新走多大。

太大：震荡或发散。

太小：训练很慢或卡住。

面试表达：

```text
The learning rate controls the step size of parameter updates. It is one of the most important hyperparameters.
```

### 7.7 Batch / Step / Epoch

| 名词 | 含义 |
| --- | --- |
| batch | 一次训练用的一小批样本 |
| step / iteration | 用一个 batch 完成一次参数更新 |
| epoch | 整个训练集被看过一遍 |

你的视角：

```text
step 是 adapter.update 的一次节奏。
epoch 是数据遍历策略的一种生命周期口径。
```

### 7.8 Checkpoint / Resume

Checkpoint 是训练中保存的状态，方便恢复。

通常包含：

```text
model parameters
optimizer state
step / epoch
random seed
config
metrics
```

你的框架里对应：

```text
TrainerStateArtifact
ModelArtifact
ArtifactBundle
```

## 8. 过拟合、泛化和验证

### 8.1 Train / Validation / Test

| 集合 | 用途 |
| --- | --- |
| train | 用来训练参数 |
| validation | 用来调超参、选模型 |
| test | 最后只看一次，估计真实泛化 |

不要把 test 集拿来反复调参，否则 test 就泄漏成 validation 了。

### 8.2 Overfitting

过拟合：训练集表现很好，新数据表现差。

你的视角：

```text
模型/decoder/spec 过度适配当前 problem data，artifact 泛化能力不足。
```

常见解决：

```text
regularization
dropout
early stopping
data augmentation
smaller model
cross validation
better validation split
```

### 8.3 Underfitting

欠拟合：模型能力不够，训练集都学不好。

解决：

```text
更强模型
更多特征
更长训练
更少正则
更好的 decoder/spec
```

### 8.4 Bias-Variance Tradeoff

Bias 高：模型太简单，系统性错。

Variance 高：模型太灵活，对训练数据噪声敏感。

你的视角：

```text
bias 是 representation/decoder/head/problem 施加的结构偏好。
variance 是搜索空间和参数自由度过大导致的泛化风险。
```

### 8.5 Data Leakage

数据泄漏：训练时不该知道的信息进入了模型。

例子：

```text
用未来信息预测过去
把 test 统计量用于 train normalization
目标 y 的派生字段混进 X
```

这类问题面试经常问，因为它比模型结构更容易导致真实失败。

## 9. 神经网络总览

神经网络不是一种单一算法，而是一类可微函数组合。

你可以把它理解成：

```text
NeuralGraphSpec
  -> NeuralGraphCodec
  -> torch module
  -> LearningProblem
  -> BackpropAdapter
```

每一层是一个数学映射：

```text
x -> layer1 -> activation -> layer2 -> activation -> output head
```

可微的原因是每一步大多数都是可导函数，可以用链式法则计算梯度。

### 9.1 Layer

Layer 是一个可训练或不可训练的映射。

例子：

```text
Linear
Convolution
Attention
Normalization
Pooling
Dropout
Activation
```

### 9.2 Activation

Activation 是非线性函数。

如果没有非线性，多层线性叠加还是线性。

常见：

```text
ReLU
GELU
SiLU
Tanh
Sigmoid
```

### 9.3 Dropout

Dropout 训练时随机丢掉一部分神经元输出，减少过拟合。

推理时通常关闭。

所以模型有：

```text
train mode
eval mode
```

### 9.4 Normalization

Normalization 让中间表示的数值更稳定。

常见：

```text
BatchNorm
LayerNorm
RMSNorm
```

Transformer 里常见的是 LayerNorm 或 RMSNorm。

## 10. MLP

MLP 是最基础的全连接神经网络。

形式：

```text
x -> Linear -> Activation -> Linear -> Activation -> Head
```

它适合一般向量输入，但不利用图像局部结构或图结构。

你的框架里，MLP 是 NeuralGraphSpec 的基础路线之一。

面试表达：

```text
An MLP is a feed-forward neural network composed of fully connected layers and nonlinear activations.
```

## 11. CNN

CNN 主要用于图像或局部空间结构数据。

### 11.1 Image Tensor

图像通常表示成：

```text
batch, channel, height, width
```

例如：

```text
32 张 RGB 图片，每张 64x64
shape = [32, 3, 64, 64]
```

### 11.2 Convolution

Convolution 用一个小窗口在图像上滑动，提取局部模式。

这个小窗口叫：

```text
kernel / filter
```

例如 3x3 filter 会看局部 3x3 像素。

### 11.3 Channel

Channel 是特征通道。

原始 RGB 图像有 3 个 channel。

CNN 中间层可能有 16、32、64 个 channel，每个 channel 学一种局部模式。

### 11.4 Feature Map

卷积输出叫 feature map。

它表示某个 filter 在不同空间位置上的响应。

### 11.5 Stride / Padding

Stride：窗口每次移动几格。

Padding：边缘补零，避免尺寸过快变小。

### 11.6 Pooling

Pooling 做下采样，常见 max pooling。

作用：

```text
减少空间尺寸
增加局部平移鲁棒性
降低计算量
```

### 11.7 CNN 的统一视角

传统说法：

```text
CNN learns local visual features using convolutional filters.
```

你的说法：

```text
CNN 是 neural graph decoder 的一种 block，它把图像 data view 映射到 head 需要的表示。卷积核是可训练参数，结构由 spec 决定。
```

面试回答：

```text
A CNN is suitable for images because convolution shares weights across spatial locations and captures local patterns efficiently.
```

## 12. GNN

GNN 用于图结构数据。

图由：

```text
nodes
edges
node features
edge features
adjacency
```

组成。

### 12.1 Node / Edge

Node 是节点，Edge 是边。

例子：

```text
分子图: atom 是 node, chemical bond 是 edge
社交网络: person 是 node, friendship 是 edge
知识图谱: entity 是 node, relation 是 edge
```

### 12.2 Adjacency

Adjacency 描述哪些节点相连。

可以是矩阵，也可以是 edge list。

### 12.3 Message Passing

GNN 的核心是 message passing：

```text
每个节点从邻居收集信息
聚合邻居信息
更新自己的表示
```

形式上：

```text
h_v_new = update(h_v, aggregate({h_u | u in neighbors(v)}))
```

### 12.4 Graph Pooling

如果任务是整张图分类，需要把所有节点表示汇总成一个图表示。

常见：

```text
mean pooling
sum pooling
max pooling
attention pooling
```

### 12.5 GCN / GAT

GCN：Graph Convolutional Network，用邻接关系做平滑/聚合。

GAT：Graph Attention Network，对不同邻居分配不同 attention 权重。

### 12.6 GNN 的统一视角

你的说法：

```text
GNN 是 neural graph decoder 的一种 block。GraphDataView 提供 node/edge/adjacency，codec 构建 message passing module，head 决定 node-level、edge-level 或 graph-level 输出。
```

面试回答：

```text
A GNN generalizes neural networks to graph-structured data by iteratively aggregating information from neighboring nodes.
```

## 13. Transformer

Transformer 是一种可配置神经网络图，核心组件是：

```text
tokenizer
embedding
positional encoding
self-attention
FFN / MLP
residual connection
normalization
head
```

你已经抓住了重点：它不是魔法，本质是很多数学映射一层层组合。

### 13.1 Tokenizer

Tokenizer 把文本切成 token，并映射到 token id。

例子：

```text
"hello world" -> ["hello", " world"] -> [15339, 1917]
```

注意：tokenizer 多数情况下不是主要学习语义的部分。它更像文本的数值化入口。

你的类比：

```text
tokenizer = 文本 numericizer
token id = 离散主键
```

### 13.2 Vocabulary

Vocabulary 是 token id 的全集。

语言模型输出时，LM head 会给 vocabulary 中每个 token 一个 logit。

### 13.3 Embedding

Embedding 是可训练查表：

```text
token_id -> vector
```

初始化时可以是随机的，训练后会因为任务反馈变得有结构。

你的修正理解是对的：

```text
embedding 不是普通 adapter，而是模型参数的一部分。
```

### 13.4 Positional Encoding

Attention 本身不天然知道顺序，所以需要位置信息。

常见：

```text
learned positional embedding
sinusoidal positional encoding
RoPE
```

RoPE 常用于现代 LLM，让 attention 带有相对位置信息。

### 13.5 Attention

Attention 是一种动态关系计算机制。

对每个 token，它会根据当前句子上下文决定更关注哪些 token。

这就是“动态”的含义：

```text
同一个 token 在不同句子里，attention 权重可以不同。
```

例如 `bank`：

```text
river bank: 更关注 river
bank account: 更关注 account
```

### 13.6 Q / K / V

Self-attention 会从每个 token 的表示生成：

```text
Q: query, 我想找什么
K: key, 我是什么特征
V: value, 我能提供什么信息
```

计算：

```text
attention_score = Q @ K.T
attention_weight = softmax(attention_score)
output = attention_weight @ V
```

面试回答：

```text
Queries and keys determine attention weights, and values provide the information being mixed.
```

你的视角：

```text
attention 是一种可训练的关系路由和信息混合机制。
```

### 13.7 Multi-Head Attention

这里的 head 和任务 head 不是同一个概念。

Multi-head attention 是把 attention 分成多组子空间并行计算。

直觉：

```text
一个 attention head 学语法关系
一个 attention head 学指代关系
一个 attention head 学局部位置关系
```

真实模型里不一定这么干净，但这个解释适合理解。

### 13.8 FFN

Transformer 里的 FFN 通常是每个 token 独立经过一个 MLP：

```text
x -> Linear(up) -> activation -> Linear(down)
```

常见是先升维，再降维。

你的类比是合理的：

```text
FFN 像一种高维非线性映射，提供局部 token 表示变换能力。
```

但它不是傅里叶变换。FFN 是 feed-forward network，不是 Fast Fourier Transform。

### 13.9 SwiGLU

SwiGLU 是一种现代 FFN 变体。

粗略形式：

```text
FFN(x) = Linear1(x) * SiLU(Linear2(x))
```

它用 gating 机制控制信息流，很多现代 LLM 使用类似结构。

### 13.10 Residual Connection

Residual 是：

```text
output = x + block(x)
```

作用：

```text
减少深层网络训练困难
保留原始信息通路
让梯度更容易传回前面层
```

你的理解“避免特征损失”是重要一部分，但更完整地说还包括优化稳定性和梯度流。

### 13.11 LayerNorm / RMSNorm

Norm 是为了稳定中间表示尺度，让训练更稳定。

现代 Transformer 常用：

```text
LayerNorm
RMSNorm
```

### 13.12 Causal Mask

语言模型预测下一个 token 时，不能看到未来 token。

Causal mask 会挡住未来位置：

```text
当前位置只能 attend 到自己和之前的位置
```

### 13.13 LM Head

LM head 把最后的 hidden state 映射到词表大小：

```text
hidden_dim -> vocab_size
```

输出是 logits。

再 softmax 得到每个 token 的概率。

### 13.14 Transformer Block

一个常见 decoder-only Transformer block：

```text
x
  -> norm
  -> masked self-attention
  -> residual add
  -> norm
  -> FFN
  -> residual add
```

大模型所谓 96 层，基本就是这种 block 堆叠很多次。每一层都是一次函数映射，所以空间和参数量非常大。

### 13.15 Transformer 的统一视角

你的说法：

```text
Transformer 是 NeuralGraphSpec 的一种 route。
tokenizer/numericizer 负责数据化，embedding 是参数表，attention/FFN/residual/norm 是 decoder 里的可微模块，head 决定分类、LM、embedding、ranking 或 preference 任务。
```

面试回答：

```text
A Transformer is a neural architecture based on self-attention, feed-forward layers, residual connections, and normalization. Self-attention allows each token to dynamically aggregate information from other tokens in the sequence.
```

## 14. LLM 推理和 generation

### 14.1 Logits

Logits 是 softmax 前的原始分数。

```text
logits -> softmax -> probabilities
```

### 14.2 Sampling

生成文本时，模型每一步输出下一个 token 的概率分布。

然后需要一个解码策略选择 token。

常见：

```text
greedy decoding
temperature sampling
top-k
top-p / nucleus sampling
beam search
```

### 14.3 Temperature

Temperature 控制采样随机性。

简化公式：

```text
probabilities = softmax(logits / temperature)
```

temperature 越低，分布越尖锐，越倾向最高分 token。

temperature 越高，分布越平，随机性越强。

### 14.4 Temperature = 0 是什么意思

面试常问。

严格说 `temperature=0` 不能直接代入 `logits / temperature`，因为除以 0 不合法。工程上通常把它解释成：

```text
不采样，直接选概率最高的 token
```

也就是 greedy decoding。

回答模板：

```text
Temperature controls randomness during generation. Temperature 0 usually means deterministic greedy decoding: at each step, choose the highest-probability token instead of sampling. It is an inference-time decoding setting, not a training hyperparameter.
```

补充：

即使 temperature=0，某些服务也可能因为并行计算、浮点差异、tie-breaking、后端实现导致极少数不完全一致，但概念上它表示确定性生成。

### 14.5 Top-k

只在概率最高的 k 个 token 中采样。

### 14.6 Top-p

从累计概率达到 p 的最小 token 集合里采样。

例如 `top_p=0.9`：只考虑累计概率前 90% 的 token。

### 14.7 KV Cache

自回归生成时，每一步都要 attend 到之前 token。

如果每次重新算所有历史 token，很慢。

KV cache 缓存历史 token 的 key/value，下一步只算新 token 的 query 和增量 key/value。

面试回答：

```text
KV cache speeds up autoregressive generation by reusing previously computed keys and values instead of recomputing the full attention history.
```

你的视角：

```text
KV cache 是 inference artifact / runtime cache，不是训练目标。
```

### 14.8 Context Window

Context window 是模型一次能处理的 token 长度。

比如 8k、32k、128k tokens。

超过窗口需要截断、压缩、RAG 或长上下文策略。

## 15. Pretraining、Fine-tuning、LoRA、QLoRA

### 15.1 Pretrained Model

Pretrained model 是已经在大数据上训练好的模型。

你一般不会从零训练大模型，因为成本太高。

### 15.2 Fine-tuning

Fine-tuning 是在预训练模型基础上继续训练，让它适配特定任务或风格。

### 15.3 PEFT

PEFT 是 Parameter-Efficient Fine-Tuning。

目标：不要更新全部参数，只更新少量新增参数。

### 15.4 LoRA

LoRA 是一种 PEFT 方法。

它冻结原模型权重，只训练低秩矩阵增量。

粗略理解：

```text
W_new = W_frozen + A @ B
```

其中 A 和 B 很小，所以训练成本低。

面试回答：

```text
LoRA adapts a large pretrained model by freezing the original weights and training low-rank update matrices, which greatly reduces trainable parameters.
```

你的视角：

```text
LoRA adapter 是 decoder 中某些层的可训练增量参数，不是重写整个模型。
```

### 15.5 QLoRA

QLoRA = quantized LoRA。

它把基础模型量化到低精度，例如 4-bit，再训练 LoRA 参数。

好处：显存占用更低。

### 15.6 Quantization

Quantization 是把模型参数从高精度变成低精度。

例如：

```text
FP32 -> FP16 -> INT8 -> 4-bit
```

作用：省显存、加速推理。

风险：可能损失精度。

## 16. RAG、Embedding Model、Vector Database

### 16.1 RAG

RAG 是 Retrieval-Augmented Generation。

流程：

```text
user query
  -> retrieve relevant documents
  -> put documents into prompt
  -> LLM generates answer
```

它不是训练模型，而是在推理时给模型补外部知识。

### 16.2 Embedding Model

Embedding model 把文本、图片或其他对象映射成向量。

相似对象应该向量距离更近。

### 16.3 Vector Database

Vector database 存 embedding 向量，并支持相似度检索。

常见检索：

```text
cosine similarity
dot product
L2 distance
```

### 16.4 Reranker

Reranker 对初步检索结果重新排序。

常见结构：

```text
retriever 召回 100 个
reranker 精排前 10 个
LLM 用前 10 个生成答案
```

## 17. Preference、DPO、RLHF

### 17.1 Preference Data

偏好数据通常长这样：

```text
prompt
chosen response
rejected response
```

表示人类或规则更喜欢 chosen。

### 17.2 RLHF

RLHF 是 Reinforcement Learning from Human Feedback。

经典流程：

```text
supervised fine-tuning
reward model
policy optimization
```

### 17.3 DPO

DPO 是 Direct Preference Optimization。

它绕过显式 reward model，直接用 chosen/rejected 训练模型偏好。

面试回答：

```text
DPO trains a model from preference pairs by increasing the likelihood of preferred responses relative to rejected ones, without requiring a separate reward model.
```

你的视角：

```text
preference head / problem 把输出解释成偏好比较，feedback 不再是单个 label，而是 pairwise preference signal。
```

## 18. Contrastive Learning 和 Retrieval

### 18.1 Contrastive Learning

Contrastive learning 的目标是：

```text
相似样本靠近
不相似样本远离
```

例子：

```text
同一张图片的两种增强视图应该接近
图片和对应文本应该接近
不匹配图片和文本应该远离
```

### 18.2 Triplet Loss

Triplet:

```text
anchor
positive
negative
```

目标：

```text
distance(anchor, positive) < distance(anchor, negative)
```

### 18.3 Retrieval Head

Retrieval head 输出 embedding，用于相似度检索。

你的视角：

```text
retrieval head 不是直接输出类别，而是输出一个可比较、可索引的向量空间。
```

## 19. Symbolic Learning / Symbolic Regression

这是你框架里很核心的一块。

传统说法：

```text
Symbolic regression searches for mathematical expressions that fit data.
```

你的统一说法：

```text
符号学习是 decoder 不固定的机器学习。
外层搜索表达式结构，内层拟合当前结构下的 constants / parameters。
```

### 19.1 Expression Tree

表达式可以表示成树：

```text
sin(a*x + b)
```

树结构：

```text
sin
  +
    *
      a
      x
    b
```

### 19.2 Primitive / Function Pool

Primitive 是可用算子。

例子：

```text
+, -, *, /, sin, cos, exp, log, pow, piecewise, hinge
```

Function pool 是当前允许搜索的函数集合。

你的关键判断：

```text
函数池不是普通模型参数，它更像 pipeline / codec 的结构资源，也可以由 dynamic pool 机制扩展。
```

### 19.3 Dynamic Pool

Dynamic pool 根据 residual、gradient、budget、gate 等信号扩池或剪枝。

它回答：

```text
现在应该允许哪些函数进入搜索？
哪些函数应该被剪掉？
```

### 19.4 Simplification / Canonicalization

Simplification 是化简表达式。

Canonicalization 是把等价表达式变成统一规范形式。

例子：

```text
x + x -> 2*x
sin(x)^2 + cos(x)^2 -> 1
exp(log(x)) -> x
```

作用：

```text
减少重复搜索
更好比较表达式
更稳定 artifact key
```

### 19.5 Truth Recovery

Truth recovery 是判断搜索出来的表达式是否恢复了真实公式。

不只是 RMSE 小，还要看结构是否等价或近似等价。

### 19.6 Phase-equivalent Scoring

例如：

```text
sin(x)
cos(x - pi/2)
sin(x + 2*pi)
```

这些在相位变换下可能属于同一个函数族。

Phase-equivalent scoring 就是识别这种相位等价或近似恢复。

### 19.7 Family-level Scoring

Family-level scoring 不要求表达式完全一样，而是判断是否恢复了正确函数族。

例如真实是：

```text
sin(a*x + b)
```

模型搜到：

```text
cos(a*x + c)
```

可能属于同一三角周期族。

### 19.8 符号正交学习

你当前更精确的理解是：

```text
Stage 1:
  外层做正交符号搜索
  搜一组 basis / atom
  每个候选结构都要内层参数拟合
  评估 orthogonality + 其他指标

Stage 2:
  把 Stage 1 的 basis artifact 注册成 Stage 2 原子
  外层搜任务表达式
  内层拟合任务参数
  评估 RMSE / coverage / classification metric 等
```

这不是普通单层符号回归，而是：

```text
(structure + parameter fitting) -> basis artifact
then
(basis-conditioned structure + parameter fitting) -> task artifact
```

编排层属于共享 Project / Case / L0 substrate；需要优化搜索时可由 `nsgablack` 语义 Case 提供外层搜索。内层训练和评估 surface 属于 `mlblack`。

## 20. Artifact

Artifact 是训练或搜索后的正式产物。

常见内容：

```text
model parameters
model spec
codec route
head type
metrics
training config
resource context
canonical expression
truth recovery
family recovery
checkpoint refs
viewer html
```

面试表达：

```text
An artifact should make the result reproducible and auditable, not just store raw weights.
```

你的视角：

```text
artifact 是模型、结构、评估、恢复、审计的统一落盘边界。
```

## 21. Benchmark、Baseline、Ablation

### 21.1 Baseline

Baseline 是对照方法。

例如：

```text
linear regression
random forest
small MLP
simple Transformer
```

没有 baseline，很难判断新方法是否真的好。

### 21.2 Benchmark

Benchmark 是标准化评测。

要固定：

```text
dataset
split
metric
budget
seed
hardware/resource context
```

### 21.3 Ablation

Ablation 是消融实验。

例如：

```text
有 dynamic pool vs 没有 dynamic pool
有 orthogonality objective vs 没有 orthogonality objective
有 LoRA vs full fine-tuning
```

目的：证明某个组件真的有贡献。

## 22. 当前 mlblack 已经验证的路线

当前仓库里已经有几条“统一机器学习”路线的 smoke 能力。

代表性入口：

```text
examples/cases/tiny_transformer_smoke/run_project.py
examples/cases/benchmarks/run_project.py
```

已验证的模型路线包括：

```text
tiny_transformer
tiny_cnn
tiny_gnn
tiny_cnn_image_contrastive
```

这说明当前设计已经支持：

```text
不同数据视图
不同 neural graph route
不同 head
不同 problem
同一套 artifact / benchmark / trainer 口径
```

但不要把它说成生产级大模型平台。更准确的说法是：

```text
The framework has a smoke-capable neural graph route that unifies MLP/CNN/GNN/Transformer-style models under the same optimization-first scaffold.
```

## 23. 面试高频问答模板

### 23.1 什么是 Transformer？

推荐回答：

```text
A Transformer is a neural architecture built around self-attention, feed-forward layers, residual connections, and normalization. Self-attention lets each token dynamically aggregate information from other tokens in the sequence, which makes it effective for language and other sequence tasks.
```

你的补充：

```text
在我的框架视角里，Transformer 是 NeuralGraphSpec 的一种 decoder route。Tokenizer 负责数值化，embedding 是可训练参数表，attention 和 FFN 是可微映射模块，head 决定是 LM、classification、retrieval 还是 preference task。
```

### 23.2 Attention 是什么？

推荐回答：

```text
Attention is a mechanism that computes context-dependent weights between tokens or elements, then mixes value vectors according to those weights.
```

简化解释：

```text
Q/K 决定关注谁，V 决定拿什么信息。
```

### 23.3 Multi-head attention 的 head 和 classification head 一样吗？

回答：

```text
No. An attention head is one parallel attention subspace inside the Transformer block. A task head is the output module for a specific task, such as classification or language modeling.
```

中文：

```text
不一样。attention head 是注意力内部的并行子空间；task head 是任务输出头。
```

### 23.4 为什么 temperature=0？

回答：

```text
Temperature is an inference-time sampling parameter. Temperature 0 is usually implemented as greedy decoding, meaning the model always chooses the highest-probability next token instead of sampling.
```

不要说：

```text
temperature=0 表示模型没有训练
temperature=0 表示 loss 是 0
temperature=0 表示概率都是 0
```

这些都是错的。

### 23.5 CNN 和 Transformer 区别？

回答：

```text
CNNs use local convolutional filters and weight sharing, which makes them efficient for grid-like data such as images. Transformers use self-attention to model dynamic relationships between all tokens or patches, which is more flexible but often more expensive.
```

你的视角：

```text
CNN 和 Transformer 都是 neural graph decoder 的不同 block 组合。CNN 的 inductive bias 是局部性和平移共享；Transformer 的 inductive bias 是基于 attention 的动态关系建模。
```

### 23.6 GNN 是什么？

回答：

```text
A GNN is a neural network for graph-structured data. It updates node representations by aggregating messages from neighboring nodes, and can produce node-level, edge-level, or graph-level predictions.
```

### 23.7 什么是 backpropagation？

回答：

```text
Backpropagation uses the chain rule to compute gradients of the loss with respect to every learnable parameter in a composed differentiable model.
```

你的视角：

```text
它是对可微 decoder 产生的计算图做反馈传播。
```

### 23.8 什么是 overfitting？

回答：

```text
Overfitting happens when a model fits the training data very well but fails to generalize to unseen data. It usually means the model has captured noise or leakage instead of stable patterns.
```

解决：

```text
regularization
validation
early stopping
data augmentation
smaller model
better split
```

### 23.9 什么是 LoRA？

回答：

```text
LoRA is a parameter-efficient fine-tuning method. It freezes the original pretrained weights and trains small low-rank update matrices, reducing memory and compute cost.
```

### 23.10 什么是 RAG？

回答：

```text
RAG retrieves relevant external documents and provides them as context to the language model during generation. It improves factual grounding without necessarily changing model weights.
```

### 23.11 你的框架为什么能统一 ML？

回答模板：

```text
I model machine learning as an optimization-first pipeline. A representation defines the unknown state, a codec decodes it into an executable model or structure, a head defines output semantics, a problem evaluates the model on data and returns feedback, and an adapter updates the unknown state. Under this view, linear models, neural networks, symbolic regression, CNNs, GNNs, and Transformers differ mainly in their representation, codec, head, and problem, not in the overall training scaffold.
```

中文理解：

```text
机器学习不是很多互不相干的模型堆砌，而是同一条优化闭环在不同 decoder/head/problem 下的具体实例。
```

### 23.12 nsgablack 和 mlblack 怎么分工？

回答模板：

```text
The shared Project / Case / L0 substrate provides orchestration, resource scheduling, groups, stages, and runtime control. nsgablack provides optimization/search semantics. mlblack provides ML-specific representation, codec, head, problem evaluation, fitting, and artifacts. For symbolic learning or architecture search, a nsgablack Case may search the outer structure while a mlblack Case evaluates and fits the ML task.
```

中文：

```text
Project / Case / L0 substrate 是统一编排与资源底座；nsgablack 是优化搜索语义层，mlblack 是 ML 语义层。mlblack 不重复实现私有 runtime/group/stage/resource 编排。
```

## 24. 术语速查表

| 术语 | 快速解释 |
| --- | --- |
| token | 文本被切分后的离散单位 |
| tokenizer | 文本 numericizer，把文本变成 token id |
| embedding | token id 到向量的可训练查表 |
| logits | softmax 前的原始分数 |
| softmax | 把 logits 转成概率分布 |
| attention | 动态关系权重和信息混合 |
| Q/K/V | query/key/value，attention 的三组投影 |
| FFN | Transformer 里的逐 token MLP |
| residual | x + block(x)，保留信息和梯度通道 |
| norm | 稳定中间表示尺度 |
| RoPE | 旋转位置编码，给 attention 加位置信息 |
| SwiGLU | 带门控的 FFN 变体 |
| causal mask | 防止语言模型看未来 token |
| KV cache | 生成时缓存历史 key/value |
| temperature | 生成采样随机性参数 |
| top-k | 只在前 k 个 token 采样 |
| top-p | 只在累计概率 p 的 token 集采样 |
| CNN | 用卷积提取局部空间模式 |
| kernel/filter | 卷积窗口 |
| stride | 卷积滑动步长 |
| padding | 图像边缘补零 |
| pooling | 下采样和聚合 |
| GNN | 图结构上的神经网络 |
| message passing | 节点从邻居聚合信息 |
| graph pooling | 把节点表示汇总成图表示 |
| LoRA | 低秩增量微调 |
| QLoRA | 量化基础模型 + LoRA |
| RAG | 检索增强生成 |
| DPO | 直接偏好优化 |
| contrastive learning | 相似拉近，不相似推远 |
| artifact | 可复现、可审计的训练产物 |
| benchmark | 固定条件下的标准评测 |
| ablation | 移除某组件验证贡献 |

## 25. 你应该怎么复习

建议顺序：

1. 先背熟第 1 节的统一链路。
2. 再掌握第 5 节 head，因为很多问题本质是 head 变了。
3. 然后复习第 7 节训练术语，尤其 gradient/backprop/optimizer。
4. 接着看 Transformer：tokenizer、embedding、attention、FFN、residual、norm、LM head、generation。
5. 再看 CNN/GNN，它们本质是 neural graph 的不同 block 和 data view。
6. 最后看 LoRA、RAG、DPO、contrastive，因为这些是 LLM 工程和应用高频词。

最重要的回答习惯：

```text
先用传统术语回答，再用你的统一框架补一层解释。
```

例如问 Transformer：

```text
传统回答：Transformer uses self-attention, FFN, residual connections and normalization.
框架回答：In my framework, it is a NeuralGraphSpec route decoded into a torch module, with task-specific heads for LM/classification/retrieval/preference.
```

这样既能让面试官听懂，也能体现你的架构理解。

## 26. 一句话总复习

机器学习可以统一看成：

```text
选择一个可优化对象
设计一个 decoder 把它变成模型
设计一个 head 定义输出语义
设计一个 problem 在数据上评价它
设计一个 adapter 根据 feedback 更新它
最后保存 artifact 让结果可复现
```

线性模型、树、神经网络、CNN、GNN、Transformer、符号回归，本质上都可以放进这条链路里。区别在于：它们的 representation、codec、head、problem、optimizer 和 artifact 设计不同。

## 27. nsgablack 与传统优化术语对照

这部分专门回答一个高频问题：`nsgablack` 的词和传统优化算法（尤其是演化算法/元启发式）到底怎么对齐。

| nsgablack 术语 | 传统优化术语 | 对齐说明 |
| --- | --- | --- |
| problem | objective function + constraints + evaluator | 不只是数学函数，还包含数据/评估协议。 |
| representation | solution encoding / search space encoding | 决策变量的编码、边界、合法性规则。 |
| unknown state / candidate | individual / solution / decision vector | 当前被优化的候选解。 |
| codec / decoder | genotype-to-phenotype mapping / decode step | 把编码向量转成可评估结构或模型。 |
| adapter | optimizer / search operator stack / metaheuristic | 搜索策略本体，负责 propose/update。 |
| feedback | fitness/objective values + constraint violation | 统一返回目标、约束、指标、梯度等信号。 |
| population | population | 与传统 EA 一致。 |
| generation | generation / iteration | 一代一代推进的优化步。 |
| pareto archive | external archive / elite archive | 非支配解集合与精英集合。 |
| bias | heuristic prior / soft preference | 软引导，不替代硬约束。 |
| plugin / capability | callback / observer / runtime hook | 工程能力层，不改核心搜索语义。 |
| solver | algorithm execution control plane | 生命周期、上下文、插件调度和评估入口。 |
| solver group / regime | algorithm portfolio / hyper-heuristic orchestration | 多策略并行或串行编排，不等于单一算法。 |
| stage / serial / parallel | workflow composition / pipeline orchestration | 把多阶段优化流程显式化。 |
| resource context / lease | scheduler resource contract | 资源授权与执行上下文（传统库通常弱化这层）。 |
| artifact / checkpoint | run artifact / restart file / experiment record | 可复现、可审计的结果边界。 |

### 27.1 常见误解纠正

| 常见误解 | 正确说法 |
| --- | --- |
| `solver = 算法` | `solver` 是控制平面；算法策略主要在 `adapter`。 |
| `plugin = 算法逻辑` | `plugin` 主要是能力扩展（日志、归档、审计、监控、持久化）。 |
| `representation = 参数本身` | `representation` 是参数/结构的编码与合法空间定义。 |
| `problem = 只有 objective` | `problem` 还包含约束、数据视图、评估流程与反馈协议。 |
| `nsgablack 只适合 NSGA` | `nsgablack` 是优化搜索语义层，不绑定单一算法家族；跨 Case 编排属于共享 Project substrate。 |

### 27.2 面试一句话模板

```text
nsgablack separates optimization into explicit layers: problem defines evaluation semantics, representation defines the search space, adapter defines search strategy, solver controls the search lifecycle, and plugins provide runtime capabilities. Cross-Case workflow orchestration and resource grants belong to the shared Project substrate.
```
