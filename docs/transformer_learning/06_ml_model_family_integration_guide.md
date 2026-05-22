# 机器学习模型族统一接入指南：nsgablack + mlblack

这份文档不是把 `mlblack` 和 `nsgablack` 当成两个互相调用的工具，而是把它们看成一个统一框架栈的不同层。

核心结论：

```text
nsgablack + mlblack = 一个统一的智能建模与优化框架栈

nsgablack:
  外层控制、搜索、编排、资源、生命周期、审计、多目标 Pareto、嵌套求解

mlblack:
  机器学习语义、数据视图、模型规格、Head、Problem、Trainer、Artifact、Backend capability

domain / external backend:
  数值求解器、仿真器、数据库、向量索引、文件系统、对象存储、Ray/K8s/云服务
```

因此，接入一个模型族通常不是新增一整套 `mlblack Workflow`，而是判断它到底改变了哪一层：

```text
只换输出语义        -> 补 Head / Problem
只换输入结构        -> 补 DataView / Codec
结构也要搜索        -> 补 Spec / Representation，并交给 nsgablack 搜索
训练过程要阶段化    -> 用 nsgablack stage / serial / group 编排
每个候选触发内层训练 -> 用 nsgablack nested evaluation + mlblack inner flow
需要 GPU/并行/云端   -> 用 nsgablack L0 ResourceContext 注入，mlblack 只消费授权资源
```

禁止的方向：

```text
mlblack TimeSeriesWorkflow
mlblack PINNWorkflow
mlblack MultiModalRuntime
mlblack ResidualTrainer
mlblack SerialRuntime
```

应该建设的方向：

```text
TimeSeriesDataView
ForecastHead
PhysicsResidualProblem
NumericalSolverBridge
PredictionIOContract
IntegratedPredictionModel
ScientificArtifact
nsgablack stage / solver / resource orchestration
```

---

## 1. 统一框架栈的层边界

### 1.1 分层职责表

| 层 | 主要对象 | 负责什么 | 不负责什么 |
| --- | --- | --- | --- |
| nsgablack Solver | `SolverBase` / `EvolutionSolver` / `SolverManager` | 生命周期、评估入口、状态、插件调度、多 solver 编排 | 具体模型结构和训练细节 |
| nsgablack Adapter | NSGA2、MOEAD、VNS、DE、SA、serial、group、event | 搜索策略、候选生成、反馈吸收、阶段切换 | 数据清洗、模型 forward/backward |
| nsgablack Representation | genome、typed spec、decode、repair | 把外层候选解变成可执行配置或内层任务 | 直接训练模型 |
| nsgablack Problem / Evaluation bridge | outer objective、constraint、nested evaluator | 把候选转成内层任务，并把内层结果投影成外层目标/约束 | 偷偷解析内层私有对象 |
| nsgablack Plugin | report、trace、checkpoint、profile、archive、runtime graph | 审计、观测、持久化、短路评估、导出 | 改写业务目标语义 |
| nsgablack L0 | `TaskEnvelope`、`ResourceRequirement`、`ResourceContext`、`TaskResult` | 任务协议、资源授权、worker、lease、artifact refs | 代替 Ray/K8s，也不直接等于 CPU/GPU |
| mlblack DataView | tabular、time series、image、text、graph、多模态 | 把原始数据变成模型可消费结构 | 搜索策略和资源调度 |
| mlblack Spec / Codec | ModelSpec、NeuralGraphSpec、Codec、ModelBuilder | 把模型语义解码成可执行模型 | 外层多目标搜索 |
| mlblack Head | Regression、Classification、Forecast、Ranking、Distribution、Policy | 定义输出语义 | 数据加载和流程编排 |
| mlblack Problem / Trainer | loss、metric、residual、trainer、backprop、fitter | 训练、评估、梯度协议、单次模型拟合 | 管理外层 solver fanout |
| mlblack Artifact | model artifact、report、checkpoint、schema、resource audit | 保存可复现产物 | 作为大对象塞进 nsgablack context |
| 外部/领域后端 | 数值求解器、仿真器、DB、向量索引、对象存储 | 提供领域计算、存储或检索能力 | 替代框架层边界 |

### 1.2 一次完整的统一运行

```text
nsgablack outer solver
  -> adapter.propose()
  -> representation.decode(candidate)
  -> build inner task / component_overrides
  -> L0 acquire ResourceLease
  -> inject ResourceContext
  -> mlblack build DataView / ModelSpec / Head / Problem / Trainer
  -> optional domain backend: numerical solver / simulator / index
  -> mlblack returns metrics / artifact refs / audit
  -> nsgablack projects inner result to objectives / violations
  -> adapter.update()
  -> plugins write trace / graph / report / archive
```

关键点：

- `mlblack` 不是被 `nsgablack` 随便调用的脚本，而是内层 ML 语义与训练执行面。
- `nsgablack` 不是只做超参搜索，它负责外层多目标、阶段、嵌套、资源、报告、Pareto 和复杂工程编排。
- 大对象通过 artifact/snapshot ref 传递，不直接塞进 context 或 task payload。
- 如果存在父级 `ResourceContext`，`mlblack` 必须服从，不能自己写死 `cuda:0` 或重新抢全局线程池。

---

## 2. 已有方法能力盘点层：先查 primitive，再谈新增

这份文档容易被误读成“为了支持更多模型族，需要从零新增很多东西”。实际不是这样。`nsgablack + mlblack` 已经有一批方法能力 primitive，新增模型或能力前应先盘点当前已有能力，再决定是复用、补窄接口、补 bridge，还是上升到外层编排。

核心规则：

```text
不要先问：还缺哪个模型？
先问：这个模型/方法由哪些已有 primitive 组合而成？

如果已有 primitive 能表达：
  复用 + 补 Head / Problem / Artifact / capability contract

如果已有 primitive 只差一个局部算子：
  补 component / provider / pipeline operator

如果需要结构搜索、阶段编排、资源调度或嵌套评估：
  交给 nsgablack 外层 orchestration
```

### 2.1 当前已有核心 primitive

| 方法能力 | 当前已有落点 | 语义 |
| --- | --- | --- |
| 外层候选表示与搜索空间 | `nsgablack/representation/base.py::RepresentationPipeline` | `encoder / repair / initializer / mutator / crossover`，负责候选解从生成、变换、修复到解码的统一流转 |
| 常见外层表示类型 | `nsgablack/representation/continuous.py`、`integer.py`、`binary.py`、`permutation.py`、`matrix.py`、`graph.py`、`dynamic.py` | 连续、整数、二进制、排列、矩阵、图、动态表示已经是现有搜索空间 primitive |
| 数据对象数值化 | `mlblack/numericizer/base.py`、`default.py` | `BaseNumericizer` / `DefaultNumericizer` 把 object-first sample 转成数值训练矩阵，支持 modality encoder 与 target codec |
| target 编码 | `mlblack/numericizer/target_codec.py` | target 不是硬写成单一 `y`，可以通过 codec 表达不同输出语义 |
| 训练前表征变换 | `mlblack/pipeline/base.py` | `BasePipeline.fit/transform/fit_transform` 是模型前表征变换的正式接口 |
| 标准数值变换 | `mlblack/pipeline/zscore.py` | z-score 等常规预处理已经是 pipeline primitive |
| 符号/候选特征空间 | `mlblack/pipeline/feature_space.py` | candidate pool、primitive registry、temporal/regime feature pack、rolling split、interval metric 等已经有集中 facade |
| 可学习表征算子 | `mlblack/pipeline/learnable_conv.py` | `LearnableConv1DPipeline` 是 trainer-agnostic 的可学习 pipeline operator，可被外层注入参数 |
| 正交源/正交基结构 | `mlblack/core/orthogonal_source/layer.py` | 已有更高阶的基结构、正交性、source composition 能力，不应再被当成“缺一个无监督模块” |
| 外层参数注入内层组件 | `mlblack/core/orchestration/component_override.py` | `ComponentOverrideCapability` / `build_learnable_operator_capability` 支持 `nsgablack outer -> mlblack inner` 的安全参数注入 |
| 通用外层组件搜索 | `nsgablack/representation/base.py::RepresentationPipeline` + `component_overrides` + `build_inner_task()` / `evaluate_from_inner_result()` | 任何 `mlblack` 组件只要有稳定参数、结构描述和 result/artifact payload，就可以被外层搜索，不是某个模型族的特例 |
| 跨框架资源授权 | `ResourceContext` / L0 runtime surface | `nsgablack` 管外层资源授权，`mlblack` 只消费生效上下文并记录审计 |

这些 primitive 的存在意味着：很多“新模型族”其实不是新 workflow，而是已有能力的不同组合。

### 2.2 nsgablack 外层搜索是通用能力，不是模型族特例

`nsgablack` 的外层搜索/编排不应该被写成“时序、PINN、符号学习、多模态等场景下才介入”的特殊判断。更准确的规则是：

```text
任何 mlblack 组件，只要暴露稳定参数、结构描述、artifact/result payload，
就可以被 nsgablack 作为外层搜索对象。
```

统一协议是：

```text
outer genome
  -> RepresentationPipeline.decode(candidate)
  -> component_overrides / inner task payload
  -> mlblack build DataView / Pipeline / Spec / Head / Problem / Trainer
  -> inner result metrics + artifact refs + audit payload
  -> nsgablack projects result to objectives / violations
  -> adapter.update() / Pareto / report
```

这已经不是纯概念，现有 `learnable_conv_component_search` 就是这个模式：外层候选解码成 `pipeline.learnable_conv1d` 的 `component_overrides`，再调用 mlblack inner flow，并把 inner metrics 投影回外层目标。

因此，后续章节出现的 `window`、`lag`、`horizon`、`feature pack`、`head route`、`model family`、`resource budget`，都只是普通的可搜索组件参数。它们不代表 `nsgablack` 在某个模型族中特殊介入，而是统一外层搜索协议的自然使用。

### 2.3 文档中的模型族应被理解成组合结果

常见模型族不应被当成互相隔离的目录，而应被解释为 primitive 的组合。

| 表面模型/方法 | 更准确的组合解释 |
| --- | --- |
| 时序预测 | `TimeSeriesDataView + WindowSpec/LagSpec/HorizonSpec + TemporalFeaturePack + ForecastHead + TimeAwareProblem + ForecastArtifact`；`window/lag/horizon/feature pack/backtest` 只是可暴露给 nsgablack 的普通搜索面 |
| Transformer | `NeuralGraphSpec + attention/block Codec + task Head + BackpropTrainer`；block、depth、width、attention route、resource budget 都可作为通用外层搜索面 |
| PINN | `Neural function approximator + PhysicsResidualProblem + NumericalSolverBridge + nsgablack loss/collocation/solver-policy search` |
| Neural ODE | `Dynamics model + ODE solver bridge + trajectory Problem`；可微时走 mlblack 梯度协议，不可微或多目标时走 nsgablack 黑箱/嵌套搜索 |
| 多模态 | `multiple DataView/encoder branches + PredictionIOContract + fusion/gating Head + nsgablack branch/stage/resource orchestration` |
| 符号学习 | `SymbolicSpec / feature_space primitive + constant fitter + nsgablack expression/basis/complexity search` |
| 推荐/检索 | `UserItem/DataView + embedding/retrieval Head + ranking Problem + index artifact/provider`；负采样、embedding 维度、召回/排序组合可作为搜索面 |
| 不确定性/区间 | `Distribution/Interval Head + calibration Problem + artifact report`；误差、覆盖率、区间宽度、稳定性可进入 Pareto 目标 |

这样写的目的，是避免后续 Agent 看到一个模型名就新建一套 family/workflow。

### 2.4 新能力接入前的四步判断

每次接入新方法，先按以下顺序判断：

```text
1. 查已有 primitive
   是否已经能用 numericizer / pipeline / feature_space / head / trainer / provider 表达？

2. 判断缺的是局部语义还是外层编排
   局部语义 -> 补 mlblack component/head/problem/artifact
   外层搜索/阶段/资源 -> 补 nsgablack representation/problem/orchestration/L0

3. 判断是否需要 domain backend
   数值求解器、仿真器、数据库、向量索引、对象存储都走 provider/bridge，不塞进 trainer 或 solver。

4. 判断产物与审计
   是否需要 artifact ref、ResourceContext、capability matrix、runtime summary、nested report？
```

### 2.5 不要把已有能力误写成缺口

以下说法应避免：

```text
缺表征变换能力
缺特征工程能力
缺外层候选表示能力
缺可学习 pipeline operator
缺外层到内层参数注入机制
```

更准确的说法是：

```text
已有表征变换、候选表示、feature space、learnable operator、component override 等 primitive。
后续工作重点是：
  1. 把这些 primitive 在文档和 catalog 中标清楚；
  2. 针对新模型族补缺失的 Head / Problem / Provider / Artifact；
  3. 对复杂组合使用 nsgablack 编排，而不是新造 mlblack workflow。
```

---

## 3. 新模型族到底要补多少东西

### 3.1 轻量接入：只补 Head / Problem

适用：模型结构和数据结构都已有，只是输出语义不同。

例子：

```text
classification -> regression
point forecast -> interval forecast
regression -> quantile regression
score -> ranking probability
```

通常只需要：

```text
Head
Problem metric / loss
Artifact report fields
```

不需要新建：

```text
新的 workflow
新的 runtime
新的 solver
```

### 3.2 中量接入：补 DataView + Head + Problem

适用：输入结构改变，但训练生命周期仍是普通 fit/evaluate。

例子：

```text
time series
text sequence
image
graph
recommender
multi-output tabular
```

通常需要：

```text
DataView
Schema / Split policy
Codec route
Head
Problem / metrics
Artifact schema
```

nsgablack 只在需要搜索窗口、特征、模型族、backtest 策略、资源时介入。

### 3.3 重量接入：补 Spec + Codec + nsgablack 外层搜索

适用：模型结构本身就是优化对象。

例子：

```text
neural architecture search
symbolic regression
GAM basis search
PINN loss-weight/collocation search
surrogate + optimizer loop
multi-model composition
```

通常需要：

```text
Spec / typed genome
Representation decode
Codec / ModelBuilder
Problem / Evaluation bridge
nsgablack Adapter / stage / Pareto
Artifact / benchmark
```

### 3.4 工程级接入：嵌套求解 + L0 资源

适用：一个外层候选会触发昂贵的内层训练、仿真、求解或多模型对照。

例子：

```text
outer candidate -> inner mlblack training
outer supply policy -> inner production scheduling solver
outer physics config -> inner numerical simulation + neural training
outer model family search -> inner cross validation / backtest
```

必须显式设计：

```text
TaskEnvelope
ResourceRequirement
ResourceLease
ResourceContext
TaskResult
DataRef / artifact refs
namespace / run id / audit
```

---

## 4. 通用组件设计模板

每接入一个模型族，都先用下面这个模板判断落点。

### 4.1 DataView

DataView 负责把原始数据变成模型可消费结构。

它必须回答：

| 问题 | 示例 |
| --- | --- |
| 输入是什么 | tabular、image、text、graph、time series、multi-modal |
| batch 形状是什么 | `[batch, features]`、`[batch, time, features]`、`[batch, channel, height, width]` |
| target 在哪里 | `y`、future horizon、next token、reward、field value |
| split 怎么做 | random、time、group、rolling、episode、spatial |
| 泄漏风险是什么 | target leakage、future leakage、global normalization leakage、graph edge leakage |

DataView 不写搜索策略，不写资源调度。

### 4.2 Spec / Representation

Spec 描述可优化对象。

```text
LinearSpec:
  feature set, regularization, coefficient shape

NeuralGraphSpec:
  blocks, hidden dim, activation, normalization, head route

TimeSeriesSpec:
  context length, horizon, covariate policy, aggregation

PhysicsSpec:
  equation residual, boundary condition, collocation policy, solver config

SymbolicSpec:
  expression tree, function pool, constant slots
```

如果 Spec 要被搜索，它就是 nsgablack representation 的语义。

### 4.3 Codec / ModelBuilder

Codec 把 Spec 变成可执行模型。

```text
Spec -> ExecutableModel
```

同一语义可以有不同 backend：

```text
NeuralGraphSpec -> TorchModule
NeuralGraphSpec -> JaxFunction
NeuralGraphSpec -> NumpyFunction
```

Codec 必须声明 capability：

```text
forward
backward
dynamic_shape
sparse_ops
mixed_precision
distributed
artifact_export
```

### 4.4 Head

Head 定义输出语义。

| Head | 输出 | 常见任务 |
| --- | --- | --- |
| RegressionHead | scalar/vector | 回归、代理模型 |
| ClassificationHead | logits/probability | 分类 |
| ForecastHead | horizon values | 时序预测 |
| IntervalHead | lower/upper or quantiles | 区间预测 |
| RankingHead | score | 推荐/排序 |
| RetrievalHead | embedding | 检索 |
| DistributionHead | distribution params | 概率预测 |
| FieldHead | scalar/vector field | PINN/FNO/科学计算 |
| PolicyHead | action distribution | 强化学习 |

### 4.5 Problem / Evaluation

Problem 把模型和数据映射成反馈。

```text
model + data + domain backend -> feedback
```

反馈可以包含：

```text
loss
metrics
objectives
constraints
gradients
residuals
runtime cost
artifact refs
```

外层 nsgablack 只消费稳定 payload，不解析 inner trainer 私有对象。

### 4.6 Adapter / Trainer

Adapter 是更新策略，不是模型语义。

| 类型 | 所属层 | 用途 |
| --- | --- | --- |
| BackpropAdapter / Trainer | mlblack | 可微模型训练 |
| FunctionalBackpropAdapter | mlblack | 统一函数式梯度协议 |
| TreeFitterAdapter | mlblack | 树模型拟合 |
| SymbolicSearchAdapter | nsgablack 或 mlblack-inner | 表达式结构搜索 |
| HyperparameterSearchAdapter | nsgablack | 超参/结构/组件组合搜索 |
| NSGA2/MOEAD/VNS/DE/SA | nsgablack | 多目标、全局、局部、扰动搜索 |
| serial/group/event adapter | nsgablack | 阶段和策略编排 |

### 4.7 Artifact

Artifact 是正式产物，不是日志附属品。

最低应包含：

```text
model spec
codec route
backend capability
head
problem config
metrics
split policy
resource context
checkpoint refs
artifact refs
```

复杂模型还需要：

```text
tokenizer ref
feature schema
window policy
graph schema
physics equation / boundary condition
numerical solver config
uncertainty report
calibration report
runtime audit
```

---

## 5. 表格模型

典型模型：

```text
Linear / Logistic Regression
GAM
Decision Tree
Random Forest
XGBoost / LightGBM / CatBoost
TabNet
FT-Transformer
```

### 5.1 统一接入链路

```text
mlblack:
  TabularDataView
  FeatureSchema / PreprocessSpec
  TabularCodec
  RegressionHead / ClassificationHead / RankingHead / IntervalHead
  SupervisedProblem
  Fitter / BackpropTrainer / TreeFitter
  TabularArtifact

可暴露给 nsgablack 外层搜索的参数:
  feature subset search
  model family selection
  hyperparameter Pareto search
  accuracy vs complexity vs latency
  nested cross validation orchestration
```

### 5.2 通常需要补什么

| 情况 | 补什么 |
| --- | --- |
| 已有 tabular fit，只换任务 | Head + Problem metric |
| 加树模型 | TreeCodec + TreeFitter + Artifact exporter |
| 加低复杂度约束 | nsgablack outer objective：error、nonzero terms、latency |
| 加 AutoML | nsgablack representation：model family + hyperparams + preprocessing |

### 5.3 风险

- target leakage。
- train/test normalization 泄漏。
- categorical encoding 对测试集新类别处理不明。
- 分类不能只看 accuracy，要看 F1、AUC、calibration。
- 外层搜索不能只最大化训练集指标。

---

## 6. 神经网络与 Transformer 类模型

典型模型：

```text
MLP
CNN
RNN / LSTM / GRU
Transformer Encoder / Decoder / Encoder-Decoder
Graph Transformer
Vision Transformer
```

### 6.1 统一接入链路

```text
mlblack:
  NeuralGraphSpec
  Layer / Block / Route / Activation / Normalization
  NeuralCodec
  Task Head
  BackpropTrainer / FunctionalBackpropAdapter
  NeuralArtifact

可暴露给 nsgablack 外层搜索的参数:
  architecture search
  head route search
  loss weight search
  low-latency Pareto
  multi-stage baseline -> neural -> ensemble
  resource-aware GPU scheduling
```

### 6.2 关键判断

很多神经模型并不需要新增一个模型族，只需要在已有 NeuralGraph/Codec 上补：

```text
新的 block
新的 head
新的 loss
新的 data view
新的 artifact 字段
```

Transformer 的解耦也应该落在这些位置：

| Transformer 部件 | 推荐落点 |
| --- | --- |
| attention block | NeuralGraph block / Codec |
| positional encoding | Spec / Codec |
| encoder/decoder route | NeuralGraphSpec |
| LM / Seq2Seq / Classification 输出 | Head |
| mask / padding | DataView + Problem |
| generation config | Artifact / inference config |

### 6.3 nsgablack 适合做什么

```text
search number of layers
search hidden dimension
search attention heads
search FFN ratio
search activation / normalization
search sparse route / adapter route
optimize accuracy vs latency vs memory
choose backend under L0 ResourceContext
```

不要在 mlblack 里新增 `TransformerWorkflow`。应该让 mlblack 提供 Transformer 语义件，让 nsgablack 决定什么时候训练、怎么搜索、用多少资源。

---

## 7. 时序模型

时序模型要接入 `mlblack`，但不要实现成 `mlblack TimeSeriesWorkflow`。正确方向是补齐一组正式的时序语义组件，让它们和现有 numericizer、pipeline、trainer、artifact 协议兼容。

同时要明确：`nsgablack` 的作用不是“时序模型在某些情况下才介入”。`nsgablack` 已经是通用外层搜索/编排层；时序里的窗口、滞后阶数、horizon、feature pack、模型族、回测预算，只是可以暴露给外层搜索的普通组件参数。

典型模型只是这些能力的不同组合：

```text
ARIMA / Prophet / ETS
RNN / LSTM / GRU
TCN
N-BEATS
Temporal Fusion Transformer
Informer
PatchTST
State Space Model / S4-like model
temporal symbolic / lag-basis model
```

### 7.1 当前已有能力与缺口

时序不是从零开始。当前 `mlblack` 已经有部分时序 primitive，`nsgablack` 也已经有外层搜索、nested evaluation 和 L0 task/result 协议。

| 能力 | 当前落点 | 语义 |
| --- | --- | --- |
| lag 构造 | `mlblack/core/symbolic/feature_space/lag_utils.py::make_lag_from_history` | 从历史序列构造 `lag1/lag2/lagk`，可作为 tabularized time series 的基础 |
| rolling split | `mlblack/core/symbolic/feature_space/cv_splitter.py::build_rolling_splits` | 时间顺序验证，避免随机打乱和未来泄露 |
| temporal feature pack | `mlblack/core/symbolic/feature_space/temporal_feature_pack.py::TemporalFeaturePackConfig` / `TemporalFeaturePackResult` / `apply_temporal_feature_pack` | rolling mean/std、momentum、cross、ratio 等时间派生特征 |
| candidate feature space | `mlblack/pipeline/feature_space.py` | 把候选特征、temporal/regime feature pack、rolling split 等纳入统一 feature-space facade |
| 外层候选表示 | `nsgablack/representation/base.py::RepresentationPipeline` | 把 outer genome 解码为窗口、lag、feature pack、head、trainer route 或 inner task payload |
| nested evaluation | `nsgablack/core/nested_solver.py::TaskInnerRuntimeEvaluator` | 支持 `build_inner_task()` / `evaluate_from_inner_result()`，让一个 outer candidate 触发完整 inner training/backtest |
| L0 task/result | `nsgablack/core/resources/model.py::TaskEnvelope` / `TaskResult` / `ResourceRequirement` | 统一 worker、lease、resource context、artifact refs 和运行审计 |
| nested parallel/distributed | `nsgablack/utils/parallel/nested.py` | 线程和 Redis 路径已经对齐到 task/result 协议 |

当前真正缺的不是“能不能做时序”，而是把这些零散能力提升成正式 `mlblack` surface：

```text
TimeSeriesDataView
TimeIndexSpec / SeriesIdSpec
WindowSpec / LagSpec / CovariateSpec / HorizonSpec
TemporalFeaturePackOperator
ForecastHead / IntervalForecastHead / ProbabilisticForecastHead
TimeAwareProblem
ForecastArtifact
可选 classical/statistical ForecastProvider
可选 sequence neural Codec / ModelBuilder
```

### 7.2 mlblack 侧正式接入链路

第一版建议先做 v0/v1，也就是把时序变成可审计的监督学习问题，而不是一开始就做完整深度时序库。

```text
raw series table
  -> TimeSeriesDataView
  -> TimeIndexSpec / SeriesIdSpec
  -> WindowSpec / LagSpec / CovariateSpec / HorizonSpec
  -> TemporalFeaturePackOperator or existing feature_space primitive
  -> existing linear/tree/neural/symbolic trainer route
  -> ForecastHead / IntervalForecastHead
  -> TimeAwareProblem
  -> ForecastArtifact
```

注意：`TemporalTrainer` 不应默认新增。很多工业预测问题第一版可以通过 lag/rolling/temporal pack 转成监督矩阵，然后复用已有 linear、tree、neural、symbolic trainer。只有当原始序列窗口直接进入 RNN/TCN/Transformer/State Space 模型时，才需要补 sequence temporal codec 和对应 model builder。

### 7.3 最小正式组件

| 组件 | 最小职责 |
| --- | --- |
| `TimeSeriesDataView` | 按 `series_id + timestamp` 排序，声明 target、time index、known/unknown future covariates |
| `TimeIndexSpec` | 声明时间列、频率、时区、缺口策略、calendar 特征策略 |
| `SeriesIdSpec` | 声明单序列/多序列、层级序列、冷启动序列处理策略 |
| `WindowSpec` | 声明 `context_length`、`horizon`、`stride`、`min_history`、窗口对齐策略 |
| `LagSpec` | 声明 lag orders、lag sources、是否允许 target lag、forecast origin 前后边界 |
| `CovariateSpec` | 区分 static covariates、observed past covariates、known future covariates、unknown future covariates |
| `HorizonSpec` | 声明 single-step、multi-step、direct/recursive、多 target horizon layout |
| `TemporalFeaturePackOperator` | 包装现有 temporal feature pack，并纳入 `mlblack` pipeline/capability/audit 协议 |
| `ForecastHead` | 输出 `[n, horizon]` 或 `[n, horizon, target_dim]` 的点预测 |
| `IntervalForecastHead` | 输出 lower/upper 或 quantile grid，并对齐 coverage/width metric |
| `ProbabilisticForecastHead` | 输出分布参数或 sample forecast，用于概率预测和风险度量 |
| `TimeAwareProblem` | rolling split、future leakage guard、MAE/RMSE/MAPE/sMAPE/coverage/peak error 等指标 |
| `ForecastArtifact` | 保存窗口、horizon、split、feature schema、forecast table、backtest report、resource audit |

### 7.4 可暴露给 nsgablack 外层搜索的时序参数

这一节不是介入条件判断。它只是列出时序组件的搜索面。只要这些参数通过 `component_overrides` 或 inner task payload 稳定暴露，`nsgablack` 就可以像搜索 learnable conv、symbolic kernel、ETF lane 一样搜索它们。

外层候选可以解码为：

```text
candidate:
  dataview:
    series_grouping_policy
    calendar_feature_policy
  window:
    context_length
    horizon
    stride
    min_history
  lag:
    lag_orders
    lag_sources
    target_lag_enabled
  temporal_pack:
    rolling_enabled
    momentum_enabled
    cross_enabled
    ratio_enabled
    cross_quantiles
  model:
    family_route
    trainer_route
    head_type
  validation:
    rolling_folds
    rolling_window_policy
    validation_metric_set
  runtime:
    inner_budget
    worker_requirement
```

对应的 `component_overrides` 可以是：

```python
component_overrides = {
    "dataview.time_series": {
        "series_id_col": "item_id",
        "timestamp_col": "date",
        "calendar_feature_policy": "standard",
    },
    "pipeline.temporal_feature_pack": {
        "rolling_enabled": True,
        "momentum_enabled": True,
        "cross_enabled": False,
        "ratio_enabled": True,
    },
    "head.forecast": {
        "horizon": 7,
        "head_type": "interval",
    },
    "trainer.route": {
        "family": "tree_boosting",
        "preset": "fast_rolling",
    },
}
```

inner result 投影为外层目标/约束：

```text
objectives:
  validation_error
  forecast_instability
  interval_width_or_miscalibration
  runtime_seconds
  artifact_size_or_training_cost

violations:
  future_leakage_detected
  insufficient_history
  invalid_horizon_layout
  runtime_budget_exceeded
```

### 7.5 三种接入深度

| 深度 | 做法 | 适用 |
| --- | --- | --- |
| v0：tabularized time series | 生成 lag/rolling/temporal pack 特征后复用现有 tabular trainer | 快速支持多数工业预测问题 |
| v1：multi-horizon forecast | 增加 `ForecastHead`、`HorizonSpec`、horizon target codec、rolling artifact | 多步预测、区间预测、稳定回测 |
| v2：sequence neural model | 增加 temporal codec，接 RNN/TCN/Transformer/State Space | 原始序列窗口直接进模型 |
| v3：classical/statistical provider | ARIMA/Prophet/ETS 作为 provider 或 fitter route | 需要传统基线或可解释统计模型 |
| v4：outer-search surface | 将 v0-v3 的参数稳定暴露为 `component_overrides` / inner task payload | 多目标、昂贵回测、需要 Pareto 或资源编排 |

推荐顺序：先固化 v0/v1 的 `DataView + Spec + Head + Problem + Artifact`，再补 v2/v3。v4 不需要等到最后才“引入 nsgablack”；它只是当组件参数稳定后自然开放的外层搜索面。

### 7.6 时序特有规则

时序数据不能按普通监督学习随机打乱。必须显式记录：

```text
series id
calendar / timestamp column
history window
prediction horizon
forecast origin
known future covariates
unknown future covariates
rolling validation folds
forecast timestamp alignment
leakage guard policy
normalization fit scope
```

禁止：

```text
全局 normalization 用到未来数据
随机 train/test split 打乱时间
target lag 构造时穿越 forecast origin
known future covariates 和 unknown future covariates 不分
rolling validation 没有记录 fold 边界
forecast artifact 只保存 aggregate metric，不保存 forecast table
```

### 7.7 产物要求

`ForecastArtifact` 至少应记录：

```text
model spec / trainer route
series schema
time index spec
window spec
lag spec
covariate spec
horizon spec
split / rolling folds
metrics by horizon
metrics by series
forecast table ref
interval / quantile report if enabled
feature pack audit
effective ResourceContext if nested
inner task / component_overrides if used by outer search
artifact refs for model, forecast table and report
```

这样时序模型首先是 `mlblack` 的正式模型能力补齐；同时，任何稳定的时序组件参数都可以被 `nsgablack` 外层搜索/编排。这和其他模型族完全同构，不需要为时序单独发明 workflow 或 runtime。

---

## 8. 文本与序列模型

典型模型：

```text
HMM
CRF
BERT-like Encoder
GPT-like Decoder
T5-like Encoder-Decoder
Seq2Seq
Preference Model
```

### 8.1 统一接入链路

```text
mlblack:
  SequenceDataView
  TokenizerSpec / LabelSchema / MaskSpec
  SequenceCodec
  TokenHead / SequenceHead / LMHead / Seq2SeqHead / PreferenceHead
  SequenceProblem
  SequenceTrainer
  SequenceArtifact

可暴露给 nsgablack 外层搜索的参数:
  tokenizer / max length / chunk policy search
  prompt/template route search
  model size vs quality vs latency Pareto
  staged fine-tuning orchestration
  multi-dataset evaluation orchestration
```

### 8.2 特殊边界

| 对象 | 应放位置 |
| --- | --- |
| tokenizer | DataView / Artifact |
| padding / mask | DataView + Problem |
| next-token loss | LMHead + Problem |
| decoding temperature / top-p | inference config / Artifact |
| RLHF / preference | PreferenceHead + PairwiseProblem，复杂时外层编排 |

### 8.3 风险

- tokenizer 必须和模型 artifact 绑定。
- padding/mask 是语义，不是实现细节。
- generation 参数不是训练参数。
- 长文本 chunk 策略会改变评价口径。

---

## 9. 视觉模型

典型模型：

```text
CNN
ResNet
EfficientNet
Vision Transformer
UNet
YOLO-like Detector
Diffusion UNet
```

### 9.1 统一接入链路

```text
mlblack:
  ImageDataView
  ImageTransformSpec
  VisionCodec
  ClassificationHead / DetectionHead / SegmentationHead / GenerationHead
  VisionProblem
  VisionTrainer
  VisionArtifact

可暴露给 nsgablack 外层搜索的参数:
  augmentation policy search
  resolution vs accuracy vs latency Pareto
  backbone/head selection
  GPU resource scheduling
  multi-stage pretrain -> finetune -> calibrate
```

### 9.2 设计重点

| 部件 | 设计重点 |
| --- | --- |
| DataView | channel order、dtype、resize/crop、label shape |
| Codec | CNN/ViT/UNet/backbone builder |
| Head | classification、detection、segmentation、generation |
| Problem | loss、mAP、IoU、Dice、augmentation policy |
| Artifact | transform config、class map、weights、calibration |

### 9.3 风险

- train/eval augmentation 必须分开。
- normalization 和 resize 必须进入 artifact。
- segmentation/detection 的 label 不是普通 `y`。
- GPU 显存必须通过 ResourceContext/lease 审计。

---

## 10. 图模型

典型模型：

```text
GCN
GAT
GraphSAGE
GIN
Message Passing Neural Network
Graph Transformer
Knowledge Graph Embedding
```

### 10.1 统一接入链路

```text
mlblack:
  GraphDataView
  GraphSchema / EdgeSchema / MessagePassingSpec
  GraphCodec
  NodeHead / EdgeHead / GraphHead / LinkPredictionHead
  GraphProblem
  GraphTrainer
  GraphArtifact

可暴露给 nsgablack 外层搜索的参数:
  message passing depth search
  neighborhood sampling policy search
  graph split strategy comparison
  graph feature selection
  accuracy vs memory vs runtime Pareto
```

### 10.2 风险

- batch graph 和 single graph 的 shape 不同。
- split 可能按 node、edge、graph 或时间切。
- graph leakage 很隐蔽，例如测试边被 message passing 间接看到。
- sparse capability 必须在 backend matrix 中声明。

---

## 11. 推荐、排序与检索模型

典型模型：

```text
Matrix Factorization
Factorization Machine
Two-Tower
Wide & Deep
DIN
DLRM
CLIP-style Retrieval
Learning to Rank
```

### 11.1 统一接入链路

```text
mlblack:
  UserItemContextDataView
  UserItemEncoderSpec
  RecommenderCodec
  RetrievalHead / RankingHead / CalibrationHead
  RankingProblem
  RecommenderTrainer
  RecommenderArtifact

可暴露给 nsgablack 外层搜索的参数:
  negative sampling policy search
  embedding dimension search
  tower architecture search
  offline metric vs latency vs memory Pareto
  index refresh / artifact export orchestration
```

### 11.2 产物要求

推荐/检索 artifact 不只是模型权重，还要记录：

```text
user/item schema
embedding table ref
vector index ref
negative sampling policy
candidate generation policy
ranking calibration report
Recall@K / NDCG / AUC / calibration
```

### 11.3 风险

- 负采样策略会改变训练语义。
- offline metric 和 online metric 可能不一致。
- 向量索引是 artifact/backend，不是普通模型字段。

---

## 12. 概率模型、区间模型与生成模型

典型模型：

```text
Bayesian Regression
Gaussian Process
Quantile Regression
VAE
Normalizing Flow
Diffusion Model
Conformal Prediction
```

### 12.1 统一接入链路

```text
mlblack:
  ProbabilisticDataView
  DistributionSpec / LatentSpec / CalibrationSpec
  ProbabilisticCodec
  DistributionHead / IntervalHead / SamplerHead
  UncertaintyAwareProblem
  ProbabilisticTrainer
  UncertaintyArtifact

可暴露给 nsgablack 外层搜索的参数:
  calibration vs sharpness Pareto
  uncertainty method selection
  sampler steps vs quality vs runtime search
  conformal calibration split orchestration
  diffusion scheduler search
```

### 12.2 关键指标

概率模型不能只看点预测误差。

必须考虑：

```text
NLL
pinball loss
coverage
interval width
calibration curve
sample quality
runtime / sampler steps
```

### 12.3 风险

- calibration 是核心，不是附加项。
- sampling seed、backend、steps 必须记录。
- diffusion/flow 往往需要更强的 backend capability 和资源审计。

---

## 13. 符号学习与可解释模型

典型模型：

```text
Symbolic Regression
GAM
Rule List
Sparse Linear Model
Decision Rules
Equation Discovery
```

### 13.1 统一接入链路

```text
mlblack:
  SymbolicDataView
  SymbolicSpec / FunctionPoolSpec / BasisSpec
  SymbolicCodec
  ExpressionHead / RuleHead
  SymbolicProblem
  ConstantFitter / LocalFitter
  SymbolicArtifact

nsgablack:
  expression structure search
  function pool search
  basis composition search
  complexity vs accuracy Pareto
  constraints / monotonicity / sparsity search
```

### 13.2 为什么 nsgablack 很适合

符号学习天然是结构搜索问题：

```text
outer candidate:
  expression tree
  function pool
  allowed interactions
  sparsity pattern
  complexity budget

evaluate:
  mlblack fits constants / coefficients
  computes error / constraints / interpretability metrics
  exports canonical expression artifact
```

### 13.3 风险

- 表达式等价性不能只靠字符串。
- 常数拟合和结构搜索必须分层。
- complexity 要进入 objective 或 constraint。
- artifact 必须保存 canonical form、feature names、unit/scale。

---

## 14. 科学机器学习：PINN、Neural ODE、FNO、Surrogate

这一类最需要按统一框架理解。它不应该被写成 `mlblack PINNWorkflow`，而应该是：

```text
nsgablack outer orchestration
  + mlblack neural / residual / artifact semantics
  + numerical solver / simulator bridge
```

典型模型：

```text
PINN
Neural ODE
DeepONet
Fourier Neural Operator
Surrogate Model
Differentiable Simulator
Equation Discovery + Neural Residual
```

### 14.1 统一接入链路

```text
nsgablack outer:
  search network spec
  search residual weights
  search collocation policy
  search boundary penalty
  search numerical solver tolerance / step policy
  optimize data loss vs physics residual vs runtime vs stability
  allocate ResourceContext for inner training / simulation

mlblack inner:
  ScientificDataView
  PhysicsSpec / OperatorSpec
  NeuralGraphSpec / ScientificCodec
  FieldHead / OperatorHead
  PhysicsResidualProblem
  BackpropTrainer / FunctionalBackpropAdapter
  ScientificArtifact

domain backend:
  NumericalSolverBridge
  ODE/PDE integrator
  mesh / boundary / initial condition provider
  simulator / residual evaluator
```

### 14.2 PINN 怎么编排

```text
outer candidate:
  network_spec
  loss_weights:
    data_loss_weight
    residual_loss_weight
    boundary_loss_weight
  collocation_policy:
    points
    sampling_strategy
    refresh_interval
  numerical_policy:
    residual_method
    tolerance
    mesh_resolution
  training_policy:
    optimizer
    max_steps
    early_stop

evaluate candidate:
  nsgablack creates TaskEnvelope
  L0 acquires ResourceLease
  ResourceContext is injected into mlblack inner flow
  mlblack builds neural model + FieldHead
  NumericalSolverBridge computes residual / boundary violation
  trainer fits or fine-tunes model
  artifact records equation, boundary, residual report, resource audit
  result returns data_loss, physics_residual, boundary_violation, runtime
```

外层目标可以是：

```text
minimize data_error
minimize physics_residual
minimize boundary_violation
minimize runtime_seconds
minimize model_complexity
```

这正是 nsgablack 多目标/Pareto 的强项。

### 14.3 Neural ODE 怎么编排

Neural ODE 里的 ODE solver 不是一个普通 Head，也不是一个普通 trainer 参数。它是 evaluation graph 里的数值后端。

```text
mlblack:
  dynamics model f_theta(t, y, context)
  ODEStateDataView
  DynamicsSpec
  TrajectoryHead
  ODETrajectoryProblem
  differentiable or non-differentiable solver bridge

nsgablack:
  search dynamics architecture
  search integrator type
  search tolerance / step size / observation window
  search loss weights
  run nested training/evaluation
  manage resource and artifact refs

numerical backend:
  solve_ivp / torchdiffeq / custom solver / domain simulator
```

如果 ODE solver 可微，mlblack 的 `FunctionalBackpropAdapter` 可以统一梯度协议。  
如果不可微，nsgablack 可以把它当黑箱仿真后端做外层搜索或 surrogate-assisted optimization。

### 14.4 Scientific ML 要补的组件

| 组件 | 推荐设计 |
| --- | --- |
| `ScientificDataView` | coordinates、field、boundary、initial condition、unit、mesh |
| `PhysicsSpec` | equation、residual terms、boundary/initial condition、scaling |
| `NumericalSolverBridge` | 统一 ODE/PDE/simulator 调用和结果 shape |
| `PhysicsResidualProblem` | data loss、residual、boundary violation、stability |
| `FieldHead` / `OperatorHead` | scalar field、vector field、operator output |
| `ScientificArtifact` | equation、mesh、solver config、residual report、runtime audit |
| nsgablack representation | outer candidate 表达网络、loss、collocation、solver 策略 |
| nsgablack L0 | 给 inner training/simulation 注入 ResourceContext |

### 14.5 风险

- 物理残差不是普通 metric，通常是 objective 或 constraint。
- 单位、尺度、边界条件必须进入 artifact。
- 数值求解器误差和神经网络训练误差要分开记录。
- collocation sampling 会改变问题定义，必须可复现。
- 数值稳定性要进入报告，不能只看 loss。
- GPU/CPU 并行仿真必须通过 L0 lease 管理。

---

## 15. 强化学习模型

典型模型：

```text
Q-learning
DQN
Policy Gradient
PPO
Actor-Critic
SAC
Model-based RL
```

### 15.1 统一接入链路

```text
mlblack:
  EnvironmentDataView / TrajectoryDataView
  StateActionSpec
  PolicyValueCodec
  PolicyHead / ValueHead
  TrajectoryProblem
  RLTrainer / ReplayBuffer component
  PolicyArtifact

可暴露给 nsgablack 外层搜索的参数:
  environment parameter search
  reward weight search
  policy architecture search
  population-based training
  multi-seed evaluation orchestration
  safety constraint Pareto
```

### 15.2 为什么 RL 最后做

普通监督学习是：

```text
batch -> prediction -> loss
```

强化学习是：

```text
state -> action -> environment transition -> reward -> trajectory feedback
```

它改变了生命周期和反馈形态，不只是新增一个 Head。

因此 RL 的优先级应该低于时序、概率、科学 ML、模型组合。等 DataView、Problem、ResourceContext、Artifact 协议更稳后再做。

---

## 16. 多模态模型与模型组合

典型模型：

```text
CLIP
BLIP
ViLT
Text + Tabular
Image + Tabular
Stacking
Residual Model
Mixture of Experts
Main Model + Correction Model
```

### 16.1 统一接入链路

```text
mlblack:
  MultiModalDataView
  PredictionIOContract
  ModalitySpec
  ComponentModelArtifacts
  PredictionIntegrationComponent
  IntegratedPredictionModel
  ContrastiveHead / FusionHead / IntegratedPredictionHead
  CompositionProblem
  IntegratedModelArtifact

nsgablack:
  branch selection
  branch training order
  stage orchestration
  residual / stacking / correction target transformation
  fusion weights / router search
  resource-aware parallel branch training
```

### 16.2 I/O contract 是核心

多模态和模型组合不能假设所有 component 都吃同一个 `X`。

```text
text_model.predict(input_ids)
image_model.predict(image_tensor)
tabular_model.predict(tabular_features)
residual_model.predict([tabular_features, main_prediction])
```

所以必须显式声明：

```text
component input key
input kind / ndim / feature count
component output shape
row alignment rule
fusion strategy
final output contract
```

这就是 `PredictionIOContract` 的职责。

### 16.3 不要新增 workflow

不要新增：

```text
HybridTrainer
ResidualWorkflow
MultiModalWorkflow
mlblack SerialRuntime
```

应该新增或复用：

```text
PredictionIOContract
PredictionIntegrationComponent
ModelConditionedTargetComponent
IntegratedPredictionModel
nsgablack serial / group / resource orchestration
```

---

## 17. 后端能力、L0 资源与模型族的关系

### 17.1 模型 backend 与运行 backend 不是一回事

```text
模型 backend:
  torch / numpy / sklearn / xgboost / jax
  负责 forward/backward/fit/predict

运行 backend:
  local thread / process / Redis worker / Ray / K8s / cloud batch
  负责在哪里执行 task

资源:
  CPU threads / GPU tokens / RAM / GPU memory
  被 ResourceRequirement 声明，被 ResourceLease 授权
```

不要把 `cuda:0`、Ray、Redis、torch 混成一个 `backend` 字段。

### 17.2 Capability matrix

每个模型族要声明后端能力需求：

| 能力 | 示例 |
| --- | --- |
| forward | 能否执行推理 |
| backward | 能否反向传播 |
| dynamic shape | 是否支持变长输入 |
| sparse / graph | 是否支持 graph/sparse ops |
| mixed precision | 是否支持 fp16/bf16 |
| differentiable solver | 是否支持可微数值积分 |
| distributed | 是否支持多设备/多机 |
| artifact export | 是否能导出权重、结构、schema |

如果需求超过 backend，要 fail-fast：

```text
GraphTransformer requires sparse graph attention.
Current backend numpy does not provide sparse_graph_attention.
Use torch backend or select a simpler graph route.
```

### 17.3 nsgablack L0 与 mlblack L0 的关系

```text
nsgablack L0:
  owns outer solver fanout
  owns ResourceLease truth
  owns TaskEnvelope / TaskResult
  injects ResourceContext into inner runtime

mlblack L0:
  consumes ResourceContext
  clamps trainer/device/thread/backend usage
  reports effective resource context
  does not acquire parent resources privately
```

嵌套时必须这样传：

```text
outer ResourceLease
  -> ResourceContext
  -> mlblack FlowAssemblySpec / TrainerAssemblySpec
  -> provider / trainer / artifact audit
```

---

## 18. 推荐落地优先级

### P0：已经接近主线，应固化

```text
Tabular
NeuralGraph: MLP / CNN / Transformer basic blocks
Symbolic learning
Model composition / PredictionIOContract
ResourceContext propagation
```

### P1：下一阶段最值得补

```text
TimeSeries
Probabilistic / Interval / Calibration
Recommender / Retrieval
Scientific ML minimal bridge: PhysicsResidualProblem + NumericalSolverBridge
```

### P2：价值高但工程更重

```text
Graph model full support
MultiModal
PINN / Neural ODE full nested orchestration
Diffusion / generative models
```

### P3：最后做

```text
RL
large-scale distributed training
production Ray/K8s backend adapters
```

原因：P3 会明显扩展生命周期、资源、状态、容错和环境接口，不适合在基础协议未稳定时硬上。

---

## 19. 新模型族接入 checklist

### 19.1 先判断改哪层

```text
[ ] 只是换输出？补 Head / Problem
[ ] 只是换输入？补 DataView / Codec
[ ] 结构需要搜索？补 Spec / Representation，并交给 nsgablack
[ ] 每个候选要训练/仿真？补 nested evaluation bridge
[ ] 需要并行/GPU/云端？补 ResourceRequirement / ResourceContext / Artifact refs
[ ] 需要数值求解器？补 NumericalSolverBridge，不要塞进 Head
```

### 19.2 mlblack 必备

```text
[ ] DataView
[ ] Spec / ModelConfig
[ ] Codec / ModelBuilder
[ ] Head
[ ] Problem / Evaluation
[ ] Trainer / Adapter
[ ] Artifact
[ ] Backend capability contract
[ ] Smoke case
[ ] Benchmark case
```

### 19.3 nsgablack 必备

当使用外层编排时：

```text
[ ] Representation decode -> component_overrides / inner task
[ ] Problem/Evaluation bridge -> stable inner result payload
[ ] objectives / violations projection
[ ] adapter / serial / group / event 编排
[ ] ResourceRequirement
[ ] ResourceContext injection
[ ] artifact refs / snapshot refs
[ ] runtime graph / trace / report
[ ] Pareto / archive / benchmark fields
```

### 19.4 产物与审计

```text
[ ] model artifact 可复现
[ ] data split / schema 记录完整
[ ] backend capability 记录完整
[ ] resource context 记录完整
[ ] 数值求解器 config 记录完整
[ ] inner/outer run id 和 namespace 清楚
[ ] 大对象通过 DataRef / artifact ref 传递
[ ] 不把 private trainer object 暴露给 outer solver
```

---

## 20. 最重要的架构原则

```text
不要为了一个模型族发明一套 workflow。
先判断它改变的是 DataView、Head、Problem、Spec、Codec、Trainer、Artifact，还是 nsgablack 编排。

不要让 mlblack 管外层资源和多阶段搜索。
它只消费 ResourceContext，执行 ML 语义。

不要让 nsgablack 硬编码 mlblack trainer 细节。
它只传 component_overrides、inner task、ResourceContext，并消费稳定 result payload。

不要把数值求解器塞进模型本体。
它是 domain backend / evaluation bridge，可以被 nsgablack 编排，也可以被 mlblack Problem 消费。
```

按这个边界，时序、图、多模态、PINN、Neural ODE、符号学习、推荐、概率模型都可以接入；区别只是补的层不同、编排深度不同、资源审计强度不同。





