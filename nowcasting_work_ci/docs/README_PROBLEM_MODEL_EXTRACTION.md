# problem_model 拆分中文架构设计说明

本文档用于系统记录 `nowcasting_work_ci` 中 `mlblack_side/problem/problem_model.py` 的持续拆分过程、当前边界、公共组件职责，以及后续标准化路线。

这不是一份简短的变更记录，而是一份面向后续继续重构、代码评审、架构对齐的设计说明。

---

## 1. 文档目的

这份文档主要回答四个问题：

1. 为什么要拆 `problem_model.py`
2. 这一轮具体拆掉了什么，分别拆到了哪里
3. 现在 `problem_model.py` 还剩什么职责，为什么还保留在那里
4. 下一步应该沿着什么路线继续标准化

从更高层看，这份文档服务于两个目标：

- 让 `nowcasting_work_ci` 场景层不断变薄，尽量只保留数据与实验装配
- 让 `mlblack` 逐渐形成自己的公共 symbolic feature/evaluation 组件层，最终具备类似标准脚手架的装配能力

---

## 2. 背景与问题定义

### 2.1 原始问题

在早期版本中，`problem_model.py` 实际上承担了远超“Problem 壳”应有范围的职责。它同时包含：

- Problem 契约定义
- 决策向量解码
- rolling fold 评估
- strict4/global 分支评估
- branch fallback 与 holiday union 逻辑
- batch 设计矩阵构造
- batched ridge 预测
- symmetric residual 区间构造
- batch 区间指标计算
- fold 汇总与 detail 报告生成
- subset 元数据组装
- cache key 生成与本地缓存

从工程角度看，这意味着一个文件里既有：

- 业务层逻辑
- 编排层逻辑
- 公共评估逻辑
- 数值工具逻辑
- 报告生成逻辑

这样的结构在功能刚起步时是可以接受的，因为它开发速度快、定位集中；但当机制开始增多以后，会带来明显问题。

### 2.2 原始结构带来的问题

原始 `problem_model.py` 的问题主要体现在下面几个方面。

#### 1. 职责混杂

Problem 层本来应当主要负责：

- 决策编码/解码
- 调用评估
- 输出目标值

但文件里混入了很多并不属于 Problem 本体的逻辑，例如：

- branch evaluator
- fold report aggregator
- subset descriptor
- batch 数值工具

这会让“场景逻辑”和“公共逻辑”的边界非常模糊。

#### 2. 复用困难

如果另一个任务也需要：

- strict4 风格的 regime routing
- batch ridge 评估
- fold summary 聚合
- subset 元数据描述

那么原先只能复制粘贴 `problem_model.py` 里的片段，而不是直接复用公共组件。

#### 3. 维护成本高

当一个文件同时拥有：

- 数据协议
- 数学实现
- 评估编排
- 报告汇总

任何一个地方出错，都需要在大文件里来回跳转。后续再加入新的 interval method、branch policy、objective preference 时，文件会继续膨胀。

#### 4. 不利于形成 mlblack 标准层

你一直强调的一点是对的：

- `nsgablack` 有相对明确的标准装配思路
- `mlblack` 也需要逐步形成自己的标准公共层

如果所有 symbolic interval 逻辑都长期留在单个场景 `problem_model.py` 里，`mlblack` 永远长不出稳定的公共基础设施。

---

## 3. 本轮改造的总体目标

本轮改造的目标不是“重写问题层”，而是更克制、更稳妥地做三件事：

1. 识别哪些逻辑已经足够通用，可以从场景层上提
2. 把这些逻辑放到 `core/symbolic/feature_space` 公共层
3. 让 `problem_model.py` 逐步变成一个真正的薄壳

这里所谓的“薄壳”，不是说它什么都不做，而是说它应当主要承担：

- 解码
- 委托
- 缓存
- Problem 契约适配

也就是：

- 负责把优化器给出的向量解释成可评估对象
- 负责调用公共评估器和公共聚合器
- 负责管理本 Problem 的本地缓存
- 负责满足 `BlackBoxProblem` 接口要求

而不再自己展开数值工具、报告汇总、subset 描述等实现细节。

---

## 4. 改造前后对比

### 4.1 改造前的 problem_model

改造前，`problem_model.py` 同时包含如下代表性逻辑：

- `_eval_fold_global`
- `_eval_fold_strict4`
- `_design_matrix_for_genome`
- `_batched_ridge_predict`
- `_symmetric_interval_batch`
- `_interval_metrics_batch`
- 大段 fold mean / std / drift 汇总代码
- 大段 detail 字段拼装代码
- `subset_candidates` 的重复组装代码

这些代码混在一起时，文件在语义上像一个“完整评估系统”，而不是一个 Problem。

### 4.2 改造后的 problem_model

改造后，`problem_model.py` 的内部结构已经明显变化：

- strict4/global fold 执行细节不再内联，而是委托公共 branch evaluator
- fold 汇总与 detail 拼装不再内联，而是委托公共 fold report
- subset 元数据与 genome 组装不再重复手写，而是委托公共 subset descriptor
- design matrix / batched ridge / symmetric interval / batch metrics 不再内联，而是委托公共 batch interval utils

因此，Problem 文件的语义已经从：

- “把所有逻辑都装进去的实现文件”

转变为：

- “面向 solver 的 Problem 壳 + 一层委托编排”

### 4.3 对比总结

可以用下面的简化对比来理解。

改造前：

```text
problem_model
  = decode
  + branch evaluation implementation
  + batch numerical implementation
  + fold reporting implementation
  + subset descriptor implementation
  + cache
  + BlackBoxProblem interface
```

改造后：

```text
problem_model
  = decode
  + evaluator delegation
  + aggregator delegation
  + descriptor delegation
  + cache
  + BlackBoxProblem interface
```

这种变化的价值不在于“代码行数变少了多少”，而在于职责边界开始稳定下来。

---

## 5. 当前已经上提的公共组件

本轮新增与前期累计形成的公共层主要位于：

- `C:\Users\hp\Desktop\mlblack\core\symbolic\feature_space`

这部分已经不再只是一些分散工具，而是在逐渐形成一个完整的 symbolic feature/evaluation 公共子系统。

### 5.1 Branch Evaluator

文件：

- `branch_evaluator.py`

#### 核心职责

这个组件负责统一处理“一个 fold 到底怎么评估”的问题，尤其是：

- global fold evaluation
- strict4 branch fold evaluation
- strict4/global 在 symmetric residual 模式下的 batch 评估
- branch 样本不足时的 fallback 与 holiday union 训练选择

#### 为什么它应该被上提

因为这部分逻辑本质上不是 nowcasting 场景专有的，而是“带 regime router 的 interval evaluator”这一类任务的共性逻辑。

Problem 层只需要知道：

- 当前要评估哪个 fold
- 使用什么配置
- 训练回调是什么
- interval 构造回调是什么

至于内部如何：

- 先跑 global
- 再按 regime 分支
- 样本不足时如何 fallback
- holiday union 何时接管

这些都不应再在 Problem 壳里展开。

#### 组件内的职责拆分

可以把它理解成四层：

1. 输入层
- 接收 `X_fit / y_fit / tr_idx / va_idx`
- 接收 genome 与 branch 配置
- 接收训练与区间构造回调

2. 路由层
- 判断是否启用 strict4
- 生成 regime index
- 决定 self / holiday_union / fallback

3. 执行层
- 跑 global fit
- 跑 branch fit
- 合并各分支预测结果

4. 输出层
- 返回 fold 级别 summary
- 产出 branch detail 与 interval info

#### 职责图

```text
problem_model
  -> branch_evaluator
       -> 读取 strict4 配置
       -> 构造 regime index
       -> 计算 self / holiday_union / fallback
       -> 执行 global 或 branch fit
       -> 生成 fold 结果
```

#### 对 Problem 层的影响

`problem_model.py` 中原本庞大的：

- `_eval_fold_global`
- `_eval_fold_strict4`
- batch strict4/global 路由逻辑

现在已经变成“薄委托 + 回调提供者”。

---

### 5.2 Fold Report Aggregator

文件：

- `fold_report.py`

#### 核心职责

这个组件负责把多个 fold 的评估结果聚合成：

- 最终 `objective`
- 最终 `detail`
- fold 指标汇总
- RMSE mean / std / drift
- family / feature concentration
- branch detail 与 interval info 归档

#### 为什么它应该被上提

因为“fold 结果怎么聚合成 summary”本质上不是某个具体场景独有的逻辑，而是 interval symbolic task 的通用 reporting 问题。

原先 `problem_model.py` 中有很多这种逻辑：

- `coverage_error_mean`
- `pinaw_mean`
- `interval_score_mean`
- `picp_mean`
- `mean_width_mean`
- `rmse_mean`
- `rmse_std`
- `rmse_drift`
- `family_concentration`
- `feature_concentration`
- `selection_meets_coverage_threshold`

这些都不应继续由 Problem 文件手写维护。

#### 它解决的工程问题

这个组件把“评估执行”和“评估汇总”分离开来。

换句话说：

- branch evaluator 解决“怎么评估”
- fold report 解决“怎么把评估结果组织成可比较、可输出、可写 summary 的对象”

这两个层次一旦分开，后续如果你要改：

- objective schema
- summary 字段
- coverage threshold 逻辑
- 报告格式

就不需要再回到 Problem 壳中改大段代码。

#### 职责图

```text
fold_results
  -> fold_report
       -> 计算 mean / std / drift
       -> 计算 concentration
       -> 组装 objective
       -> 组装 detail
```

#### 对 Problem 层的影响

Problem 层不再自己写“从一堆 fold 数组到 objective/detail”的长逻辑，而是改为：

- 收集 `fold_results`
- 调 `build_interval_subset_report(...)`
- 获得 `obj, detail`

---

### 5.3 Subset Descriptor

文件：

- `subset_descriptor.py`

#### 核心职责

这个组件负责把：

- `candidates`
- `subset_idx`

统一转换成：

- `genome`
- `subset_candidates`

#### 为什么它应该被上提

因为这类逻辑虽然看起来简单，但非常容易在多个地方重复出现，而且一旦字段约定不一致，会引起非常隐蔽的问题。

例如原先 Problem 层里反复手写：

- `{"name": ..., "expr": ...}`
- `{"name": ..., "family": ..., "complexity": ..., "features": ...}`

这些写法如果散落多处，后面字段一变，就会出现多个位置要同步改。

#### 它解决的工程问题

这个组件实际上在做一件很重要的事：

- 统一 subset 的描述协议

这样以后别的 symbolic subset 任务，只要候选项对象协议一致，就可以复用：

- genome 生成逻辑
- subset metadata 生成逻辑

#### 职责图

```text
candidates + subset_idx
  -> subset_descriptor
       -> genome
       -> subset_candidates metadata
```

#### 对 Problem 层的影响

Problem 层不再需要知道：

- genome 的字典怎么拼
- subset_candidates 的字段列表怎么拼

它只需要拿 descriptor 输出结果继续往下传。

---

### 5.4 Batch Interval Utils

文件：

- `batch_interval_utils.py`

#### 核心职责

这个组件负责 Problem 中原本最“数值工具化”的部分：

- 设计矩阵构造
- batched ridge 预测
- symmetric residual 区间批处理
- batch interval metrics 计算

#### 为什么它应该被上提

因为这些逻辑本质上已经不是业务逻辑，而是 batch 数值工具逻辑。把它们放在 Problem 文件里，会导致：

- Problem 层显得像数值库
- 后续别的任务即使想复用，也只能从场景文件复制代码

#### 四个核心函数的意义

1. `as_2d(...)`
- 统一二维化输入
- 保证内部 shape 契约稳定

2. `design_matrix_for_genome(...)`
- 从 genome 和 `X` 构建设计矩阵
- 支持 graph cache 与无 cache 两种路径

3. `batched_ridge_predict(...)`
- 按不同 term count 分组做 batched ridge
- 有 torch 时走批量线性代数
- 无 torch 时回退逐 candidate ridge

4. `symmetric_interval_batch(...)`
- 在 residual 对称区间模式下，批量构造上下界

5. `interval_metrics_batch(...)`
- 批量计算 PICP / PINAW / interval score / coverage error

#### 职责图

```text
genome / X / y
  -> batch_interval_utils
       -> design matrix
       -> batched ridge predict
       -> symmetric interval batch
       -> interval metrics batch
```

#### 对 Problem 层的影响

Problem 层已经不再包含：

- `_design_matrix_for_genome`
- `_batched_ridge_predict`
- `_symmetric_interval_batch`
- `_interval_metrics_batch`

这意味着 Problem 文件不再承载 batch 数学实现。

---

## 6. 本轮改造与前期公共层的关系

除了本轮新增的 4 个组件，前期已经上提了一批更偏“特征空间”构造的公共组件，包括：

- `temporal_feature_pack.py`
- `regime_feature_pack.py`
- `activation_config.py`
- `primitive_registry.py`
- `generation_grammar.py`
- `candidate_pool.py`
- `feature_bundle.py`
- `builder.py`
- `regime_router.py`
- `lag_utils.py`

如果把这些组件与本轮新增内容放在一起看，可以发现 `core/symbolic/feature_space` 正在形成两个大区块：

### 6.1 特征空间构造区块

负责回答：

- 候选池怎么建
- grammar 怎么扩张
- primitive family 怎么注册
- temporal/regime 特征怎么加入
- full candidate pool 怎么生成

### 6.2 评估与报告区块

负责回答：

- fold 怎么评估
- branch 怎么路由
- batch interval 怎么算
- fold 结果怎么聚合
- subset 怎么描述

这意味着 `feature_space` 已经不只是“feature engineering”，而是在向更广义的“symbolic feature + symbolic evaluation support layer” 演进。

---

## 7. 当前 problem_model 还剩哪些函数与职责

虽然 `problem_model.py` 已经明显变薄，但它仍然保留了一些核心能力。这些能力并不都是坏味道，其中不少确实属于 Problem 壳该保留的部分。

### 7.1 初始化与 Problem 契约

典型函数：

- `__init__(...)`

职责：

- 持有 `X_fit / y_fit / candidates`
- 定义 `dimension / bounds / objectives`
- 初始化 splits 与 cache
- 初始化 bridge
- 满足 `BlackBoxProblem` 构造需求

为什么应保留：

- 这是 Problem 实例的身份定义
- 不应轻易外提

### 7.2 决策解码

典型函数：

- `_decode(...)`

职责：

- 从连续向量解码出 subset
- 处理 family bias
- 处理 threshold 选择
- 处理 interaction cap
- 产出 subset 与 decode meta

为什么应保留：

- 这是具体 Problem 的决策编码本体
- 属于 Problem 的领域定义，而不是通用工具

### 7.3 interval 边界构造桥接

典型函数：

- `_build_interval_bounds(...)`

职责：

- 在 Problem 内统一 native quantile 与 symmetric residual 两条 interval 路径
- 对 evaluator 提供统一的 interval 边界构造回调

为什么目前保留：

- 它仍然强绑定当前 Problem 的 interval method 选择
- 后续可以继续抽象，但目前作为桥接层是合理的

### 7.4 fold 摘要桥接

典型函数：

- `_summarize_fold(...)`

职责：

- 将 evaluator 输出结果归一化成 fold summary 协议
- 支持既用即时计算，也用预计算 interval metrics

为什么目前保留：

- 它是 Problem 与公共 evaluator / aggregator 之间的协议适配点
- 目前仍适合作为 Problem 内部桥接函数存在

### 7.5 evaluator 回调装配

典型函数：

- `_branch_eval_config(...)`
- `_fit_predict(...)`

职责：

- 给公共 evaluator 提供配置对象
- 给公共 evaluator 提供训练回调

为什么目前保留：

- 这部分本质上是“Problem 如何接入公共 evaluator”的适配层
- 还不是纯通用组件，但已经是壳层而不是实现层

### 7.6 单点 / 批量 Problem 评估入口

典型函数：

- `_evaluate_subset(...)`
- `_evaluate_decoded(...)`
- `_evaluate_decoded_batch(...)`

职责：

- 处理 cache 命中
- 调用 subset descriptor
- 调用 branch evaluator
- 调用 fold report
- 返回 objective

为什么应保留：

- 这是 Problem 壳最核心的职责之一
- 只是内部实现需要尽量薄，不应继续承载底层工具逻辑

### 7.7 cache 相关逻辑

典型函数：

- `cache_top(...)`

职责：

- 通过公共 cache key 协议读取与写入 cache
- 以当前 Problem 的口径管理 cached results
- 输出按目标偏好排序后的 top cache rows

为什么目前仍值得关注：

- cache key 协议本身已经抽到公共层
- 当前 Problem 内仍然保留 cache 持有与 cache 结果视图输出
- 后续如果继续标准化，可以再考虑把 cache view / cache repository 进一步抽象

### 7.8 目标偏好策略

典型函数：

- 当前 Problem 通过公共 objective policy 调用排序逻辑
- `cache_top(...)` 中使用 `interval_objective_sort_key(...)`

职责：

- 使用统一的 coverage error 与 coverage-threshold-first 排序策略
- 保持 Problem 输出结果与公共 objective preference policy 一致

为什么这一块仍值得关注：

- objective policy 本身已经从 Problem 中抽出
- 但 Problem 仍保留 policy 的调用点与阈值配置
- 后续如果要做更彻底的策略切换，可继续把 policy 装配与配置边界再标准化

---

## 8. 当前系统的调用链与边界图

目前整体调用关系可以概括为：

```text
run / workflow
  -> build_problem(...)
       -> problem_model
            -> decode
            -> subset_descriptor
            -> branch_evaluator
                 -> batch_interval_utils
            -> fold_report
            -> cache
            -> BlackBoxProblem result
```

如果进一步展开，可以理解为下面四层：

### 第 1 层：场景装配层

位于：

- `nowcasting_work_ci/run.py`
- `nowcasting_work_ci/run_solver.py`
- `nowcasting_work_ci/runtime/...`

职责：

- 读取数据
- 组装特征与候选池
- 构造 Problem
- 启动 solver

### 第 2 层：Problem 壳层

位于：

- `mlblack_side/problem/problem_model.py`

职责：

- 解码
- 缓存
- 委托公共组件
- 满足 solver 接口

### 第 3 层：公共评估/报告支持层

位于：

- `core/symbolic/feature_space/branch_evaluator.py`
- `core/symbolic/feature_space/fold_report.py`
- `core/symbolic/feature_space/subset_descriptor.py`
- `core/symbolic/feature_space/batch_interval_utils.py`

职责：

- 评估编排
- 数值工具
- subset 描述
- 报告聚合

### 第 4 层：更底层的 symbolic 与模型能力

位于：

- `core/symbolic/...`
- `model/interval_fit.py`

职责：

- symbolic expression evaluation
- ridge / three-layer fit
- interval model fitting

这个分层说明：

- `problem_model` 已经从“实现中心”变成“中间壳层”
- 真正可复用的复杂逻辑正在被推向公共层

---

## 9. 函数级迁移台账

这一节用于建立一张“函数迁移总表”，回答下面这个非常具体的问题：

- `problem_model.py` 原来有哪些函数或逻辑块
- 这些函数现在已经迁到了哪个公共组件
- 哪些已经完全迁出
- 哪些还只是桥接层
- 哪些原则上就应该保留在 Problem 内

这张表的意义在于：后续继续重构时，不需要再靠记忆判断“这段逻辑是不是已经抽过了”，而是可以直接对照台账推进。

### 9.1 迁移总表

| `problem_model.py` 原函数/逻辑 | 现在迁到哪个公共组件 | 当前状态 |
| --- | --- | --- |
| `_eval_fold_global(...)` | `branch_evaluator.py::evaluate_global_fold(...)` | 已迁出 |
| `_eval_fold_strict4(...)` | `branch_evaluator.py::evaluate_strict4_fold(...)` | 已迁出 |
| batch strict4/global 路由逻辑 | `branch_evaluator.py::evaluate_symmetric_residual_fold_batch(...)` | 已迁出 |
| `_design_matrix_for_genome(...)` | `batch_interval_utils.py::design_matrix_for_genome(...)` | 已迁出 |
| `_batched_ridge_predict(...)` | `batch_interval_utils.py::batched_ridge_predict(...)` | 已迁出 |
| `_symmetric_interval_batch(...)` | `batch_interval_utils.py::symmetric_interval_batch(...)` | 已迁出 |
| `_interval_metrics_batch(...)` | `batch_interval_utils.py::interval_metrics_batch(...)` | 已迁出 |
| fold mean / std / drift 汇总逻辑 | `fold_report.py::build_interval_subset_report(...)` | 已迁出 |
| detail 报告字段拼装 | `fold_report.py::build_interval_subset_report(...)` | 已迁出 |
| `genome` 组装逻辑 | `subset_descriptor.py::build_subset_genome(...)` | 已迁出 |
| `subset_candidates` 元数据组装 | `subset_descriptor.py::build_subset_candidate_metadata(...)` | 已迁出 |
| `genome + subset_candidates` 联合描述装配 | `subset_descriptor.py::build_subset_descriptor(...)` | 已迁出 |
| `_rolling_splits(...)` | `cv_splitter.py::build_rolling_splits(...)` | 已迁出 |
| `_cache_sig(...)` | `evaluation_cache_key.py::build_meta_signature(...)` / `build_subset_meta_cache_key(...)` | 已迁出 |
| `_coverage_error(...)` | `objective_policy.py::coverage_error(...)` | 已迁出 |
| `_objective_sort_key(...)` | `objective_policy.py::interval_objective_sort_key(...)` | 已迁出 |
| `_build_interval_bounds(...)` | 目前仍在 Problem 内，作为 interval method 桥接层 | 部分保留 |
| `_summarize_fold(...)` | 目前仍在 Problem 内，作为 evaluator/aggregator 协议桥接层 | 部分保留 |
| `_branch_eval_config(...)` | 目前仍在 Problem 内，作为 evaluator 配置装配层 | 部分保留 |
| `_fit_predict(...)` | 目前仍在 Problem 内，作为训练回调适配层 | 部分保留 |
| `_evaluate_subset(...)` | 仍在 Problem 内，负责单点评估编排与 cache | 保留在 Problem |
| `_evaluate_decoded(...)` | 仍在 Problem 内，作为 bridge 单点评估入口 | 保留在 Problem |
| `_evaluate_decoded_batch(...)` | 仍在 Problem 内，作为 bridge 批量评估入口 | 保留在 Problem |
| `evaluate_population_batch(...)` | 仍在 Problem 内，满足 `BlackBoxProblem` 批量接口 | 保留在 Problem |
| `evaluate(...)` | 仍在 Problem 内，满足 `BlackBoxProblem` 单点评估接口 | 保留在 Problem |
| `cache_top(...)` | 仍在 Problem 内，作为当前 Problem 的 cache 视图输出 | 保留在 Problem |
| `__init__(...)` | 不迁出，属于 Problem 身份与契约定义 | 应保留 |
| `_decode(...)` | 不迁出，属于 Problem 决策编码本体 | 应保留 |

### 9.2 如何理解这些状态

#### `已迁出`

表示：

- 这部分职责已经正式搬到公共组件中
- 后续不应再把实现写回 `problem_model.py`
- 如需增强，应优先修改公共组件

当前典型代表有：

- `branch_evaluator`
- `fold_report`
- `subset_descriptor`
- `batch_interval_utils`

#### `部分保留`

表示：

- 这部分不再是重实现逻辑
- 它现在更多是桥接层、适配层、装配层
- 仍然留在 Problem 内，是因为它需要把当前 Problem 的配置、回调、协议接给公共组件

当前典型代表有：

- `_build_interval_bounds(...)`
- `_summarize_fold(...)`
- `_branch_eval_config(...)`
- `_fit_predict(...)`

这类函数一般不是本轮最优先继续拆的对象，但将来仍可进一步接口化。

#### `保留在 Problem`

表示：

- 这部分职责当前就应由 Problem 壳承担
- 不是“放错位置”，而是应继续保持薄实现

当前典型代表有：

- `_evaluate_subset(...)`
- `_evaluate_decoded(...)`
- `_evaluate_decoded_batch(...)`
- `evaluate(...)`
- `evaluate_population_batch(...)`

#### `应保留`

表示：

- 这些函数原则上就不应该迁出
- 它们定义的是具体 Problem 的身份、边界与决策编码本体

当前典型代表有：

- `__init__(...)`
- `_decode(...)`

### 9.3 这张表给出的直接启发

从迁移台账可以看出，本轮最关键的三项“尚未迁出”职责已经完成公共化：

1. split 协议
   - `_rolling_splits(...) -> cv_splitter.py`

2. cache key 协议
   - `_cache_sig(...) -> evaluation_cache_key.py`

3. objective preference policy
   - `_coverage_error(...) / _objective_sort_key(...) -> objective_policy.py`

这说明文档里此前标记的三项“尚未形成”已经和代码完成对齐。

也就是说，接下来的拆分重点不再是这三把刀，而会进一步转向：

- 更彻底地减少 Problem 内部桥接函数
- 继续把 runtime / config / evaluation provider 标准化
- 推进 `mlblack` 自己的标准脚手架层次

这也意味着 `problem_model.py` 已进一步接近真正的薄壳形态。

---

## 10. 为什么说当前方向是正确的

这轮拆分的正确性，主要体现在下面几个方面。

### 10.1 公共层与场景层边界正在清晰化

现在已经能明确区分：

哪些属于场景：

- 数据列
- strict4 是否启用
- interval method 选择
- 当前实验配置

哪些属于公共能力：

- branch 评估编排
- fold 汇总
- subset 描述
- batch 数值工具

这就是健康架构最重要的一步。

### 10.2 Problem 文件已经不再是“功能黑洞”

之前任何功能都往 `problem_model.py` 塞，现在已经不再这样。

新功能如果落在：

- branch evaluator
- fold report
- subset descriptor
- batch interval utils

说明我们已经开始具备“先判断职责归属，再放文件”的能力。

### 10.3 为 mlblack 脚手架化提供了积累

这轮拆分不是局部清理，而是在为 `mlblack` 的标准公共层铺路。

也就是说，现在上提的组件不是一次性拆分产物，而是未来可沉淀进：

- `features`
- `evaluation`
- `reporting`
- `descriptor`
- `runtime`
- `problem`

这些标准层次里的正式成员。

---

## 11. 下一步标准化路线图

如果继续沿着当前方向推进，最自然、最值得继续上提的有以下三类能力。

### 11.1 CV Splitter

已完成来源：

- `_rolling_splits(...)`

当前公共模块：

- `cv_splitter.py`

当前职责：

- 统一 rolling split 生成策略
- 接受 folds / val_ratio / min_train 等参数
- 对极端样本量自动 fallback
- 为时序 symbolic 评估提供统一 split 协议

当前状态：

- 已抽到公共层
- `problem_model.py` 已改为调用 `build_rolling_splits(...)`
- rolling split 已不再是 Problem 内部私有实现

### 11.2 Evaluation Cache Key

已完成来源：

- `_cache_sig(...)`
- subset 与 meta 组合成 cache key 的逻辑

当前公共模块：

- `evaluation_cache_key.py`

当前职责：

- subset key 规范化
- meta signature 规范化
- cache key 稳定序列化
- 不同 evaluator 共享统一缓存签名协议

当前状态：

- 已抽到公共层
- 已形成：
  - `build_meta_signature(...)`
  - `build_subset_meta_cache_key(...)`
- `problem_model.py` 的 cache key 组装已经切到公共协议

### 11.3 Objective Policy

已完成来源：

- `_coverage_error(...)`
- `_objective_sort_key(...)`

当前公共模块：

- `objective_policy.py`

当前职责：

- 定义 coverage-threshold-first 策略
- 定义 coverage error 的标准口径
- 支持 future policy 切换，例如：
  - 先满足 coverage，再压 PINAW
  - 先满足 coverage，再压 interval score
  - coverage 与 PINAW 联合偏好

当前状态：

- 已抽到公共层
- 已形成：
  - `coverage_error(...)`
  - `interval_objective_sort_key(...)`
- `cache_top(...)` 等排序逻辑已切到公共 objective policy

---

## 12. 中期目标：把 problem_model 收缩成真正薄壳

如果继续完成：

- `cv_splitter`
- `evaluation_cache_key`
- `objective_policy`

那么 `problem_model.py` 将非常接近理想状态。届时它主要只保留：

1. Problem 初始化
2. 决策解码
3. 调用公共 evaluator
4. 调用公共 aggregator
5. 调用公共 descriptor
6. 管理 Problem 级本地缓存
7. 满足 `BlackBoxProblem` 契约

那时它就不再是“大量逻辑堆叠的实现文件”，而是一个真正的 Problem 壳。

---

## 13. 对 mlblack 标准脚手架的启示

这份文档不仅是在描述 `problem_model.py` 的拆分，也是在给 `mlblack` 未来的标准脚手架摸路径。

从当前趋势看，未来更合理的 `mlblack` 公共结构很可能是：

### 13.1 features

负责：

- primitive registry
- grammar
- candidate pool
- feature bundle
- temporal/regime feature pack

### 13.2 evaluation

负责：

- branch evaluator
- cv splitter
- cache key
- objective policy
- batch interval utils

### 13.3 reporting

负责：

- fold report
- summary writer
- table / plot aggregation

### 13.4 descriptor 或 problem_support

负责：

- subset descriptor
- future decision descriptor
- candidate metadata descriptor

### 13.5 runtime

负责：

- workflow
- orchestration
- experiment wiring

### 13.6 problem

负责：

- 薄壳 Problem
- encode / decode
- delegate / cache / contract

### 13.7 config

负责：

- 每个模块自己的 config.py
- 避免 workflow 持有大量业务细节

这也正是你前面一直强调的方向：

- 每层有自己的 config
- workflow 只编排
- problem 只做壳
- 公共逻辑沉入公共层

---

## 14. 与 nsgablack 架构对照

这一节不是说 `mlblack` 要机械复制 `nsgablack`，而是要说明：

- 哪些结构思想是可以对齐的
- 哪些职责边界在两个系统里是同构的
- 哪些地方目前还没有达到 `nsgablack` 那种清晰度

### 14.1 nsgablack 的核心结构

`nsgablack` 的标准架构，本质上是一个很清晰的多层正交系统：

- `Solver`
  - 控制平面
  - 负责生命周期、状态、评估入口、插件调度
- `Adapter`
  - 算法策略平面
  - 负责 propose / update
- `Representation`
  - 表示平面
  - 负责 init / mutate / repair / encode / decode
- `Plugin`
  - 能力平面
  - 负责 checkpoint、trace、evaluation shortcut、backend 等增强能力

也就是说，`nsgablack` 的一个核心优点是：

- 控制、策略、表示、能力增强这四类职责边界非常明确

### 14.2 当前 mlblack 还不是同一种分层

当前 `mlblack` 这边的拆分，和 `nsgablack` 还不是同一维度的架构拆分。

原因是：

- `nsgablack` 是优化框架
- `mlblack` 这里正在形成的是“符号学习评估与特征空间公共层”

所以它们不是一一同名对应的关系，而是“职责对照”的关系。

更准确地说：

- `nsgablack` 管的是“怎么搜”
- `mlblack` 这边管的是“搜到一个候选结构之后，怎么解释、怎么评估、怎么汇总、怎么构造特征空间”

所以我们不应该硬说：

- `branch_evaluator = adapter`
- `fold_report = plugin`

这会误导。更合理的说法是：

- `mlblack` 正在补的是“Problem 背后的评估基础设施层”

### 14.3 当前已经对齐 nsgablack 的地方

虽然不是同一维度，但有几个结构思想已经明显在向 `nsgablack` 靠拢。

#### 1. Problem 正在变薄

这点最接近 `nsgablack` 的标准思想。

在 `nsgablack` 里，理想的 Problem 不应自己背大量运行时能力或算法策略；它更像一个契约壳。

现在 `mlblack_side/problem/problem_model.py` 也在朝这个方向变：

- 不再自己展开 branch 评估
- 不再自己展开 batch 数值工具
- 不再自己展开 fold 汇总
- 不再自己展开 subset 元数据装配

这和 `nsgablack` 的“Problem/壳层不应吞掉整条评估链”这一思想是对齐的。

#### 2. 编排与实现开始分离

`nsgablack` 一个很强的点是：

- `Solver` 负责编排
- `Adapter` 负责算法实现
- `Plugin` 负责增强能力

现在 `mlblack` 这里也开始出现类似的趋势：

- `problem_model`
  - 更像委托与装配层
- `branch_evaluator`
  - 更像评估编排层
- `batch_interval_utils`
  - 更像底层实现层
- `fold_report`
  - 更像输出聚合层

虽然名字不同，但“编排不要和具体实现混在一起”的思想已经对齐。

#### 3. 公共协议开始出现

`nsgablack` 的强处之一是它有明确的 API 契约：

- adapter 的 `propose/update`
- plugin 的生命周期钩子
- representation 的 encode/decode/repair

`mlblack` 这边虽然还远没有那么完整，但已经在出现一些雏形：

- `branch_evaluator` 接收统一回调：
  - `fit_predict_fn`
  - `build_interval_bounds_fn`
  - `summarize_fold_fn`
- `fold_report` 有固定输入输出协议：
  - `fold_results -> objective/detail`
- `subset_descriptor` 有固定输入输出协议：
  - `candidates + subset_idx -> genome/subset_candidates`

这说明我们已经不只是“拆文件”，而是在逐步形成组件协议。

### 14.4 当前还没有对齐 nsgablack 的地方

这部分最重要，因为它决定了我们接下来还差什么。

#### 1. mlblack 还没有完整的标准层级

`nsgablack` 现在是很明确的：

- solver
- adapter
- representation
- plugin

而 `mlblack` 当前还没有完全稳定下来的标准层次命名。

虽然我们已经在提：

- `features`
- `evaluation`
- `reporting`
- `runtime`
- `problem`
- `config`

但这些层还没有像 `nsgablack` 那样形成统一的“官方脚手架结构 + 标准 API”。

这意味着：

- 现在更像是“正在长出标准层次”
- 还不是“已经完全标准化”

#### 2. config 还没有彻底模块化

在 `nsgablack` 的理想脚手架里，一个非常重要的点是：

- 每层通常有自己的配置入口
- build/assembly 阶段只负责装配，不在 workflow 里硬塞大量业务细节

`mlblack` 现在虽然已经有一些 config 拆分，但还没有做到：

- `features` 一整层一套明确 config
- `evaluation` 一整层一套明确 config
- `reporting` 一整层一套明确 config
- `runtime` 只负责编排，不再吞大量参数解释

这点还需要继续向 `nsgablack` 学。

#### 3. runtime / workflow 还不够像标准 orchestrator

`nsgablack` 的 `Solver` 很清楚是控制平面。

现在 `mlblack` 里的 `workflow` / `runtime` 虽然在往“只编排”靠，但还没有完全做到类似 `Solver` 的那种纯控制平面角色。

目前 `mlblack` 的 runtime 仍可能夹杂：

- 数据预处理决策
- 实验业务参数解释
- 某些场景特有逻辑

这说明 runtime 层仍需继续清理。

#### 4. evaluation 层还没有形成像 plugin/provider 那样的可插拔入口

`nsgablack` 的一个强点是：

- 很多增强能力通过 plugin/provider 接入
- 不是把所有逻辑直接写回主流程

`mlblack` 当前虽然已经把 evaluator 做成公共组件，但还没有形成像 `nsgablack` 那样明确的：

- evaluation provider 接口
- report provider 接口
- feature-space builder 接口

也就是说，组件已经有了，但“标准接入点”还不够强。

### 14.5 可以怎样理解两者关系

最合理的理解不是“mlblack 模仿 nsgablack 的目录名”，而是：

- `nsgablack` 提供了一个非常成熟的“职责分层方法论”
- `mlblack` 现在正在把这套方法论翻译到自己的问题域里

具体来说：

#### nsgablack 的关键方法论

1. 壳层要薄
2. 编排不要吞实现
3. 公共能力不要写死在单场景中
4. 配置与实现分层
5. 接口先于散乱函数

#### mlblack 当前的落实方式

1. `problem_model` 变薄
2. `branch_evaluator / fold_report / subset_descriptor / batch_interval_utils` 上提
3. `feature_space` 逐渐变成公共层
4. 下一步准备继续提 `cv_splitter / cache_key / objective_policy`
5. 后续再继续补标准 config 与 runtime 编排层

### 14.6 一个更准确的对应关系

如果只是讲“思想类比”，那么上一阶段那种说法是成立的；但如果是为了真正指导后续重构，口径还要再收紧一层：

- 不只是“职责相似”
- 而是要尽量做到“脚手架槽位一一对应”

也就是说，后续 `mlblack` 不应该继续沿着“场景里拆几个子目录”这条路慢慢长，而应该尽量对着 `nsgablack` 的标准骨架去补自己的根层。

更直接的对应关系应该写成下面这样：

```text
nsgablack
  problem/      -> mlblack problem/        （公共问题契约、bridge、proxy）
  pipeline/     -> mlblack pipeline/       （特征空间、编码/解码、candidate pool、feature bundle）
  bias/         -> mlblack bias/           （branch policy / objective policy / dynamic pool policy）
  solver/       -> mlblack workflow/       （控制平面；后续也可再收敛为 solver/ 命名）
  adapter/      -> mlblack model/          （训练器/预测器/区间策略）
  evaluation/   -> mlblack evaluation/     （评估运行时、provider、batch evaluator）
  plugins/      -> mlblack plugins/        （报告、日志、cache、snapshot、副作用 hook）
  config/       -> mlblack config/         （各层自己的 config.py 与统一根配置）
  build_solver  -> mlblack assembly/build  （统一装配）
  run_solver    -> mlblack run.py          （唯一入口）
```

这里最关键的变化是：

- `workflow` 不再只是“runtime 的另一个名字”，而是明确对应 `nsgablack` 的控制平面
- `model` 不是随便放几个拟合函数，而是明确对应“策略实现层”
- `pipeline` 不是零散特征工具，而是明确对应“表示与管线层”
- `plugins` 不是报表脚本堆放处，而是明确对应“副作用与观察能力层”

换句话说，真正要对齐的不是几个函数，而是整套骨架槽位。

### 14.7 直接按 nsgablack 骨架对齐时，mlblack 应有的标准层

下面这张表不是“现状描述”，而是后续重构应该遵循的目标骨架。

| nsgablack 标准层 | mlblack 应对应的标准层 | 这层只应该负责什么 | 当前落位情况 |
| --- | --- | --- | --- |
| `problem/` | 顶层 `problem/` | 问题契约、解码对象、评估桥接、代理协议 | 已有雏形，已开始上提 |
| `pipeline/` | 顶层 `pipeline/` | 特征空间、候选池、subset 描述、design matrix 输入标准化 | 只有一部分已上提，主体仍在场景侧 |
| `bias/` | 顶层 `bias/` | 目标偏好、branch policy、动态扩池偏好 | 已基本形成 |
| `solver/` | 顶层 `workflow/` | 控制平面、阶段编排、生命周期与 hook 总线 | 已形成最小骨架，但仍需继续纯化 |
| `adapter/` | 顶层 `model/` | 训练器、预测器、区间策略对象与统一接口 | 还没有根层标准层，主要仍在场景侧 |
| `evaluation/` | 顶层 `evaluation/` | batch-first evaluator、provider、评估 runtime | 还没有根层标准层，主要仍在场景侧 |
| `plugins/` | 顶层 `plugins/` | 报告、日志、缓存、快照、图表、DOCX 等副作用 hook | 还没有根层标准层，主要仍在场景侧 |
| `config/` | 顶层 `config/` + 各模块 `config.py` | 模块参数注册、装配参数、统一 schema | 部分存在，但还未全域统一 |
| `build_solver.py` | 顶层装配模块 | 按骨架顺序统一装配问题、管线、策略、评估、插件 | 已有 `run.py / build_solver` 雏形 |
| `run_solver.py` | 顶层唯一入口 | 只负责解析入口、调用装配、启动运行 | 已有统一入口，但旧兼容入口仍较多 |

### 14.8 当前阶段的结论

所以如果把当前 `mlblack` 和 `nsgablack` 放在一起看，可以得出一个比较稳的判断：

- 现在的 `mlblack` 还没有达到 `nsgablack` 那种“根层槽位完整、每层都有稳定接口”的标准脚手架程度
- 但方向已经不该再停留在“拆函数”，而应该明确转成“补齐根层骨架”
- 这轮改造真正有价值的地方，是已经开始把几层关键槽位长出来：
  - `bias/`
  - `problem/`
  - `pipeline/`
  - `workflow/`

换句话说：

- 现在不是“已经完全对齐”
- 而是“已经确定要按 nsgablack 这套骨架继续补齐”

### 14.9 对照表：nsgablack vs mlblack 当前状态

下面这张表用于做架构讨论时快速定位：

- `nsgablack` 的标准能力项是什么
- `mlblack` 当前有没有对应物
- 对齐程度处于什么状态

| nsgablack | mlblack 当前对应物 | 状态 |
| --- | --- | --- |
| `Solver` 作为纯控制平面 | 顶层 `workflow/` + 场景侧 `runtime/stages.py` | 部分对齐 |
| `Problem` 作为薄壳契约层 | 顶层 `problem/` + 场景侧 `problem/problem_model.py` | 部分对齐 |
| `Representation` ?? encode/decode/init/mutate/repair | ?? `pipeline/feature_space.py` + `pipeline/feature_space_builder.py` | ???? |
| `Adapter` 负责算法策略 propose/update | 顶层 `model/` 已形成，当前以 `interval_fit.py` 为主 | 部分对齐 |
| `Evaluation` 负责评估通路/provider | 顶层 `evaluation/` 已形成，当前以 `problem_callbacks.py` 为主 | 部分对齐 |
| `Plugin/Provider` 负责能力增强与外接 | 顶层 `plugins/` 已形成，当前以 `report_writer.py / report_writer_plugin.py` 为主 | 部分对齐 |
| 模块级清晰边界 | `bias / model / evaluation / plugins / problem / pipeline / workflow` 已成雏形 | 部分对齐 |
| 模块各自 `config.py` | 已有局部 config，但未完全模块化 | 部分对齐 |
| `workflow` 只负责编排 | 当前 runtime 仍夹杂部分业务解释与装配细节 | 部分对齐 |
| 统一公共 API 契约 | bridge / proxy / evaluator / descriptor / report 已开始形成协议 | 部分对齐 |
| 公共能力不写死在单场景 | `feature_space` 已承接大量公共能力 | 已对齐 |
| 评估编排与 Problem 分离 | `branch_evaluator` 已从 `problem_model.py` 上提 | 已对齐 |
| 报告聚合与 Problem 分离 | `fold_report` 已从 `problem_model.py` 上提 | 已对齐 |
| subset 元数据装配公共化 | `subset_descriptor` 已上提 | 已对齐 |
| batch 数值工具公共化 | `batch_interval_utils` 已上提 | 已对齐 |
| 统一 split 协议层 | `cv_splitter.py::build_rolling_splits(...)` | 已对齐 |
| 统一 cache key 协议层 | `evaluation_cache_key.py::build_meta_signature(...) / build_subset_meta_cache_key(...)` | 已对齐 |
| 统一 objective policy 层 | `objective_policy.py::coverage_error(...) / interval_objective_sort_key(...)` | 已对齐 |
| 稳定的 evaluation provider 接口 | 当前已有 problem bridge/proxy，但顶层 `evaluation/` 仍未独立成层 | 部分对齐 |
| 稳定的 reporting provider 接口 | 当前已有报表写入器与 hook，但顶层 `plugins/` 仍未独立成层 | 部分对齐 |
| 标准脚手架目录与装配模板 | 还没有正式沉淀成 mlblack 标准模板 | 尚未形成 |

### 14.10 如何解读这张对照表

这张表最重要的不是“有多少项打勾”，而是帮助判断下一步工作应该落在哪一层。

#### `已对齐`

表示：

- 这部分已经完成了从场景层到公共层的职责搬迁
- 不再建议把相关实现写回 `problem_model.py`
- 后续重点应放在稳定接口，而不是重新回退到场景文件里堆逻辑

当前属于这一类的典型项有：

- 评估编排与 Problem 分离
- 报告聚合与 Problem 分离
- subset 元数据装配公共化
- batch 数值工具公共化
- 公共能力不再写死在单场景

#### `部分对齐`

表示：

- 方向已经正确
- 已经出现明确对应物
- 但还没有达到 `nsgablack` 那种“标准层次 + 标准接口 + 标准装配”都稳定的程度

当前属于这一类的典型项有：

- runtime / workflow 作为控制平面
- Problem 作为薄壳
- evaluator / report 的协议化
- config 模块化

这类项通常是下一阶段最值得继续收口的对象。

#### `尚未形成`

表示：

- 这不是“做得不好”，而是“这层还没有被正式抽象出来”
- 当前通常还留在 `problem_model.py` 或 runtime 里
- 应作为后续明确的公共组件候选目标

当前最典型的三项就是：

- `cv_splitter`
- `evaluation_cache_key`
- `objective_policy`

这三项一旦补齐，`problem_model.py` 会进一步明显变薄。

---

## 15. 当前阶段结论

到目前为止，`problem_model.py` 已经完成了一轮非常关键的瘦身。已经成功上提的包括：

- strict4/global 分支评估编排
- fold 汇总与 detail 报告聚合
- subset 元数据与 genome 组装
- batch interval 数值工具

这意味着现在的 `problem_model.py` 已经不再是：

- 训练逻辑中心
- 数值工具中心
- 报告生成中心
- 子集元数据装配中心

而是逐步转变为：

- Problem 契约层
- 决策解码层
- 委托编排层
- 缓存层

这条路线是正确的，而且已经具有清晰的延续性。下一阶段最值得继续做的是：

1. 把 split 抽成 `cv_splitter`
2. 把 cache key 抽成 `evaluation_cache_key`
3. 把目标偏好抽成 `objective_policy`
4. 再基于这些沉淀，写出 `mlblack` 自己的标准脚手架草案

到那时，`nowcasting_work_ci` 就会更加接近你要的状态：

- 场景层只剩数据与实验装配
- mlblack 公共层负责承载通用 symbolic learning 逻辑

---

## 16. 防碎片化约束（脚手架契约）

拆分的目标是解耦与复用，而不是把逻辑打散。为避免“文件变多但可用性变差”，这里定义一套对齐 `nsgablack` 思路的硬约束：按平面分责、按契约接入、按禁区防回流。

### 16.1 控制平面（Control Plane）

对标：

- `nsgablack` 的 `SolverBase / ComposableSolver`

在 `mlblack` 中对应：

- `runtime/workflow` 层
- 未来可收敛为 `BaseEvaluator` 或 `ExperimentOrchestrator`

只能做什么（Allowed）：

- 驱动生命周期：`prepare -> split -> evaluate -> aggregate -> finalize`
- 编排阶段顺序与预算控制
- 组织依赖注入（pipeline、strategy、plugins）
- 统一入口（等价 `solve()`）

接口清单（Contract）：

- `run(context) -> ExperimentResult`
- `run_fold(fold_ctx) -> FoldResult`
- `on_stage_start/on_stage_end`（可选）

禁区清单（Forbidden）：

- 不允许内联具体模型训练（例如直接写 Ridge/CQR 细节）
- 不允许内联特征工程细节（例如手工拼接 lag/strict4 分支）
- 不允许承担日志、落盘、报告生成等副作用实现

### 16.2 策略平面（Strategy Plane）

对标：

- `nsgablack` 的 `Adapter`（`propose/update` 契约）

在 `mlblack` 中对应：

- 预测策略与区间策略（point model + uncertainty model）
- 例如 Ridge、Tree、CQR、symmetric residual 等策略对象

只能做什么（Allowed）：

- 给定标准输入矩阵，完成训练与预测
- 输出标准化预测结果（点预测/区间或分位数）
- 暴露可替换策略参数，不影响控制流

接口清单（Contract）：

- `fit(X, y) -> self`
- `predict(X) -> y_hat`
- `predict_interval(X, alpha) -> (lower, upper)` 或
- `predict_quantiles(X, quantiles) -> q_preds`

禁区清单（Forbidden）：

- 不允许自己切分数据集（split 属于控制平面）
- 不允许自己读写缓存、文件、图表、报告
- 不允许依赖场景字段名或 hard-code 业务列

### 16.3 表示与管线平面（Representation / Pipeline）

对标：

- `nsgablack` 的 `RepresentationPipeline`（encode/decode/repair）

在 `mlblack` 中对应：

- ?? `pipeline/feature_space.py`?`pipeline/feature_space_builder.py`?`subset_descriptor`?`feature_bundle`?`candidate_pool`
- 未来统一成 `FeaturePipeline`

只能做什么（Allowed）：

- 把原始数据转换为标准 `Design Matrix`
- 完成 subset -> genome -> matrix 的确定性流转
- 处理分支特征装配（strict4/holiday/lag/cross）但只输出标准化结果

接口清单（Contract）：

- `transform(raw_data, context) -> FeatureBundle`
- `build_design_matrix(bundle, subset) -> X`
- `describe_subset(candidates, subset_idx) -> descriptor`

禁区清单（Forbidden）：

- 不允许训练模型
- 不允许计算最终业务指标与排序策略
- 不允许写日志、写文件、产报告

### 16.4 插件平面（Plugin / Hooks）

对标：

- `nsgablack` 的 `PluginManager` 与生命周期钩子

在 `mlblack` 中对应：

- Observer / MetricLogger / SnapshotWriter / ReportWriter 等副作用能力

只能做什么（Allowed）：

- 监听生命周期事件并执行副作用
- 记录指标、持久化快照、输出图表和文档（含 DOCX）
- 在不破坏主流程契约下做可插拔增强

接口清单（Contract）：

- `on_experiment_start(ctx)`
- `on_fold_end(fold_result, ctx)`
- `on_experiment_finish(result, ctx)`

禁区清单（Forbidden）：

- 不允许修改核心算法语义
- 不允许替代主评估流程中的关键计算
- 不允许将业务逻辑反向塞回控制平面或策略平面

### 16.5 跨层约束（最关键）

这几条是防碎片化的“红线”，任何重构都应遵守：

1. 控制平面只编排，不实现算法。
2. 策略平面只实现算法，不管理生命周期。
3. Pipeline 只做表示转换，不做训练与排序。
4. 插件只做副作用，不改核心语义。
5. 任何新能力先判断层归属，再决定放置位置。
6. 若一个改动同时触发多层职责，优先补接口，不直接跨层调用。

### 16.6 PR 检查清单（执行版）

每个相关 PR 至少回答以下问题：

1. 这个改动属于哪一层？
2. 是否新增了可替换接口，而不是把实现写死？
3. 是否把副作用留在插件层？
4. 是否让 `problem_model.py` 变薄，而不是再次变厚？
5. 是否出现跨层耦合（例如策略层直接写文件）？
6. 是否保持批量评估与缓存协议一致？

如果第 1 条答不清楚，说明改动很可能正在制造新的碎片化。

### 16.7 禁区映射（目录与文件）

这一节把“禁区”从原则落到具体落盘位置，确保评审时可以按文件路径直接判断。

| 平面 | 禁区规则 | 禁止出现的目录/文件 | 正确落位 |
| --- | --- | --- | --- |
| 控制平面 | 编排层写死模型训练细节（Ridge/CQR） | `nowcasting_work_ci/mlblack_side/runtime/workflow.py`、`nowcasting_work_ci/mlblack_side/runtime/runner.py` | 顶层 `model/`、顶层 `evaluation/` |
| ???? | ???????????????? | `nowcasting_work_ci/mlblack_side/runtime/*` | ?? `pipeline/feature_space_builder.py`?`core/symbolic/feature_space/` |
| 控制平面 | 编排层直接写报告/落盘副作用 | `nowcasting_work_ci/mlblack_side/runtime/*` | 顶层 `plugins/`、Hook 插件 |
| 策略平面 | 策略层自行 split 数据 | 顶层 `model/*`、顶层 `evaluation/problem_callbacks.py` | `core/symbolic/feature_space/cv_splitter.py` + 控制平面调用 |
| 策略平面 | 策略层写缓存协议/键生成 | 顶层 `model/*` | `core/symbolic/feature_space/evaluation_cache_key.py` |
| 策略平面 | 策略层写排序偏好策略 | 顶层 `model/*` | `core/symbolic/feature_space/objective_policy.py` |
| ??/???? | Feature/Pipeline ?????? | ?? `pipeline/*`?`core/symbolic/feature_space/feature_bundle.py` | ?? `model/` ? ?? `evaluation/` |
| ??/???? | Feature/Pipeline ?????????? | ?? `pipeline/*` | `core/symbolic/feature_space/fold_report.py`?`objective_policy.py` |
| ??/???? | Feature/Pipeline ??????? | ?? `pipeline/*` | ?? `plugins/` ? Hook ?? |
| 插件平面 | Hook 插件修改核心评估语义 | `nowcasting_work_ci/mlblack_side/runtime/hook_bus.py`、插件实现文件 | 控制平面/策略平面显式契约参数，不在 hook 中偷改 |
| 插件平面 | Hook 插件替代主评估计算 | 插件实现文件 | 仅监听 `on_stage_*` / `on_experiment_*`，不替代主流程 |
| Problem 壳层 | `problem_model.py` 回流重实现（大段 batch 工具/聚合逻辑） | `nowcasting_work_ci/mlblack_side/problem/problem_model.py` | `core/symbolic/feature_space/*`、顶层 `evaluation/problem_callbacks.py` |

#### 执行规则

1. 任何 PR 涉及上述“禁止出现目录/文件”时，必须在描述里说明为什么不是违规。
2. 若新增功能无法归入“正确落位”，先补契约再写实现，不允许临时塞进 `problem_model.py` 或 `runtime/*`。
3. 评审优先问“层归属”再问“代码风格”，层归属不清直接退回修改。

### 16.8 本轮代码化落地（Control + Plugin）

为避免契约只停留在文档，本轮已落地最小骨架：

- 控制平面入口：
  - `nowcasting_work_ci/mlblack_side/runtime/orchestrator.py`
  - `ExperimentOrchestrator.run(...)` 负责统一实验生命周期驱动
- 插件平面总线：
  - `nowcasting_work_ci/mlblack_side/runtime/hook_bus.py`
  - `HookBus` + `RuntimeHook` 协议负责生命周期 hook 派发
- 统一 runtime 入口壳：
  - `nowcasting_work_ci/mlblack_side/runtime/workflow.py`
  - `main(argv, hooks=...)` 通过 orchestrator + hook bus 驱动旧运行主流程

这意味着控制平面和插件平面已经从“建议”变成“可执行契约”。
