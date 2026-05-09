# 面向机制的 Basis 发现协议

把正交基生成从 trainer 内部逻辑提升为 `mlblack` symbolic family 正式协议的设计稿。

本文同时使用两个互相关联的名字：

- 正式机制对象名：`OrthogonalBasisGenerationMechanism`
- 系统级协议名：`Mechanism-Oriented Basis Discovery Protocol`

状态：

- 设计稿
- 还没有完全 materialize 成 catalog 中的正式框架对象
- 目的是指导后续 symbolic family 实现、runtime surface 和 experiment 评估面的统一

---

## 1. 为什么需要这份协议

symbolic 学习里的核心问题，很多时候不只是：

- 最终表达式怎么拟合

更前面的问题其实是：

- 正确的 basis 坐标系怎么生成

如果 basis 生成机制本身较弱，那么后面的 symbolic regression 即便拟合得不错，也很可能恢复出来的是：

- proxy 结构
- 冗余结构
- 不稳定结构
- 语义重复结构

所以这次要推进的升级是：

- 把正交 basis 生成当成一等机制对象
- 而不是当成 trainer 里的一个私有筛选 helper
- 也不是埋在某个 symbolic preset 里的实现细节

---

## 2. 问题定义

在 symbolic 学习里，“正交”不能只被理解成低 pairwise correlation。

在这份协议里，正交是相对的、多维度的。

一个好的 basis set 应该同时具备：

1. 统计互补性
2. 对当前残差有继续解释力
3. 语义上不冗余
4. 结构上可组合
5. 在 piecewise / gate 情况下具备 regime 感知能力
6. 稳定到足以通过 multi-run consensus 留下核心项

这份设计稿的核心研究判断是：

- 在搜索最终符号表达式之前，系统应该先搜索一套好的机制坐标系

---

## 3. 它在 `mlblack` 里的位置

这份协议应该归属 `mlblack`，而不是 `nsgablack`。

原因：

- `mlblack` 负责 symbolic family 的训练语义
- `mlblack` 负责 basis 生成语义
- `mlblack` 负责 basis 的含义、等价类和 artifact schema
- `nsgablack` 应该编排和观察这些 run，而不是定义 basis 本身的机制语义

在框架分类上：

- `family = symbolic`
- `component = OrthogonalBasisGenerationMechanism`
- `structure_head = basis_set | expression`
- `prediction_head = point | interval`
- 机制绑定级别更接近 `defining` 或 `bound`，而不是可有可无的 `optional`

这很重要，因为：

- 它不应该被误做成一个假的新 family
- 也不应该只活在某个 preset 的 trainer 分支里
- `basis_set` 应被理解为 symbolic 专属的一种 structure head，而不是新 family

---

## 4. 和现有 symbolic contracts 的关系

当前 symbolic 结构栈已经有：

- `SymbolicRegimeDiscoveryContract`
- `SymbolicBasisDiscoveryContract`
- `BudgetedSymbolicAssemblerContract`

这份协议的定位是：

- 当 symbolic family 采用 orthogonal-basis-first 路线时，它定义 `SymbolicBasisDiscoveryContract` 应该如何被系统级地理解和实现

也就是说，推荐映射是：

1. `SymbolicBasisDiscoveryContract`
   - 面向 family 的结构阶段合同
2. `OrthogonalBasisGenerationMechanism`
   - 实现该合同的具体机制层
3. `Mechanism-Oriented Basis Discovery Protocol`
   - 规定语义、评分、持久化和 runtime-surface 预期的上位系统协议

---

## 5. 核心论点

最准确的理解方式不是：

- 一个上游搜索器，再加一个下游训练器

而是：

- 同一个 symbolic family
- 共用一套核心训练骨架
- 在不同阶段切换不同的 structure head 和目标合同

### 5.1 同骨架，不同 Head

两个阶段本质上都还是 symbolic regression。

它们共享：

- 同一份数据语义
- 同一套 symbolic search backbone
- 同一种通用循环，例如 `propose -> evaluate -> update -> select`

真正变化的主要是：

- 搜索对象
- 评估合同
- 输出 head

第一阶段：

- `structure_head = basis_set`
- 搜索对象：基础特征空间上的 primitive 组合
- 输出对象：`selected_basis`
- 目标重点：orthogonality、residual complementarity、semantic novelty、consensus alignment

第二阶段：

- `structure_head = expression`
- 搜索对象：选中 basis 空间上的表达式
- 输出对象：最终 symbolic expression
- 目标重点：fit、simplicity、budget、truth recovery

当第二阶段输出的是预测型 artifact 时，还可以进一步叠加：

- `prediction_head = point | interval`

### 5.2 两种合法执行协议

据此，同一个 symbolic family 至少有两种合法执行模式。

`generic symbolic protocol`

- 没有上游 basis stage
- 直接用 `structure_head = expression`
- 在通用 symbolic 空间里搜索

`basis-orchestrated symbolic protocol`

1. 先运行一个 `structure_head = basis_set` 的 symbolic stage
2. 产出 `selected_basis`
3. 再运行一个 `structure_head = expression` 的 symbolic stage
4. 在 basis-conditioned 空间里组装最终表达式

### 5.3 为什么这点重要

在这个理解下，basis discovery 不是 preprocessing。

它本身就是一次正式的 symbolic training stage，只不过产物不是最终公式，而是 basis set。

所以系统真正的搜索对象升级成了：

- 第一步：找一套好的机制 basis 空间
- 第二步：在这套 basis 空间里找表达式

这比普通 symbolic regression 更严格，也更有原创性。

---

## 6. 正式对象定义

### 6.1 `OrthogonalBasisGenerationMechanism`

这是拟议中的正式机制对象。

它应该负责：

- basis candidate proposal
- basis equivalence normalization
- basis-set search
- basis scoring
- overlap / redundancy 报告
- regime-aware 的 piecewise / gate basis 处理

它不应该负责：

- 最终 head 语义
- report 写出等 side effect
- experiment 持久化 side effect
- 跨外部 outer solver 的编排

这里需要明确：

- 这个机制对象本身不是 `basis_set` head
- 它是可复用的 component，用来约束一个 `structure_head = basis_set` 的 symbolic stage

### 6.2 `Mechanism-Oriented Basis Discovery Protocol`

这份协议负责规定系统级规则：

- basis 生成消耗什么
- basis 生成输出什么
- 它怎么被评分
- 它怎么被持久化
- 它如何影响 family identity
- 它如何进入 runtime DB 和 experiment UI
- 一个 symbolic stage 何时应被解释成 `basis_set` head，何时应被解释成 `expression` head

---

## 7. I/O 合同

### 7.1 Consume

必需输入：

- feature matrix 或 feature provider
- target values
- symbolic primitive library
- basis search configuration

可选输入：

- residual baseline state
- fold / resample partition
- regime partition hint
- consensus prior rows
- locked core seed rows
- semantic equivalence rules
- known-relation truth contract

### 7.2 Produce

必需输出：

- `structure_head = basis_set`
- `basis_candidates`
- `basis_scores`
- `selected_basis`
- `basis_overlap_report`
- `basis_semantics`
- `basis_generation_protocol`

可选输出：

- `prediction_head = none`
- `basis_context`
- `basis_family_manifest`
- `piecewise_gate_basis`
- `residual_complementarity_report`
- `semantic_dedup_report`
- `consensus_alignment_report`
- `candidate_screen_report`
- `basis_search_trace`

---

## 8. 组合合同

推荐使用稳定声明：

- `requires`
- `provides`
- `mutates`
- `cache`

建议形态：

- `requires`
  - symbolic primitives
  - feature context
  - basis search config
- `provides`
  - basis candidates
  - selected basis set
  - basis diagnostics
- `mutates`
  - symbolic structure state
  - basis search trace state
- `cache`
  - candidate evaluation cache
  - equivalence normalization cache
  - residual reuse cache

这份协议不应该主要依赖散落的 `hasattr(...)` 判断或 trainer 私有上下文猜测。

---

## 9. 持久化合同

下面这些输出必须区分清楚：

- 面向 artifact 的 basis structure
- trainer_state / resume state
- search trace / report payload
- experiment/runtime surface projection

建议 artifact 段：

- `basis_structure`
- `piecewise_gate_basis`
- `basis_overlap_report`
- `basis_semantics`

建议 trainer-state 段：

- search frontier
- candidate score cache
- equivalence cache
- locked seed state

建议 report 段：

- basis ranking table
- candidate screen diagnostics
- semantic dedup diagnostics
- residual complementarity diagnostics

---

## 10. 兼容性合同

这套机制应该影响 symbolic family identity。

它应该进入：

- `family_signature_payload()`
- warm-start compatibility check
- resume drift check
- experiment surface 的合同层

至少下面这些字段应该被视为 compatibility-relevant：

- `basis_generation_protocol`
- `screening_protocol`
- `outer_search_protocol`
- primitive library signature
- equivalence mode
- regime handling mode
- locked-core seeding mode

如果这些发生漂移，系统应该能解释：

- 变了什么
- 这个变化是否安全
- 是否应该拒绝 warm-start 或 resume

---

## 11. 正交性的定义

这份协议采用更丰富的正交定义。

### 11.1 统计正交

- 低有害相关
- 已解释方差的重叠尽量小

### 11.2 残差正交

- 每个 basis 项都应该继续解释当前已选 basis 尚未覆盖的残差

### 11.3 语义正交

- 不能把几个本质上表达同一个 proxy 的不同写法当成“新 basis”

### 11.4 结构正交

- basis 项之间应当能被组合成紧凑的第二阶段 symbolic expression

### 11.5 Regime 正交

- 在 piecewise / gate 场景下，不同 regime 的 basis 不应被压扁成一个误导性的全局 proxy

### 11.6 共识正交

- 一个 basis 项如果能在多次 run 之后反复存活，更有资格被视为 core mechanism coordinate

---

## 12. 评分栈

screening 和 search 应该组合多种信号。

### 12.1 Candidate Screen

当前目标设计：

- `target corr`
- `residual gain`
- `semantic novelty`
- `consensus prior`

### 12.2 Basis-Set Search Objective

建议的 basis-set objective 应包括：

- mean candidate screen score
- orthogonality score
- residual complementarity
- semantic uniqueness
- pairwise correlation penalty
- feature overlap penalty
- family diversity bonus
- regime coverage bonus
- piecewise / gate bonus

### 12.3 Inner Validation

每提出一组 basis set，都应触发一次更小预算的 inner symbolic assembler。

外层分数不应只依赖静态 basis 诊断项。

还应看：

- `inner_fit_score`
- `assembly_budget_usage`
- `assembly_trace_quality`
- 当存在 truth contract 时的 mechanism-faithfulness 指标

### 12.4 `search_input_space`

这份协议应该把当前 symbolic stage 实际工作的搜索输入空间显式化。

推荐取值：

- `raw_feature_space`
- `basis_object_space`

推荐语义：

- `structure_head = basis_set` 通常应使用 `raw_feature_space`
- 通用 `structure_head = expression` 也可以使用 `raw_feature_space`
- basis-conditioned 的 `structure_head = expression` 应使用 `basis_object_space`

这个字段很重要，因为它明确了当前 stage 到底是在：

- 直接对原始特征空间搜索
- 还是对上一阶段产出的 basis object 空间搜索

### 12.5 `pool_expansion_unit`

动态拓池也应该显式声明它的原子扩展单位。

推荐取值：

- `raw_feature`
- `basis_object`

推荐语义：

- 在原始特征阶段，扩展单位是原始特征及其直接 symbolic seed
- 在 basis-conditioned 阶段，扩展单位是 basis object

如果一个 basis-conditioned stage 收到的 selected basis 是：

- `x1*x2`
- `sin(x3)`
- `x4`

那么拓池逻辑应把这些对象当成原子单位。

除非显式开启 escape / decomposition policy，否则它不应自由拆回：

- `x1`
- `x2`
- `x3`
- `x4`

### 12.6 `gradient_guidance_mode`

梯度引导也应该是显式的、分 stage 的。

推荐取值：

- `off`
- `raw_feature_gradient`
- `basis_object_gradient`
- `hybrid`

推荐语义：

- `raw_feature_gradient`：梯度在原始特征空间指导候选生成
- `basis_object_gradient`：梯度在 selected basis object 空间指导候选生成
- `hybrid`：同时允许两种视角，但必须显式开启

关键规则是：

- 当 `search_input_space = basis_object_space` 时，默认梯度视图也应当是对象级的

这意味着梯度信号帮助排序和扩展的对象应该是：

- basis object
- basis-object pair
- basis-object transform

而不是隐式退回到原始特征级别扩展。

---

## 13. 分层搜索视角

这份协议建议把整个过程理解成嵌套搜索。

### 13.1 外层

搜索 basis-generation program 或 basis-set hypothesis。

### 13.2 中层

用 orthogonality、novelty、residual complementarity、consensus alignment 去评分 basis-set hypothesis。

### 13.3 内层

在选中的 basis set 上运行 budgeted symbolic assembler。

这意味着 “basis mechanism” 不是一个 preprocessing 步骤。

它是最终表达式搜索之上的一层结构搜索。

---

## 14. Piecewise / Gate Basis

piecewise 或 gate-conditioned basis 必须是一等公民。

这份协议应直接支持：

- gate-conditioned basis row
- regime-local basis row
- breakpoint-sensitive basis candidate
- failed regime diagnostics
- local-to-global aggregation summary

如果数据明显存在 regime change，系统就不该强迫所有结构都塞进同一个全局 basis 词汇表。

---

## 15. 等价类语义

basis 等价类必须显式化。

至少应支持：

- `exact`
- `phase-equivalent`
- `family-level`

这很重要，因为很多 symbolic 结构：

- 语法不同
- 功能相近
- 机制相关

如果没有正式等价类处理，consensus 和 truth recovery 都会不稳定。

---

## 16. Runtime Surface 与 Experiment Surface 投影

这套机制应该进入 runtime / experiment surface。

建议投影字段：

- `structure_head`
- `prediction_head`
- `orchestration_mode`
- `basis_binding_mode`
- `basis_source`
- `basis_generation_protocol`
- `screening_protocol`
- `outer_search_protocol`
- `basis_equivalence_mode`
- `consensus_prior_row_count`
- `selected_core_row_count`
- `joint_core_score`
- `basis_overlap_report`
- `basis_semantics`
- `piecewise_gate_basis`

这些字段应进入：

- run metadata
- artifact metadata
- experiment tracker materialization
- dashboard filter 和 detail panel

---

## 17. Benchmark 合同

这份协议在 known-relation suite 上尤其有意义。

推荐 benchmark family：

- `ohm_like`
- `ideal_gas_like`
- `arrhenius_gate_like`
- `periodic_gate_like`
- `redundant_proxy_control`

推荐评价层次：

- fit metrics
- `exact_basis_hit_score`
- `exact_term_recovery_score`
- `phase_equivalent_term_recovery_score`
- `family_level_term_recovery_score`

对这套机制来说，benchmark 问题不只是：

- 拟合得好不好

更是：

- basis mechanism 是否恢复了对的 mechanism coordinates

---

## 18. 当前研究诊断

目前已经能看到一个典型失败模式：

- 系统可以变稳定
- consensus 可以变强
- locked-core 可以变稳
- 但稳定下来的仍然可能是错误 proxy family

所以当前下一步前沿不只是继续加预算。

真正重要的是：

- 更强的 mechanism-aware basis generation
- 更强的 anti-proxy semantic dedup
- 更强的 regime-aware basis candidate
- 更强的 outer preference，让 truth-like mechanism family 更容易胜出

---

## 19. 正式化推进计划

要把这份设计稿推进成正式框架面，比较合理的步骤是：

1. 定义稳定的 mechanism metadata key
2. 把机制挂进 symbolic family structure contract
3. 把机制字段 materialize 到 artifact schema
4. 把机制字段投影到 experiment runtime surface
5. 在 experiment dashboard 和 catalog-like surface 里开放筛选
6. 当表面稳定后，再把它作为 `component` 加进 catalog

这里要明确：

- 这份文档是在提出正式方向
- 并不宣称完整框架物化已经全部完成

---

## 20. 非目标

这份协议并不意味着：

- 每个 symbolic preset 都要变成一个新 family
- 每个 basis heuristic 都要过早冻结
- orchestration 应该搬进 `mlblack` 而不是留在 `nsgablack`
- report / persistence side effect 应该挪进 trainer 内部

真正要做的是：

- 把 basis generation 升级成正式 symbolic-family mechanism
- 同时让 orchestration、persistence 和产品面围绕它保持对齐

---

## 21. 两个正式机制护栏

当前协议应该显式区分两个机制级的护栏对象。

### 21.1 `EquivalenceExpressionHandlingMechanism`

目标：

- 把语法不同但机制相近的项收进等价类
- 避免把 phase-equivalent 或 proxy-equivalent 项误判为“新颖 basis”
- 减少在同一局部等价类内部无效的组合扩张

建议协议字段：

- `equivalence_expression_protocol`
- `equivalence_expression_mode`
- `equivalence_class_scope`

当前合同理解：

- 它位于候选生成和 consensus 解释之间
- 负责经验等价、残差等价、语义等价的统一处理
- 这并不等于等价类内部的代表替换已经完全硬化

### 21.2 `InterferenceFeatureHandlingMechanism`

目标：

- 识别或抑制 proxy-like 与源头重叠的 basis 项
- 惩罚在高重叠特征上做浅层非线性伪装
- 迫使系统去寻找真正新的机制坐标

建议协议字段：

- `interference_feature_protocol`
- `interference_feature_mode`
- `cross_explanatory_rejection_mode`
- `trivial_nonlinearity_penalty_mode`
- `environment_invariance_audit_mode`

当前合同理解：

- 这个机制需要先被正式化，再逐步硬化
- 当前实现仍是 heuristic-first
- 但协议面必须先稳定下来，这样后续 anti-proxy 升级不会把 schema 和 runtime 命名打散
