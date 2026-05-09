## PeriodicEquivalenceDisambiguationMechanism

### 目标
避免在观测区间较窄、局部拟合等价性较强时，`tanh`、浅层非线性、局部 surrogate 一类表达错误替代真实周期项。

### 协议位置
- Family: `symbolic`
- Parent protocol: `EquivalenceExpressionHandlingProtocol`
- Canonical child mode: `periodic_mode`
- Structure head: `basis_set`
- Primary stage: outer orthogonal structure search
- Secondary projection: artifact / report / runtime metadata

### Rehomed Position
这份文档现在描述的是叶子机制，而不是整个 equivalence-handling 层。

它是 `EquivalenceExpressionHandlingProtocol` 之下当前已经落地的 periodic specialization。

### 核心输入
- 原始特征矩阵
- 候选 symbolic basis 池
- `search_hints.periodic_feature_names`
- 候选项所属的 semantic family / expression family

### 正式字段
- `periodic_equivalence_protocol`
- `periodic_equivalence_disambiguation_mode`
- `phase_spectrum_audit_mode`
- `periodic_family_prior_mode`
- `periodic_family_prior_weight`
- `periodic_candidate_screen_reserve`
- `periodic_feature_names`

### 当前已实现模式
- `periodic_equivalence_disambiguation_mode=center_edge_holdout_penalty`
- `phase_spectrum_audit_mode=center_edge_holdout_report`
- `periodic_family_prior_mode=semantic_family_boost`

### 选择逻辑
1. 标记哪些特征属于周期特征。
2. 判断候选项是否落在周期 family。
3. 对相关候选做 center-vs-edge holdout 审计。
4. 对边缘区依然稳定的 periodic-family 候选给予 prior。
5. 对局部拟合好但边缘泛化差的非周期 surrogate 进行惩罚。
6. 需要时强制保留一部分 periodic challengers 进入 screened pool。

### 分数挂点
- Screen score:
  - `+ periodic_family_prior_weight * periodic_prior`
  - `- periodic_penalty`
- Group score:
  - `+ overall_periodic_disambiguation_score`
  - `- local_equivalence_penalty_mean`
- Outer objective:
  - `periodic_equivalence_score`
  - `periodic_equivalence_penalty`

### Artifact Schema
- 顶层字段：`periodic_equivalence_disambiguation`
- 报告载荷包括：
  - `protocol`
  - `parent_protocol`
  - `parent_mode_slot`
  - `canonical_mode_name`
  - `mode`
  - `phase_spectrum_audit_mode`
  - `periodic_family_prior_mode`
  - `periodic_candidate_screen_reserve`
  - `periodic_feature_names`
  - `coverage_score`
  - `overall_periodic_disambiguation_score`
  - `local_equivalence_penalty_mean`
  - `term_reports`

### 推荐 Lane 模式
- Lane id: `periodic_truth_lane`
- Screening protocol: `target_corr+residual_gain+semantic_novelty+consensus_prior+phase_spectrum`
- Challenger objective: `outer_objective+periodic_disambiguation`
- Pool expansion bias: `periodic_family_bias`

### 与通用 Symbolic 的关系
这是一个可选叶子机制。通用 symbolic regression 不一定需要它。

当它启用时，它修改的是 basis screening 与 outer basis-set evaluation，不改变 symbolic family 的通用父契约。
