## EquivalenceExpressionHandlingProtocol

### 目的
定义符号学习家族如何识别、归并、评分、持久化那些“表面写法不同，但机制上等价或近似等价”的表达式。

这是上位父协议。

它应该位于任何单一特例之上，例如周期项恢复。

### 为什么上一版实现做窄了
- 它解决了一个很重要的子问题：真实周期项与局部替代表达的区分。
- 但它还没有把“表达式等价处理”的父级语义完整定义出来。
- 所以 `PeriodicEquivalenceDisambiguationMechanism` 看起来像一个独立大协议，实际上更适合被理解为一个已落地的子模式。

### 父协议范围
- semantic-family equivalence
- phase-equivalent recovery
- local-equivalence disambiguation
- representative-expression selection
- equivalence-aware truth recovery

### 正式字段
- `equivalence_expression_protocol`
- `equivalence_expression_mode`
- `equivalence_class_scope`
- artifact 字段：`equivalence_expression_handling`

### 子模式
1. `periodic_mode`
   - 已实现的叶子机制：`PeriodicEquivalenceDisambiguationMechanism`
   - artifact 槽位：`periodic_equivalence_disambiguation`
   - 作用：把真实周期项与局部等价的非周期替代项区分开

### 当前命名决议
- 保留 `EquivalenceExpressionHandlingProtocol` 作为父协议名。
- 保留 `PeriodicEquivalenceDisambiguationMechanism` 作为叶子机制名。
- 不再把 `PeriodicEquivalenceDisambiguationMechanism` 误当成整个 equivalence-handling 层。
- 标准子模式名统一为：`periodic_mode`

### 当前已经落地的语义
- semantic / family / phase-equivalent 归并
- 带真值契约的等价恢复统计
- screening 阶段的 periodic family prior
- center-edge holdout 风格的 periodic audit

### 仍然缺的部分
- 一个等价类内部的全局 canonical representative 选择
- 超出 periodic family 之外的通用 local-equivalence 处理
- 更广义的函数族等价坍缩，例如 `sin` 与局部 spline / hinge surrogate 的处理

### Artifact / Runtime 期望
父协议 payload 现在应该明确表达：
- 当前实现了哪些子模式
- 当前启用了哪个子模式
- 当前仍然窄在什么地方

周期叶子 payload 仍然保留，用于兼容和审计细节。
