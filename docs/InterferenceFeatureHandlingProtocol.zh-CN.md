## InterferenceFeatureHandlingProtocol

### 目的
定义符号学习家族如何识别、抑制、惩罚、或在必要时策略性复用那些会干扰干净机制恢复的特征。

这是上位父协议。

它应该位于 anti-proxy 启发式、浅层非线性惩罚、区域残差修正之上。

### 为什么上一版实现做窄了
- 它已经实现了有用的 anti-proxy 与 residual-correction 逻辑。
- 但这些逻辑之前是以分散机制的方式出现，父级 interference 语义没有被正式写清楚。
- 所以 `RegionalCorrectionBasisProtocol` 看起来太像一个独立协议，实际上它更适合被理解成一个子模式。

### 父协议范围
- proxy suppression
- trivial nonlinearity rejection
- source-overlap control
- environment-invariance audit
- regional residual correction

### 正式字段
- `interference_feature_protocol`
- `interference_feature_mode`
- artifact 字段：`interference_feature_handling`

### 子模式
1. `proxy_suppression_mode`
   - 通过 `cross_explanatory_rejection_mode` 落地
   - 由 `proxy_group_policy` 提供支持
   - 作用：阻止 proxy 重复项同时进入最终 basis

2. `trivial_nonlinearity_rejection_mode`
   - 通过 `trivial_nonlinearity_penalty_mode` 落地
   - 作用：惩罚在已解释或 proxy-like 来源上做浅层非线性伪装

3. `regional_correction_mode`
   - 已实现的叶子机制：`RegionalCorrectionBasisProtocol`
   - 语义槽位：`regional_residual_correction`
   - artifact 槽位：`regional_correction_basis`
   - 作用：从结构化残差区域中恢复 gate / piecewise 修正 basis

### 当前命名决议
- 保留 `InterferenceFeatureHandlingProtocol` 作为父协议名。
- 保留 `RegionalCorrectionBasisProtocol` 作为叶子机制名，以兼容已有实现。
- 标准子模式名统一为：`regional_correction_mode`
- 当描述它在父协议中的功能位置时，语义槽位名统一为：`regional_residual_correction`

### 当前已经落地的语义
- proxy-aware cross-explanatory rejection
- outer objective 中的 trivial nonlinearity penalty
- 可切换的 environment audit
- 对 screened gate / piecewise candidates 的 residual-scan 提升

### 仍然缺的部分
- 更接近因果干预的 anti-proxy 逻辑
- 超出显式 proxy 组与 screened residual candidates 之外的通用 interference 处理
- 干扰清理之后真正重新打开的一轮 regime-specific structure search

### Artifact / Runtime 期望
父协议 payload 现在应该明确表达：
- 当前实现了哪些子模式
- 当前启用了哪个子模式
- 当前仍然窄在什么地方

regional correction 的叶子 payload 仍然保留，用于兼容和细节审计。
