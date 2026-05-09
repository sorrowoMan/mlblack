## RegionalCorrectionBasisProtocol

### 目标
当 outer basis 已经解释了主干机制，但残差里仍然存在明显的 gate / piecewise / 区域修正结构时，把这些“局部修正 basis”正式提回到 basis-conditioned 的内层 symbolic assembler。

### 协议位置
- Family: `symbolic`
- Parent protocol: `InterferenceFeatureHandlingProtocol`
- Canonical child mode: `regional_correction_mode`
- Semantic slot: `regional_residual_correction`
- Structure head: `expression`
- Primary stage: basis-conditioned inner symbolic assembly
- Input dependency: selected outer basis set

### Rehomed Position
这份文档现在描述的是叶子机制，而不是整个 interference-handling 层。

它是 `InterferenceFeatureHandlingProtocol` 下当前已经落地的 regional correction specialization。

### 正式字段
- `regional_correction_protocol`
- `residual_regime_identification_mode`
- `regional_correction_basis_mode`
- `regional_correction_promotion_mode`
- `regional_correction_feature_scope`
- `regional_correction_topk`
- `regional_correction_min_r2_gain`

### 当前已实现模式
- `residual_regime_identification_mode=selected_basis_residual_scan`
- `regional_correction_basis_mode=screened_piecewise_candidates`
- `regional_correction_promotion_mode=topk_residual_gain`

### 当前机制
1. 先拟合当前选中的 outer basis set。
2. 计算当前 target 的 residual。
3. 回看 screened 过的 piecewise / gate candidates，并按配置的 feature scope 过滤。
4. 用 residual correlation 与 marginal `R^2` gain 给候选打分。
5. 将最强的候选作为额外 basis objects 提升进 basis-conditioned inner assembler。

### 分数挂点
- Candidate promotion score:
  - `0.70 * marginal_r2_gain + 0.30 * residual_abs_corr`
- Outer objective projection:
  - `regional_correction_score`

### Feature Scope 语义
- `gate_only`
- `selected_features`
- `gate_or_selected`
- `all`

### Artifact Schema
- 顶层字段：`regional_correction_basis`
- 报告载荷包括：
  - `protocol`
  - `parent_protocol`
  - `parent_mode_slot`
  - `canonical_mode_name`
  - `semantic_slot_name`
  - `residual_regime_identification_mode`
  - `regional_correction_basis_mode`
  - `regional_correction_promotion_mode`
  - `regional_correction_feature_scope`
  - `regional_correction_topk`
  - `regional_correction_min_r2_gain`
  - `candidate_pool_count`
  - `promoted_count`
  - `regional_correction_score`
  - `promoted_candidates`

### 推荐 Lane 模式
- Lane id: `regional_correction_lane`
- Screening protocol: `target_corr+residual_gain+semantic_novelty+consensus_prior+regional_green_channel`
- Challenger objective: `outer_objective+regional_correction_gain`
- Pool expansion bias: `regional_gate_bias`

### 与 Basis-Conditioned Symbolic 的关系
只有在 basis-conditioned symbolic 模式下，这个叶子机制才真正有意义。

它的作用是扩展 object space，而不是替代通用 inner symbolic search contract。
