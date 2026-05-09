## RegionalCorrectionBasisProtocol

### Purpose
Recover missing gate / piecewise / regional correction terms after the outer basis set already explains the global backbone but leaves structured residual regimes.

### Protocol Position
- Family: `symbolic`
- Parent protocol: `InterferenceFeatureHandlingProtocol`
- Canonical child mode: `regional_correction_mode`
- Semantic slot: `regional_residual_correction`
- Structure head: `expression`
- Primary stage: basis-conditioned inner symbolic assembly
- Input dependency: selected outer basis set

### Rehomed Position
This document now describes a leaf mechanism, not the whole interference-handling layer.

It is the currently implemented regional-correction specialization under `InterferenceFeatureHandlingProtocol`.

### Formal Fields
- `regional_correction_protocol`
- `residual_regime_identification_mode`
- `regional_correction_basis_mode`
- `regional_correction_promotion_mode`
- `regional_correction_feature_scope`
- `regional_correction_topk`
- `regional_correction_min_r2_gain`

### Current Implemented Modes
- `residual_regime_identification_mode=selected_basis_residual_scan`
- `regional_correction_basis_mode=screened_piecewise_candidates`
- `regional_correction_promotion_mode=topk_residual_gain`

### Current Mechanism
1. Fit the currently selected outer basis set.
2. Compute residual against the current target.
3. Revisit screened piecewise/gate candidates that match the configured feature scope.
4. Score each candidate by residual correlation and marginal `R^2` gain over the locked basis set.
5. Promote top candidates into the basis-conditioned inner assembler as extra basis objects.

### Score Hooks
- Candidate promotion score:
  - `0.70 * marginal_r2_gain + 0.30 * residual_abs_corr`
- Outer objective projection:
  - `regional_correction_score`

### Feature Scope Semantics
- `gate_only`
- `selected_features`
- `gate_or_selected`
- `all`

### Artifact Schema
- Top-level field: `regional_correction_basis`
- Report payload:
  - `protocol`
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

### Recommended Lane Pattern
- Lane id: `regional_correction_lane`
- Screening protocol: `target_corr+residual_gain+semantic_novelty+consensus_prior+regional_green_channel`
- Challenger objective: `outer_objective+regional_correction_gain`
- Pool expansion bias: `regional_gate_bias`

### Relation To Basis-Conditioned Symbolic
This protocol is only meaningful when symbolic regression is running in a basis-conditioned mode. It augments the object space; it does not replace the generic inner symbolic search contract.
