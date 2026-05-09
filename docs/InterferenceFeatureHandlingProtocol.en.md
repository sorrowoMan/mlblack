## InterferenceFeatureHandlingProtocol

### Purpose
Define how the symbolic family detects, suppresses, penalizes, or strategically reuses features that interfere with clean mechanism recovery.

This is the parent protocol.

It should sit above anti-proxy heuristics, trivial nonlinearity penalties, and regional residual correction.

### Why The Last Implementation Was Too Narrow
- It implemented useful anti-proxy and residual-correction logic.
- But it expressed them as separate mechanisms without making the parent interference semantics explicit.
- So `RegionalCorrectionBasisProtocol` looked too standalone, when it is better understood as one child mode.

### Parent Scope
- proxy suppression
- trivial nonlinearity rejection
- source-overlap control
- environment-invariance audit
- regional residual correction

### Formal Parent Fields
- `interference_feature_protocol`
- `interference_feature_mode`
- artifact field: `interference_feature_handling`

### Child Modes
1. `proxy_suppression_mode`
   - implemented through `cross_explanatory_rejection_mode`
   - supported by `proxy_group_policy`
   - role: stop proxy duplicates from coexisting in the selected basis set

2. `trivial_nonlinearity_rejection_mode`
   - implemented through `trivial_nonlinearity_penalty_mode`
   - role: penalize shallow nonlinear disguise on already-explained or proxy-like sources

3. `regional_correction_mode`
   - implemented leaf mechanism: `RegionalCorrectionBasisProtocol`
   - semantic slot: `regional_residual_correction`
   - artifact slot: `regional_correction_basis`
   - role: recover gate or piecewise correction basis from structured residual regimes

### Current Naming Decision
- Keep `InterferenceFeatureHandlingProtocol` as the parent protocol name.
- Keep `RegionalCorrectionBasisProtocol` as the leaf mechanism name for compatibility.
- Use `regional_correction_mode` as the canonical child-mode name.
- Use `regional_residual_correction` as the semantic slot name when describing its role in the parent protocol.

### Current Implemented Semantics
- proxy-aware cross-explanatory rejection
- trivial nonlinearity penalty inside the outer objective
- switchable environment audit
- residual-scan promotion of screened gate or piecewise candidates

### Still Missing
- full causal intervention style anti-proxy logic
- generalized interference handling beyond configured proxy groups and screened residual candidates
- reopened regime-specific structure search after interference pruning

### Artifact / Runtime Expectation
The parent payload should now explain:
- which child modes are implemented
- which one is active
- what current narrowness remains

The leaf regional-correction payload should remain available for backward compatibility and detailed audit.
