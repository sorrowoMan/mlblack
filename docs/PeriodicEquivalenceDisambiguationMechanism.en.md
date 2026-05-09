## PeriodicEquivalenceDisambiguationMechanism

### Purpose
Prevent local non-periodic surrogates from replacing true periodic basis terms when the observed interval is narrow or when multiple families are locally fit-equivalent.

### Protocol Position
- Family: `symbolic`
- Parent protocol: `EquivalenceExpressionHandlingProtocol`
- Canonical child mode: `periodic_mode`
- Structure head: `basis_set`
- Primary stage: outer orthogonal structure search
- Secondary projection: artifact/report/runtime metadata

### Rehomed Position
This document now describes a leaf mechanism, not the whole equivalence-handling layer.

It is the currently implemented periodic specialization under `EquivalenceExpressionHandlingProtocol`.

### Core Inputs
- Raw feature matrix
- Candidate symbolic basis pool
- `search_hints.periodic_feature_names`
- Candidate semantic family / expression family

### Formal Fields
- `periodic_equivalence_protocol`
- `periodic_equivalence_disambiguation_mode`
- `phase_spectrum_audit_mode`
- `periodic_family_prior_mode`
- `periodic_family_prior_weight`
- `periodic_candidate_screen_reserve`
- `periodic_feature_names`

### Current Implemented Modes
- `periodic_equivalence_disambiguation_mode=center_edge_holdout_penalty`
- `phase_spectrum_audit_mode=center_edge_holdout_report`
- `periodic_family_prior_mode=semantic_family_boost`

### Selection Logic
1. Mark configured periodic features.
2. Detect whether a candidate touching that feature belongs to a periodic family.
3. Run a center-vs-edge holdout audit on candidate generalization.
4. Boost periodic-family candidates with stable edge behavior.
5. Penalize non-gate, non-periodic local surrogates with high center fit but weak edge transfer.
6. Reserve periodic challengers in the screened pool when requested.

### Score Hooks
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
- Top-level field: `periodic_equivalence_disambiguation`
- Report payload:
  - `protocol`
  - `mode`
  - `phase_spectrum_audit_mode`
  - `periodic_family_prior_mode`
  - `periodic_candidate_screen_reserve`
  - `periodic_feature_names`
  - `coverage_score`
  - `overall_periodic_disambiguation_score`
  - `local_equivalence_penalty_mean`
  - `term_reports`

### Recommended Lane Pattern
- Lane id: `periodic_truth_lane`
- Screening protocol: `target_corr+residual_gain+semantic_novelty+consensus_prior+phase_spectrum`
- Challenger objective: `outer_objective+periodic_disambiguation`
- Pool expansion bias: `periodic_family_bias`

### Relation To Generic Symbolic
This mechanism is optional. Generic symbolic regression can run without it. When enabled, it modifies basis screening and outer basis-set evaluation, but not the generic symbolic family contract itself.
