# Mechanism-Oriented Basis Discovery Protocol

Design draft for promoting orthogonal basis generation from trainer-local logic to a formal `mlblack` symbolic-family protocol.

This document uses two coordinated names:

- formal mechanism object: `OrthogonalBasisGenerationMechanism`
- system-level protocol: `Mechanism-Oriented Basis Discovery Protocol`

Status:

- design draft
- not yet fully materialized as a cataloged formal framework object
- intended to guide future symbolic-family implementation, runtime surfaces, and experiment evaluation

---

## 1. Why This Exists

The central symbolic problem is often not only:

- how to fit the final expression

It is first:

- how to generate the right basis coordinates

If basis generation is weak, downstream symbolic regression can still fit well while recovering:

- proxy structure
- redundant structure
- unstable structure
- semantically duplicated structure

So the key upgrade is:

- treat orthogonal basis generation as a first-class mechanism
- not as a trainer-private screening helper
- not as an implementation detail buried inside one symbolic preset

---

## 2. Problem Statement

For symbolic learning, "orthogonal" should not be reduced to low pairwise correlation.

In this protocol, orthogonality is relative and multi-dimensional.

A good basis set should be:

1. statistically complementary
2. residual-useful
3. semantically non-redundant
4. structurally composable
5. regime-aware under piecewise or gate behavior
6. stable enough to survive multi-run consensus

The practical research claim is:

- before searching for a final symbolic expression, the system should search for a good mechanism coordinate system

---

## 3. Position In `mlblack`

This protocol belongs to `mlblack`, not to `nsgablack`.

Reason:

- `mlblack` owns symbolic-family training semantics
- `mlblack` owns basis generation semantics
- `mlblack` owns basis meaning, basis equivalence, and basis artifact schema
- `nsgablack` should orchestrate and inspect these runs, not define the symbolic basis semantics

Framework classification:

- `family = symbolic`
- `component = OrthogonalBasisGenerationMechanism`
- `structure_head = basis_set | expression`
- `prediction_head = point | interval`
- mechanism binding level = `defining` or `bound`, not merely optional

This is important:

- it should not become a fake new family
- it should not live only as one preset-specific trainer branch
- `basis_set` should be treated as a symbolic-specific structure head, not as a new family

---

## 4. Relationship To Existing Symbolic Contracts

The current symbolic structure stack already has:

- `SymbolicRegimeDiscoveryContract`
- `SymbolicBasisDiscoveryContract`
- `BudgetedSymbolicAssemblerContract`

This protocol is positioned as:

- the system-level design and contract guide for how `SymbolicBasisDiscoveryContract` should behave when the symbolic family uses orthogonal-basis-first discovery

So the intended mapping is:

1. `SymbolicBasisDiscoveryContract`
   - family-facing structure-stage contract
2. `OrthogonalBasisGenerationMechanism`
   - concrete mechanism layer that realizes the contract
3. `Mechanism-Oriented Basis Discovery Protocol`
   - the higher-level system protocol that fixes semantics, scoring, persistence, and runtime-surface expectations

---

## 5. Core Thesis

The most accurate interpretation is not:

- one upstream searcher plus one downstream trainer

It is:

- one symbolic family with a shared backbone
- different stage objectives
- different structure heads

### 5.1 Same Backbone, Different Head

Both stages are still symbolic regression.

They share:

- the same dataset semantics
- the same symbolic search backbone
- the same general loop such as `propose -> evaluate -> update -> select`

What changes across stages is mainly:

- the search object
- the evaluation contract
- the output head

Stage A:

- `structure_head = basis_set`
- search object: primitive combinations over base feature space
- output object: `selected_basis`
- objective focus: orthogonality, residual complementarity, semantic novelty, consensus alignment

Stage B:

- `structure_head = expression`
- search object: expressions over the selected basis space
- output object: final symbolic expression
- objective focus: fit, simplicity, budget, and truth recovery

When Stage B emits a predictive artifact, it may additionally use:

- `prediction_head = point | interval`

### 5.2 Two Legitimate Execution Protocols

This leads to two legitimate symbolic-family execution modes.

`generic symbolic protocol`

- no upstream basis stage
- directly train with `structure_head = expression`
- search over the generic symbolic space

`basis-orchestrated symbolic protocol`

1. run a symbolic stage with `structure_head = basis_set`
2. emit `selected_basis`
3. run a symbolic stage with `structure_head = expression`
4. assemble the final expression inside the basis-conditioned space

### 5.3 Why This Matters

Under this interpretation, basis discovery is not preprocessing.

It is a formal symbolic-training stage whose artifact is a basis set rather than a final expression.

So the search target becomes:

- first: find a good mechanism basis space
- then: fit expressions inside that space

This is a stricter and more original target than ordinary symbolic regression.

---

## 6. Formal Object

### 6.1 `OrthogonalBasisGenerationMechanism`

This is the proposed formal mechanism object.

It should own:

- candidate basis proposal
- basis equivalence normalization
- basis-set search
- basis scoring
- overlap and redundancy reporting
- regime-aware piecewise or gate basis handling

It should not own:

- final head semantics
- report-writing side effects
- experiment persistence side effects
- orchestration across external outer solvers

Important clarification:

- the mechanism object is not itself the `basis_set` head
- it is the reusable component that conditions a symbolic stage whose `structure_head` is `basis_set`

### 6.2 `Mechanism-Oriented Basis Discovery Protocol`

This protocol owns the system-level rules for:

- what basis generation consumes
- what it emits
- how it is scored
- how it is persisted
- how it affects family identity
- how it is surfaced to runtime DB and experiment UI
- when a symbolic stage should be read as `basis_set` head versus `expression` head

---

## 7. I/O Contract

### 7.1 Consume

Required:

- feature matrix or feature providers
- target values
- symbolic primitive library
- basis search configuration

Optional:

- residual baseline state
- fold or resample partitions
- regime partition hints
- consensus prior rows
- locked core seed rows
- semantic equivalence rules
- known-relation truth contract for benchmark evaluation

### 7.2 Produce

Required outputs:

- `structure_head = basis_set`
- `basis_candidates`
- `basis_scores`
- `selected_basis`
- `basis_overlap_report`
- `basis_semantics`
- `basis_generation_protocol`

Optional outputs:

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

## 8. Composition Contract

Preferred declarations:

- `requires`
- `provides`
- `mutates`
- `cache`

Recommended shape:

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

This protocol should not rely mainly on scattered `hasattr(...)` checks or trainer-private context guessing.

---

## 9. Persistence Contract

The following outputs must stay distinct:

- artifact-facing basis structure
- trainer_state or resume state
- search trace or report payload
- experiment/runtime surface projection

Recommended artifact sections:

- `basis_structure`
- `piecewise_gate_basis`
- `basis_overlap_report`
- `basis_semantics`

Recommended trainer-state sections:

- search frontier
- candidate score cache
- equivalence cache
- locked seed state

Recommended report sections:

- basis ranking table
- candidate screen diagnostics
- semantic dedup diagnostics
- residual complementarity diagnostics

---

## 10. Compatibility Contract

This mechanism should affect symbolic family identity.

It should participate in:

- `family_signature_payload()`
- warm-start compatibility checks
- resume drift checks
- experiment surface contract layers

At minimum, compatibility-relevant fields should include:

- `basis_generation_protocol`
- `screening_protocol`
- `outer_search_protocol`
- primitive library signature
- equivalence mode
- regime handling mode
- locked-core seeding mode

If these drift, the system should be able to explain:

- what moved
- whether the move is safe
- whether a warm-start or resume should be rejected

---

## 11. Orthogonality Definition

This protocol uses a richer definition of orthogonality.

### 11.1 Statistical Orthogonality

- low harmful correlation
- low overlap in explained variance

### 11.2 Residual Orthogonality

- each basis term should explain residual not already captured by current selected basis

### 11.3 Semantic Orthogonality

- different symbolic forms that mean the same proxy should not count as novel

### 11.4 Structural Orthogonality

- basis terms should remain composable into a compact second-stage symbolic expression

### 11.5 Regime Orthogonality

- under piecewise or gate behavior, basis terms for different regimes should not collapse into one misleading global proxy

### 11.6 Consensus Orthogonality

- a basis term that repeatedly survives across runs has stronger claim to being a core mechanism coordinate

---

## 12. Scoring Stack

The screening and search stack should combine several signals.

### 12.1 Candidate Screen

Current target design:

- `target corr`
- `residual gain`
- `semantic novelty`
- `consensus prior`

### 12.2 Basis-Set Search Objective

Recommended basis-set objective should include:

- mean candidate screen score
- orthogonality score
- residual complementarity
- semantic uniqueness
- pairwise correlation penalty
- feature overlap penalty
- family diversity bonus
- regime coverage bonus
- piecewise or gate bonus

### 12.3 Inner Validation

Each proposed basis set should trigger a smaller inner symbolic assembler.

The outer score should not rely only on static basis diagnostics.

It should also look at:

- `inner_fit_score`
- `assembly_budget_usage`
- `assembly_trace_quality`
- mechanism-faithfulness metrics when truth contracts are available

### 12.4 `search_input_space`

This protocol should make the active search input space explicit.

Recommended values:

- `raw_feature_space`
- `basis_object_space`

Recommended semantics:

- `structure_head = basis_set` should normally use `raw_feature_space`
- generic `structure_head = expression` may also use `raw_feature_space`
- basis-conditioned `structure_head = expression` should use `basis_object_space`

This field is important because it clarifies whether the current symbolic stage is:

- searching directly over raw features
- or searching over basis objects emitted by an earlier symbolic stage

### 12.5 `pool_expansion_unit`

Dynamic pool expansion should also declare its atomic expansion unit.

Recommended values:

- `raw_feature`
- `basis_object`

Recommended semantics:

- in raw-feature stages, expansion units are raw features and their immediate symbolic seeds
- in basis-conditioned stages, expansion units are basis objects

If a basis-conditioned stage receives:

- `x1*x2`
- `sin(x3)`
- `x4`

as selected basis objects, the expansion logic should treat those objects as atomic units.

It should not freely decompose them back into:

- `x1`
- `x2`
- `x3`
- `x4`

unless an explicit escape or decomposition policy says otherwise.

### 12.6 `gradient_guidance_mode`

Gradient guidance should be explicit and stage-aware.

Recommended values:

- `off`
- `raw_feature_gradient`
- `basis_object_gradient`
- `hybrid`

Recommended semantics:

- `raw_feature_gradient`: gradients guide candidate generation in raw feature space
- `basis_object_gradient`: gradients guide candidate generation over selected basis objects
- `hybrid`: allow both views, but only when explicitly intended

The key rule is:

- when `search_input_space = basis_object_space`, the default gradient view should also be object-level

This means the gradient signal should help rank and expand:

- basis objects
- basis-object pairs
- basis-object transforms

rather than implicitly dropping back to raw feature-level expansion.

---

## 13. Layered Search View

This protocol suggests a nested search interpretation.

### 13.1 Outer Layer

Searches over basis-generation programs or basis-set hypotheses.

### 13.2 Middle Layer

Scores basis-set hypotheses using orthogonality, novelty, residual complementarity, and consensus alignment.

### 13.3 Inner Layer

Runs a budgeted symbolic assembler over the selected basis set.

This means the "basis mechanism" is not just preprocessing.

It is a structured search layer above the final expression search.

---

## 14. Piecewise / Gate Basis

Piecewise or gate-conditioned basis should be first-class.

This protocol should directly support:

- gate-conditioned basis rows
- regime-local basis rows
- breakpoint-sensitive basis candidates
- failed regime diagnostics
- local-to-global aggregation summaries

The system must not force all structure into one global basis vocabulary when the data clearly contain regime changes.

---

## 15. Equivalence Semantics

Basis equivalence should be explicit.

At minimum, the protocol should support:

- `exact`
- `phase-equivalent`
- `family-level`

This matters because many symbolic structures are:

- syntactically different
- functionally similar
- mechanism-related

Without formal equivalence handling, consensus and truth recovery both become unstable.

---

## 16. Runtime Surface And Experiment Surface Projection

This mechanism should project into runtime and experiment surfaces.

Recommended fields:

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
- `equivalence_expression_protocol`
- `equivalence_expression_mode`
- `equivalence_class_scope`
- `interference_feature_protocol`
- `interference_feature_mode`
- `cross_explanatory_rejection_mode`
- `trivial_nonlinearity_penalty_mode`
- `environment_invariance_audit_mode`

Current protocol hierarchy:

- parent: `EquivalenceExpressionHandlingProtocol`
  - child mode: `periodic_mode`
  - current leaf mechanism: `PeriodicEquivalenceDisambiguationMechanism`
- parent: `InterferenceFeatureHandlingProtocol`
  - child modes: `proxy_suppression_mode`, `trivial_nonlinearity_rejection_mode`, `regional_correction_mode`
  - current leaf mechanism for regional correction: `RegionalCorrectionBasisProtocol`

These should appear in:

- run metadata
- artifact metadata
- experiment tracker materializations
- dashboard filters and detail panels

---

## 17. Benchmark Contract

This protocol is especially meaningful on known-relation suites.

Recommended benchmark families:

- `ohm_like`
- `ideal_gas_like`
- `arrhenius_gate_like`
- `periodic_gate_like`
- `redundant_proxy_control`

Recommended evaluation levels:

- fit metrics
- `exact_basis_hit_score`
- `exact_term_recovery_score`
- `phase_equivalent_term_recovery_score`
- `family_level_term_recovery_score`

For this mechanism, the benchmark question is not only:

- did the model fit well

It is also:

- did the basis mechanism recover the right mechanism coordinates

---

## 18. Current Research Diagnosis

The motivating failure case is already visible:

- a system can stabilize
- consensus can become strong
- locked-core can become stable
- yet the stable structure can still be the wrong proxy family

So the next frontier is not merely more budget.

It is:

- better mechanism-aware basis generation
- better anti-proxy semantic dedup
- better regime-aware basis candidates
- better outer preference for truth-like mechanism families

---

## 19. Promotion Plan

To promote this from design draft to formal framework surface, the likely steps are:

1. define stable mechanism metadata keys
2. bind the mechanism into symbolic family structure contracts
3. materialize mechanism fields into artifact schema
4. project mechanism fields into experiment runtime surfaces
5. expose mechanism filters in experiment dashboard and catalog-like surfaces
6. add a catalog entry as a `component` when the surface is stable enough

Important:

- this document proposes a formal direction
- it does not claim the full framework materialization is already complete

---

## 20. Non-Goals

This protocol does not mean:

- every symbolic preset becomes a new family
- every basis heuristic should be frozen too early
- orchestration belongs inside `mlblack` instead of `nsgablack`
- report or persistence side effects should move into trainer internals

The point is stricter:

- make basis generation a formal symbolic-family mechanism
- keep orchestration, persistence, and product surfaces aligned around it

---

## 21. Two Formal Guard Mechanisms

The current protocol should explicitly distinguish two mechanism-level guards.

### 21.1 `EquivalenceExpressionHandlingMechanism`

Purpose:

- cluster syntactically different but mechanism-near terms
- avoid counting phase-equivalent or proxy-equivalent terms as false novelty
- reduce pointless combinatorial branching inside one local equivalence class

Recommended protocol keys:

- `equivalence_expression_protocol`
- `equivalence_expression_mode`
- `equivalence_class_scope`

Current contract interpretation:

- this mechanism sits between candidate proposal and consensus interpretation
- it is responsible for empirical / residual / semantic equivalence handling
- it does not yet imply perfect representative replacement inside an equivalence class

### 21.2 `InterferenceFeatureHandlingMechanism`

Purpose:

- detect or suppress proxy-like and source-overlapping basis terms
- prevent trivial nonlinear disguises on highly overlapping features
- force the system to search for genuinely new mechanism coordinates

Recommended protocol keys:

- `interference_feature_protocol`
- `interference_feature_mode`
- `cross_explanatory_rejection_mode`
- `trivial_nonlinearity_penalty_mode`
- `environment_invariance_audit_mode`

Current contract interpretation:

- this mechanism is formalized before it is fully hardened
- current implementations are still heuristic-first
- the protocol surface should be explicit now so later anti-proxy upgrades do not fragment schema or runtime naming
