# Symbolic Learning Migration Inventory

Source repo: `C:\Users\hp\Desktop\mlblack`
Target repo: `C:\Users\hp\Desktop\新建文件夹 (2)`

This note records the symbolic-learning mechanisms before implementation work starts in the new optimization-first `mlblack`. The goal is to avoid losing the old symbolic engine concepts while keeping the new architecture clean.

## 1. Core Decision

Symbolic learning should be treated as nested standard scaffolds, not as a standalone `mlblack` trainer family.

The correct decomposition is:

```text
Stage 1: Orthogonal Symbolic Basis Search
  1A. nsgablack outer scaffold
      Search multi-symbol structures / operator choices / basis-set structure.
  1B. mlblack inner scaffold
      Fit the numeric parameters of each decoded basis-set candidate.
      Return orthogonality, stability, complexity, and audit metrics.

Stage 2: Basis-Conditioned Symbolic Task Search
  2A. nsgablack outer scaffold
      Search symbolic task expressions using Stage 1 basis outputs as atoms.
  2B. mlblack inner scaffold
      Fit numeric parameters for the fixed task expression.
      Return task metrics such as RMSE, coverage, logloss, calibration, etc.
```

So the practical view is two serial stages and four actual scaffold surfaces.

Important correction:

- Orthogonal symbolic search is not structure-only.
- A candidate basis such as `sin(p0 * x0 + p1)` cannot be scored for orthogonality until its parameters are fitted or otherwise resolved.
- Therefore each outer evaluation in Stage 1 contains an inner parameter optimization.

## 2. Old Repository Statistics

Structured scan of `C:\Users\hp\Desktop\mlblack`:

| scope | py files | LOC | classes | functions |
| --- | ---: | ---: | ---: | ---: |
| `core/symbolic` | 42 | 30751 | 63 | 813 |
| `core/trainers` | 15 | 10445 | 43 | 246 |
| `core/artifacts` | 10 | 1547 | 10 | 98 |
| `my_project/known_relation_symbolic` | 24 | 1240 | 2 | 23 |
| `nowcasting_work_ci/mlblack_side` | 27 | 6151 | 20 | 139 |
| `nowcasting_work_ci/nsgablack_side` | 14 | 471 | 1 | 16 |
| `examples/cases` | 41 | 2550 | 5 | 64 |

Keyword scan across 621 Python files, verified against the source repo on 2026-05-17:

| keyword | files | hits |
| --- | ---: | ---: |
| `symbolic` | 213 | 3885 |
| `orthogonal` | 71 | 1296 |
| `basis` | 77 | 3208 |
| `candidate_pool` | 31 | 205 |
| `dynamic_pool` | 31 | 115 |
| `gradient` | 46 | 502 |
| `expression` | 58 | 890 |
| `grammar` | 10 | 102 |
| `operator` | 11 | 81 |
| `primitive` | 27 | 300 |
| `decode` | 30 | 178 |
| `encode` | 39 | 264 |
| `artifact` | 150 | 2699 |
| `nsgablack` | 47 | 178 |
| `inner` | 79 | 1963 |
| `outer` | 124 | 1223 |

## 3. Mechanism Groups To Preserve

### 3.1 Symbolic Encoding / Decoding Engine

Old locations:

- `core/symbolic/symbolic_dsl.py`
- `core/symbolic/symbolic_structure_search.py`
- `core/symbolic/trainer_family.py`
- `core/symbolic/stage_head_protocol.py`

Key objects and functions:

- `ParameterSpec`
- `default_genome`, `default_genome_v2`
- `normalize_genome`
- `collect_parameter_specs`
- `evaluate_expression_numpy`
- `evaluate_expression_torch`
- `evaluate_genome_numpy`
- `evaluate_genome_torch`
- `expression_to_string`
- `genome_to_strings`

New interpretation:

```text
symbolic DSL = codec / decoder layer
symbolic genome = encoded decoder spec
expression tree = decoded callable structure
ParameterSpec = numeric slot contract for inner mlblack optimization
```

Migration target:

```text
mlblack/models/symbolic.py
mlblack/representations/codecs/symbolic.py
mlblack/representations/symbolic.py
```

Responsibilities in the new framework:

- Normalize symbolic expression specs.
- Collect parameter slots.
- Evaluate decoded expressions with numpy/torch backends.
- Convert expression trees to stable strings and artifact payloads.
- Avoid doing outer structure search inside this layer.

### 3.2 Multi-Symbol / Orthogonal Set Head

Old conceptual locations:

- `core/symbolic/orthogonal_basis_search.py`
- `core/trainers/symbolic_orthogonal_trainer.py`
- `core/trainers/symbolic_orthogonal_interval_trainer.py`
- `nowcasting_work_ci/mlblack_side/orthogonal_basis.py`
- `docs/MECHANISM_ORIENTED_BASIS_DISCOVERY_PROTOCOL.zh-CN.md`

Core idea:

```text
single symbolic decoder:
  one block -> one expression

multi-symbol head:
  full encoded candidate -> automatic block split
  block_i -> expression_i
  output -> OrthogonalBasisSet / OrthogonalAtomSet
```

New interpretation:

- The symbolic decoder itself should stay general.
- Orthogonal output is a head/output semantic issue.
- The head should output a set of expressions, not a scalar prediction.

Migration target:

```text
mlblack/representations/heads/symbolic.py
mlblack/models/symbolic.py
mlblack/integrations/nsgablack_symbolic/orthogonal_basis.py
```

Expected objects:

- `SymbolicExpressionModel`
- `SymbolicBasisSetModel`
- `MultiSymbolHead`
- `OrthogonalBasisSetHead`
- `OrthogonalBasisSetArtifact`

### 3.3 Function Pool / Candidate Pool

Old locations:

- `core/symbolic/feature_space/candidate_pool.py`
- `core/symbolic/feature_space/primitive_registry.py`
- `core/symbolic/feature_space/generation_grammar.py`
- `core/symbolic/feature_space/builder.py`
- `core/symbolic/feature_space/feature_bundle.py`

Key objects and functions:

- `CandidateTerm`
- `UnaryPrimitiveSpec`
- `PairGrammarRule`
- `PrimitiveRegistry`
- `GrammarCandidate`
- `ActivationPlan`
- `default_primitive_registry`
- `build_candidate_pool`
- `build_full_candidate_pool`
- `_build_candidate_pool`
- `_expand_candidate_pool_from_residual`
- `_prune_candidate_pool`
- `generate_unary_candidates`
- `generate_pair_candidates`
- `generate_recursive_unary_candidates`
- `generate_recursive_pair_candidates`

Mechanisms to preserve:

- Unary primitive families: polynomial, bounded, saturation, radial, trig, safe log/exp/ratio.
- Pair and interaction families: basic interaction, polynomial interaction, ratio, radial, saturation, rational, gate interaction.
- Candidate metadata: name, expression tree, complexity, family, feature indices, prior correlation.
- Feature bundle and candidate pool are not the final solver; they are search-space construction surfaces.

New interpretation:

```text
function pool = symbolic search-space provider
candidate pool = outer nsgablack representation/search context
primitive registry = symbolic operator catalog
```

Migration target:

```text
mlblack/pipeline/symbolic/primitives.py
mlblack/pipeline/symbolic/function_space.py
mlblack/pipeline/symbolic/pool_pipeline.py
mlblack/pipeline/symbolic/grammar.py
mlblack/integrations/nsgablack_symbolic/search_space.py
```

### 3.4 Dynamic Pool Expansion / Pruning

Old locations:

- `core/symbolic/feature_space/candidate_pool.py`
- `core/symbolic/feature_space/activation_config.py`
- `nowcasting_work_ci/mlblack_side/runtime/actions/outer_search_dynamic_pool.py`
- `bias/dynamic_pool_policy.py`

Key objects and functions:

- `DynamicActivationConfig`
- `parse_family_budget_csv`
- `resolve_dynamic_activation_kwargs`
- `maybe_expand_candidate_pool`
- `_build_dynamic_threshold_variants`
- `_expand_candidate_pool_from_residual`
- `_prune_candidate_pool`

Mechanisms to preserve:

- Start with a minimal candidate pool.
- Expand families based on residuals, gradient signals, regime/gate hints, or budget state.
- Prune redundant or low-value candidates.
- Maintain family budgets so dynamic expansion does not explode search cost.

New interpretation:

```text
dynamic pool = outer search-space mutation / stage policy
not an mlblack trainer behavior
```

Migration target:

```text
mlblack/pipeline/symbolic/pool_pipeline.py
mlblack/pipeline/symbolic/dynamic_pool.py
mlblack/bias/policies.py             # only soft hints if needed
mlblack/pipeline/conditional/         # reusable gate/conditional primitives only
```

Contract direction:

- `context_requires`: `outer.residuals`, `outer.search_state`, `resource.context`, `signal.pool`
- `context_provides`: `symbolic.candidate_pool`, `symbolic.pool_delta`
- `context_mutates`: `outer.search_state`

Exact key names should be finalized when the integration module is created.

### 3.5 Gradient Extension / Gradient-Guided Symbolic Expansion

Old locations:

- `core/symbolic/symbolic_gradient.py`
- `core/symbolic/gradient_parser.py`
- `core/symbolic/gradient_correction.py`
- `core/symbolic/symbolic_structure_search.py`

Key objects and functions:

- `GradientSignal`
- `GradientParser`
- `GradientCorrection`
- `GradientCorrectionConfig`
- `differentiate_expression_wrt_param`
- `differentiate_expression_wrt_feature`
- `gradient_formula_strings`
- `evaluate_gradient_numpy`
- `_build_grad_projection_candidates`

Mechanisms to preserve:

- Differentiate symbolic expressions with respect to parameters or features.
- Parse residual/error gradients into feature priority and cross-feature priority.
- Score candidates by gradient alignment.
- Generate gradient-projection candidate terms.
- Use gradient signal as guidance, not as a hidden replacement for the outer optimizer.

New interpretation:

```text
gradient extension = signal plugin / search-space bias / candidate scoring component
```

Migration target:

```text
mlblack/models/symbolic_gradient.py
mlblack/pipeline/symbolic/dynamic_pool.py
```

### 3.6 Orthogonal Basis Search Metrics

Old locations:

- `core/symbolic/orthogonal_basis_search.py`
- `nowcasting_work_ci/mlblack_side/orthogonal_basis.py`
- `core/symbolic/basis_consensus.py`
- `core/symbolic/structure_metadata.py`

Mechanisms to preserve:

- Pairwise absolute correlation.
- Feature-overlap penalties.
- Condition number / rank checks.
- Residual complementarity.
- Semantic deduplication.
- Fold stability.
- Basis semantics payloads.
- Basis overlap reports.
- Consensus basis tables.

New interpretation:

```text
orthogonal metrics = Stage 1 problem/evaluation objectives
basis consensus = artifact/report/capability layer
```

Migration target:

```text
mlblack/integrations/nsgablack_symbolic/orthogonal_problem.py
mlblack/integrations/nsgablack_symbolic/artifacts.py
```

### 3.7 Stage Contracts / Head Semantics

Old locations:

- `core/symbolic/structure_contract.py`
- `core/symbolic/search_mechanism_contract.py`
- `core/symbolic/stage_head_protocol.py`
- `docs/SYMBOLIC_REGIME_BASIS_ASSEMBLER_CONTRACTS.md`

Important old contracts:

- `SymbolicRegimeDiscoveryContract`
- `SymbolicBasisDiscoveryContract`
- `BudgetedSymbolicAssemblerContract`
- search mechanism contracts with consume/produce/mutate/checkpoint/replay fields

New interpretation:

```text
regime/basis/assembler = stage contracts
point/interval/probability = head semantics
```

Migration target:

```text
mlblack/integrations/nsgablack_symbolic/specs.py
mlblack/integrations/nsgablack_symbolic/builders.py
```

### 3.8 Artifacts / Report Surface

Old locations:

- `core/symbolic/artifact_schema.py`
- `core/artifacts/symbolic_artifact.py`
- `core/artifacts/symbolic_interval_artifact.py`
- `core/artifacts/piecewise_symbolic_interval_artifact.py`
- `core/symbolic/structure_metadata.py`
- `core/symbolic/basis_consensus.py`

Artifact sections to preserve:

- `regime_structure`
- `basis_structure`
- `assembler_structure`
- `piecewise_gate_basis`
- symbolic family block
- symbolic structure engine block
- head semantics
- complexity metrics
- stability metrics
- candidate lineage
- simplification trace
- truth contract recovery
- basis semantics / overlap / consensus payloads

New interpretation:

```text
symbolic artifact = nested plan report surface
not a trainer-private debug dict
```

Migration target:

```text
mlblack/core/artifacts.py             # generic typed artifact classes
mlblack/integrations/nsgablack_symbolic/artifacts.py
```

### 3.9 Existing nsgablack Boundary Evidence

Old repo already states the correct boundary in `AGENTS.md`:

```text
If the structure is fixed, use an mlblack trainer.
If the structure must be searched, it must enter an nsgablack outer solver.
mlblack owns candidate evaluation: fitting, metrics, objective payloads, constraints, symbolic artifacts, interval heads, and audit reports.
nsgablack owns outer orchestration, population/frontier management, multi-objective search, solver/adapter choice, and search trace.
```

Old integration/case locations:

- `my_project/known_relation_symbolic/`
- `nowcasting_work_ci/nsgablack_side/`
- `nowcasting_work_ci/mlblack_side/runtime/actions/outer_search_problem.py`
- `examples/cases/symbolic_kernel_digits_classification/`
- `examples/cases/symbolic_representation_object_digits_best_accuracy/`

New interpretation:

```text
mlblack core must stay nsgablack-free.
mlblack/integrations/nsgablack_symbolic may import nsgablack explicitly.
```

### 3.10 Old Strategy Mapping Table

This is the working table for migration. The key rule is: do not migrate old symbolic mechanisms by file name; migrate them by architectural role.

| old strategy / mechanism | new layer | canonical target | contract key surface | inputs | outputs | migration status |
| --- | --- | --- | --- | --- | --- | --- |
| symbolic expression DSL | model / codec | `models/symbolic.py`, `representations/codecs/symbolic.py` | `symbolic.expression_spec`, `symbolic.parameter_specs` | expression tree, feature names, parameter values | `SymbolicExpressionModel`, expression string, parameter slots | migrated first pass |
| multi-expression genome | representation / head | `representations/symbolic.py`, `representations/heads/symbolic.py` | `symbolic.genome`, `candidate.symbolic_basis_model` | multi-symbol genome, encoded state | basis-set model, block metadata | migrated first pass |
| primitive registry | pipeline catalog | `pipeline/symbolic/primitives.py` | `symbolic.primitive_registry` | primitive family config | unary/pair rule catalog | migrated first pass |
| function / candidate pool | symbolic pipeline | `pipeline/symbolic/function_space.py`, `pool_pipeline.py` | `symbolic.function_pool`, `symbolic.function_space` | `data.X_train`, `data.y_train`, feature names, registry | candidate terms, families, prior corr, complexity | migrated first pass |
| dynamic pool expansion | symbolic pipeline / policy | `pipeline/symbolic/dynamic_pool.py` | `symbolic.pool_delta`, `signal.pool` | residuals, gradient scores, budget ratio, gate scores | expanded/pruned pool, pool signal | migrated first pass |
| gradient parser / correction | model signal / pool guidance | `models/symbolic_gradient.py`, dynamic pool | `symbolic.gradient_signal`, `feedback.gradients` | expression, residuals, features, fitted params | feature scores, derivative values, parameter gradients | migrated first pass |
| graph expression cache | capability / cache | `integrations/nsgablack_symbolic/graph_cache.py` | `symbolic.graph_cache` | expression, X batch, param values | cached values, derivative expressions, cache stats | migrated |
| path memory | capability + search prior | `integrations/nsgablack_symbolic/path_memory.py` | `symbolic.path_memory` | expr keys, candidate outcomes, namespace | accept rate, seen count, path prior | migrated |
| overfit guard | search guard / score policy | `integrations/nsgablack_symbolic/overfit_guard.py` | `symbolic.overfit_guard` | train/valid metrics | triggered flag, reasons, penalty | migrated |
| candidate scoring | search policy / bias surface | `integrations/nsgablack_symbolic/search_policy.py` | `symbolic.candidate_score` | objectives, constraints, selected terms, metrics, path prior | scalar score, score parts, audit report | migrated and wired |
| seat guard / duplicate term guard | structure guard | `integrations/nsgablack_symbolic/structure_guard.py` | `symbolic.structure_guard` | selected term expression keys | duplicate count, penalty, reason | migrated first pass |
| feature reuse isolation | structure guard | `integrations/nsgablack_symbolic/structure_guard.py` | `symbolic.structure_guard` | selected term feature ids | feature reuse counts, excess, penalty | migrated first pass |
| value / chart stability | structure guard | `integrations/nsgablack_symbolic/structure_guard.py` | `symbolic.structure_guard` | selected term materialized values | stability score, instability penalty | migrated first pass |
| pole safety | structure guard | `integrations/nsgablack_symbolic/structure_guard.py` | `symbolic.structure_guard` | selected term materialized values | tail safety score, pole penalty | migrated first pass |
| native structure score | structure guard + scorer | `integrations/nsgablack_symbolic/structure_guard.py` | `symbolic.native_structure_score` | term family, complexity | native score, native bonus/penalty | migrated first pass |
| redundancy guard | structure guard | `integrations/nsgablack_symbolic/structure_guard.py` | `symbolic.structure_guard` | selected term value vectors | redundant pair count, penalty | migrated first pass |
| orthogonal basis objective | Stage 1 problem | `integrations/nsgablack_symbolic/orthogonal_problem.py`, `problems/symbolic.py` | `basis.metrics`, `feedback.objectives` | decoded basis set, fitted params, data | max corr, condition, rank, complexity | migrated |
| Stage 1 basis artifact | artifact layer | `integrations/nsgablack_symbolic/artifacts.py` | `artifact.symbolic_basis_ref`, `symbolic.artifact_schema` | best Stage 1 record | typed basis artifact, schema v2 | migrated |
| basis-conditioned task search | Stage 2 problem | `integrations/nsgablack_symbolic/task_symbolic_problem.py` | `task.metrics`, `artifact.symbolic_task_ref` | basis artifact, task expression, data | task objectives, fitted task artifact | migrated |
| interval task head | head + problem | `representations/heads/interval.py`, Stage 2 config | `candidate.interval_model`, `model.predict_interval` | expression state, interval head kind | coverage, width, miss constraints | migrated |
| probability / classification head | head + problem | `representations/heads/probability.py`, `problems/classification.py` | `candidate.probability_model`, `model.predict_proba` | expression state, class labels | logloss, error, AUC/F1/PR metrics | migrated |
| basis consensus / overlap report | artifact / reporting capability | `integrations/nsgablack_symbolic/basis_consensus.py` | `basis.consensus`, `basis.overlap_report`, `artifact.report` | basis artifacts, optional data matrix | consensus table, expression frequency, semantic overlap, atom-value overlap matrix | migrated second pass |
| branch evaluator / fold evaluator | evaluation capability | `integrations/nsgablack_symbolic/evaluation.py` | `symbolic.fold_report`, `symbolic.branch_report`, `symbolic.evaluation_events`, `stage.audit` | fitted artifact, folds/branches, data view, branch spec | fold reports, branch subset metrics, optional branch-local refit deltas, aggregate stability metrics | migrated as evaluator only; parallel/batch scheduling belongs to nsgablack L0 |
| simplification / equivalence trace | artifact audit producer | `integrations/nsgablack_symbolic/expression_audit.py` | `symbolic.simplification_trace`, `symbolic.equivalence_report`, `artifact.report` | expression specs, selected terms, optional data matrix | simplified expressions, algebra trace, canonical/value duplicate groups | migrated first pass |
| truth contract recovery | artifact audit producer | `integrations/nsgablack_symbolic/expression_audit.py` | `symbolic.truth_contract_recovery`, `artifact.report` | metadata truth contracts, selected terms, simplified expressions | matched contract count, exact-term recovery score, match rows | migrated first pass |
| replay / candidate lineage | replay artifact producer | `integrations/nsgablack_symbolic/replay.py`, `artifacts.py` | `symbolic.replay_record`, `symbolic.candidate_lineage`, `artifact.report` | Stage evaluation record, resource context, selected terms, score report | stable replay id, replay inputs, compressed inner/candidate/cache audit | migrated first pass |
| full artifact schema sections | artifact layer | `artifacts.py` | `symbolic.artifact_schema` | Stage records, guards, cache stats, lineage | symbolic plan schema sections, descriptor, lineage sections | expanded v2; deep truth-recovery scoring parity still partial |

Current first-pass migration rule:

```text
Pipeline builds the symbolic material.
Representation/codec/head decodes the material.
Problem evaluates the fitted decoded object.
Search policy scores and biases outer candidates.
Capability/cache/report persists and audits.
nsgablack adapter proposes and updates outer candidates.
```

## 4. Proposed New Target Layout

Do not recreate the old symbolic trainer family. Split old content into these surfaces:

```text
mlblack/
  models/
    symbolic.py

  pipeline/
    symbolic/
      primitives.py
      function_space.py
      pool_pipeline.py

  representations/
    symbolic.py
    codecs/
      symbolic.py
    heads/
      symbolic.py

  problems/
    symbolic.py                  # fixed symbolic model evaluation only

  integrations/
    nsgablack_symbolic/
      __init__.py
      specs.py                   # SymbolicGenomeSpec, SymbolicDecoderSpec, BasisStageSpec
      search_space.py            # adapts pipeline.symbolic FunctionPool to outer nsgablack representation
      grammar.py                 # grammar expansion and recursive generation, if it becomes outer-specific
      dynamic_pool.py            # optional outer-search pool expansion/pruning policies
      gradient_guidance.py       # optional nsgablack-facing gradient signal adapter
      structure_guard.py         # seat/reuse/native/pole/chart/redundancy structure guards
      search_policy.py           # candidate scoring and path-memory prior composition
      basis_consensus.py         # basis consensus, expression frequency and overlap report
      evaluation.py      # fitted-artifact fold and branch audit evaluator
      expression_audit.py        # simplification, equivalence, periodic/interference and truth-contract audit producer
      replay.py                  # stable candidate replay and compressed lineage record
      orthogonal_metrics.py      # pair corr, overlap, condition, stability
      orthogonal_problem.py      # Stage 1 outer problem using mlblack inner fit
      task_symbolic_problem.py   # Stage 2 outer problem using mlblack inner fit
      builders.py                # build nested serial plan
      artifacts.py               # symbolic plan artifacts
      README.md
```

## 5. Migration Order

### Phase 0: Cleanup Before Symbolic Migration

- Rename `MLBlackTrainingProxy.contract` to `training_contract` to avoid conflict with component `ComponentContract`.
- Keep `MLBlackTrainingProxy.contract = ComponentContract(...)` as the component-level contract.

### Phase 1: Core Decoder Surface

Move only generic fixed-decoder pieces into core `mlblack`:

- symbolic expression tree model
- symbolic parameter slot model
- expression evaluator for numpy first, torch optional later
- expression-to-string stable serialization
- symbolic codec / representation that decodes fixed symbolic specs
- symbolic heads: point expression, interval expression, multi-symbol / orthogonal set output

No outer structure search in this phase.

First-pass status:

- Added `mlblack/models/symbolic.py`.
- Added `mlblack/representations/codecs/symbolic.py`.
- Added `mlblack/representations/symbolic.py`.
- Added `mlblack/representations/heads/symbolic.py`.
- Added `mlblack/problems/symbolic.py`.
- Added `mlblack/pipeline/symbolic/*` as the function-space pipeline surface.
- Added `mlblack/integrations/nsgablack_symbolic/specs.py` as JSON-compatible integration metadata only.

Second-pass status:

- Added `mlblack/models/symbolic_gradient.py` for symbolic derivative expressions, array chain-rule derivatives, parameter Jacobians, MSE parameter gradients, and residual-gradient signals.
- Switched `SymbolicExpressionModel.parameter_gradient(...)` to analytic symbolic gradients with finite-difference fallback.
- Added `mlblack/pipeline/symbolic/dynamic_pool.py` for residual/gradient/gate expansion and budget/redundancy pruning.
- Added `mlblack/integrations/nsgablack_symbolic/orthogonal_problem.py` for Stage 1 outer basis search: outer candidate selects function-pool terms; inner mlblack random-search fitting optimizes symbolic parameters; output returns orthogonality/condition/complexity/rank objectives.

Full-pool and nested-plan status:

- Added `mlblack/pipeline/symbolic/grammar.py` and moved grammar concepts out of ad-hoc pipeline helpers: `GrammarCandidate`, `DynamicActivationConfig`, `ActivationPlan`, unary/pair generation, recursive unary/pair generation, conditional primitive lowering, family-budget parsing, and activation-plan selection.
- Reworked `FunctionPoolPipeline` so seed, unary, pair, recursive, and conditional candidates all flow through the shared grammar surface.
- Updated `DynamicFunctionPoolPipeline` to use the full registry family names and dynamic activation defaults instead of partial aliases.
- Added `mlblack/integrations/nsgablack_symbolic/artifacts.py` with `SymbolicArtifactSchema`, `OrthogonalBasisSetArtifact`, and `SymbolicTaskArtifact`.
- Added `mlblack/integrations/nsgablack_symbolic/task_symbolic_problem.py` for Stage 2: outer candidate selects basis-conditioned function-pool terms; inner mlblack fits the fixed task expression parameters with `SymbolicExpressionRepresentation + FixedSymbolicRegressionProblem + GradientDescentAdapter`.
- Added `mlblack/integrations/nsgablack_symbolic/search_space.py` for index-coded outer search-space decoding over a `FunctionPool`.
- Added `mlblack/integrations/nsgablack_symbolic/builders.py` with `build_symbolic_orthogonal_suite(...)` and `SymbolicOrthogonalNestedSuite`.
- Added core typed artifact classes for symbolic model and symbolic interval model surfaces.

### Phase 2: nsgablack Symbolic Integration Scaffold

Create `mlblack/integrations/nsgablack_symbolic` and define:

- outer symbolic genome/spec contracts
- function pool and grammar search-space providers
- outer candidate decode to multi-symbol/basis set
- stage contracts for basis search and basis-conditioned symbolic task search
- JSON-compatible inputs/outputs for cross-framework reports

### Phase 3: Stage 1 Orthogonal Basis Search

Implement the first nested stage:

```text
nsgablack outer:
  searches multi-symbol basis-set structure
mlblack inner:
  fits basis parameters for each candidate basis set
problem objectives:
  pairwise correlation
  feature overlap
  condition number / rank
  fold stability
  complexity
  resource cost
```

### Phase 4: Stage 2 Basis-Conditioned Symbolic Task Search

Implement the second nested stage:

```text
nsgablack outer:
  searches expressions over Stage 1 basis atoms
mlblack inner:
  fits task parameters for each fixed expression
problem objectives:
  task RMSE / MAE / R2
  interval coverage / width if interval head
  logloss / AUC / calibration if probability head
  expression complexity
  constraint violation
```

### Phase 5: Serial Nested Plan

Expose one builder:

```python
build_symbolic_orthogonal_suite(...)
```

It should assemble:

```text
serial(
  NestedOrthogonalBasisSearch(nsgablack outer + mlblack inner),
  NestedBasisConditionedSymbolicSearch(nsgablack outer + mlblack inner),
)
```

## 6. What Not To Do

- Do not create a new `SymbolicTrainer` as the main structure-search engine.
- Do not hide structure search inside mlblack trainer private loops.
- Do not mix function-pool expansion with data numericization.
- Do not put nsgablack imports into `mlblack.core`, `mlblack.representations`, `mlblack.problems`, or `mlblack.pipeline`.
- Do not treat orthogonality as a head-only issue; the output is a head issue, but orthogonal scoring is a problem/evaluation objective after parameter fitting.

## 7. Key Old Files To Revisit First

Highest priority:

- `core/symbolic/symbolic_dsl.py`
- `core/symbolic/feature_space/primitive_registry.py`
- `core/symbolic/feature_space/generation_grammar.py`
- `core/symbolic/feature_space/candidate_pool.py`
- `core/symbolic/feature_space/activation_config.py`
- `core/symbolic/symbolic_gradient.py`
- `core/symbolic/gradient_parser.py`
- `core/symbolic/gradient_correction.py`
- `core/symbolic/orthogonal_basis_search.py`
- `nowcasting_work_ci/mlblack_side/orthogonal_basis.py`
- `nowcasting_work_ci/mlblack_side/runtime/actions/outer_search_dynamic_pool.py`
- `nowcasting_work_ci/mlblack_side/runtime/actions/outer_search_problem.py`

Reference docs:

- `docs/MECHANISM_ORIENTED_BASIS_DISCOVERY_PROTOCOL.zh-CN.md`
- `docs/SYMBOLIC_REGIME_BASIS_ASSEMBLER_CONTRACTS.md`
- `docs/EquivalenceExpressionHandlingProtocol.zh-CN.md`
- `docs/InterferenceFeatureHandlingProtocol.zh-CN.md`
- `docs/PeriodicEquivalenceDisambiguationMechanism.zh-CN.md`
- `docs/RegionalCorrectionBasisProtocol.zh-CN.md`
- `nowcasting_work_ci/docs/README_ORTHOGONAL_BASIS.md`

## 8. Immediate Next Design Questions

These should be answered before implementation:

1. What is the minimal symbolic expression spec for the new codec?
2. How does `MultiSymbolHead` split an encoded candidate into expression blocks?
3. Which parameter slots are fitted by the Stage 1 inner mlblack trainer?
4. What exact objectives does Stage 1 return to nsgablack?
5. How is the Stage 1 `OrthogonalBasisSetArtifact` passed into Stage 2?
6. Which function-pool families are mandatory for the MVP?
7. Which dynamic-pool triggers are included in the first pass: residual, gradient, gate/regime, budget?
8. What is the minimal report payload required to debug four scaffold surfaces?



