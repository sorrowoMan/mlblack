## EquivalenceExpressionHandlingProtocol

### Purpose
Define how the symbolic family recognizes, groups, scores, and persists expressions that are different in surface form but equivalent or near-equivalent in mechanism meaning.

This is the parent protocol.

It should sit above any one special case such as periodic recovery.

### Why The Last Implementation Was Too Narrow
- It solved one important subtype: periodic truth versus local surrogate truth.
- But it did not yet define the full parent semantics for expression equivalence.
- So `PeriodicEquivalenceDisambiguationMechanism` looked like a standalone protocol, when it is better understood as one implemented child mode.

### Parent Scope
- semantic-family equivalence
- phase-equivalent recovery
- local-equivalence disambiguation
- representative-expression selection
- equivalence-aware truth recovery

### Formal Parent Fields
- `equivalence_expression_protocol`
- `equivalence_expression_mode`
- `equivalence_class_scope`
- artifact field: `equivalence_expression_handling`

### Child Modes
1. `periodic_mode`
   - implemented leaf mechanism: `PeriodicEquivalenceDisambiguationMechanism`
   - artifact slot: `periodic_equivalence_disambiguation`
   - role: distinguish true periodic terms from locally equivalent non-periodic surrogates

### Current Naming Decision
- Keep `EquivalenceExpressionHandlingProtocol` as the parent protocol name.
- Keep `PeriodicEquivalenceDisambiguationMechanism` as the leaf mechanism name.
- Stop treating `PeriodicEquivalenceDisambiguationMechanism` as if it were the whole equivalence-handling layer.
- Canonical child-mode name: `periodic_mode`

### Current Implemented Semantics
- semantic/family/phase-equivalent grouping
- truth-recovery aware equivalence accounting
- periodic-family prior during screening
- center-edge holdout style periodic audit

### Still Missing
- global canonical representative selection inside one equivalence class
- generalized local-equivalence handling beyond periodic families
- broader function-family collapse handling such as `sin` vs local spline or hinge surrogates

### Artifact / Runtime Expectation
The parent payload should now explain:
- which child modes are implemented
- which one is active
- what current narrowness remains

The leaf periodic payload should remain available for backward compatibility and audit detail.
