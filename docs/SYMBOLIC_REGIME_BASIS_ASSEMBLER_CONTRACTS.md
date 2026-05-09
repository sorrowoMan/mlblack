# Symbolic Regime / Basis / Assembler Contracts

See also:

- `docs/MECHANISM_ORIENTED_BASIS_DISCOVERY_PROTOCOL.md`

That protocol should be read as the higher-level system design for
`SymbolicBasisDiscoveryContract` when the symbolic family uses an
orthogonal-basis-first basis discovery path.

## Why This Exists

`mlblack` symbolic family should not be framed as:

- symbolic search -> simple head

The more accurate framing is:

1. regime discovery
2. basis discovery
3. small-budget symbolic assembly
4. head semantics (`point` / `interval`)

This matters because the expensive and interesting part is often not the final readout.
The hard part can be discovering:

- when one global expression is insufficient
- which basis terms are low-overlap / reusable
- whether local piecewise or gate-conditioned structure is needed

## Formal Contracts

The symbolic family now exposes three structure-stage contracts in `core.symbolic.structure_contract`:

- `SymbolicRegimeDiscoveryContract`
- `SymbolicBasisDiscoveryContract`
- `BudgetedSymbolicAssemblerContract`

They follow the same formal style as symbolic search mechanism contracts:

- `consume`
- `produce`
- `mutate`
- `checkpoint`
- `replay`
- `checkpointable`
- `replayable`
- `affects_family_signature`

## Intended Semantics

### 1. `SymbolicRegimeDiscoveryContract`

Owns the decision of whether the symbolic artifact stays global or branches into local regimes.

Typical outputs:

- `global_regime`
- `regime_partition`
- `regime_manifest`
- `gate_basis`

### 2. `SymbolicBasisDiscoveryContract`

Owns discovery of reusable symbolic basis terms before final assembly.

Typical objectives:

- low overlap
- low redundancy
- fold stability
- semantic reusability

Typical outputs:

- `basis_candidates`
- `basis_scores`
- `selected_basis`
- `basis_overlap_report`
- `basis_semantics`

### 3. `BudgetedSymbolicAssemblerContract`

Owns the deliberately smaller second-stage symbolic regression over discovered basis terms.

Typical outputs:

- `assembled_expression`
- `assembly_score`
- `assembly_trace`
- `assembly_budget_usage`

This is where we explicitly keep the final symbolic composition budget small, instead of mixing the expensive basis search and final assembly into one opaque stage.

## Family Identity

These contracts are part of `SymbolicTrainerFamilySpec.family_signature_payload()`.

That means:

- they are not documentation-only
- they affect symbolic family identity
- warm start / resume compatibility can reject drift in these contracts

Compatibility drift is now expected to report not only hash mismatch, but also which search or structure contracts moved.

## Artifact Schema

Symbolic artifacts now expose four first-class structure sections:

- `regime_structure`
- `basis_structure`
- `assembler_structure`
- `piecewise_gate_basis`

This applies to:

- point symbolic artifacts
- interval symbolic artifacts
- piecewise interval symbolic artifacts

The goal is to keep regime/basis/assembler structure visible in artifact/report/catalog surfaces, instead of hiding it only in raw metadata or debug traces.

## Piecewise / Gate Basis

Piecewise and gate-conditioned structure is treated as first-class symbolic structure, not as an afterthought.

Relevant artifact fields include:

- gate feature names / indices
- selected regime keys
- failed regimes
- local basis counts
- gate-conditioned basis payload

## Architectural Mapping

Inside `mlblack` terminology, this should be read as:

- `family = symbolic`
- `component = regime discovery / basis discovery / assembler policies`
- `head = point / interval / future quantile or distribution semantics`

So we do not want to scatter these into separate fake algorithm families.
They are structured stages inside one formal symbolic family.
