# AI_DEVELOPMENT_GUIDELINES

## 1) Purpose

This document is the detailed development guide for AI collaborators in `mlblack`.

The root `AGENTS.md` is the first-read contract.
This document expands that contract into design and implementation rules.

The intent is to keep future AI work aligned on:

1. componentization
2. formal contracts
3. field alignment
4. classification discipline
5. catalog as a formal surface
6. catalog persistence into database

## 2) What `mlblack` Is

`mlblack` is a composable learning framework.

It is not:

- a flat list of trainer names
- a bag of unrelated algorithm wrappers
- a place where UI, persistence, routing, and training semantics are mixed together

It should be understood as a system that separates:

- training backbone
- enhancement mechanisms
- output semantics
- evaluation support
- side effects
- discoverability surface

## 3) Formal Classification Discipline

Every new formal object should first be classified into one of five kinds.

### 3.1 `family`

Defines how a function family is fit.

Current formal family layer:

- `linear family`
- `tree family`
- `tree_boosting family`
- `neural family`
- `symbolic family`

### 3.2 `component`

Enhances a family without becoming the family itself.

Typical examples:

- regularization
- dropout
- sampling policy
- state signal view
- routing primitive
- warm-start policy

### 3.3 `head`

Defines output semantics.

Current formal heads:

- `point`
- `interval`

Likely future heads:

- `quantile`
- `distribution`
- `classification logits`

### 3.4 `provider`

Provides external support to training/evaluation paths.

Typical examples:

- bridge
- proxy
- cache-backed evaluator
- numerical teacher / solver

### 3.5 `plugin`

Owns side effects and observability.

Typical examples:

- experiment tracker
- trainer state checkpoint
- report writer
- reproducibility
- runtime audit

## 4) Componentization Rules

The framework should be decomposed by responsibility, not by convenience.

### 4.1 Good componentization

Prefer explicit modules for:

- backbone family logic
- preset assembly
- output head logic
- reusable mechanisms
- provider capabilities
- plugin side effects

### 4.2 Bad componentization

Avoid:

- trainer-private branching for business routing
- provider behavior hidden as trainer helper code
- persistence mixed into core fit logic
- output semantics hidden in ad hoc post-processing

### 4.3 Mechanism binding levels

For reusable mechanisms, consider three binding levels:

- `optional`
- `bound`
- `defining`

This prevents two common mistakes:

- treating every reusable mechanism as "just a loose plugin"
- treating every family-specific mechanism as globally non-reusable

## 5) Contract Rules

`mlblack` should continue moving toward explicit contracts.

### 5.1 I/O Contract

Each formal object should make clear:

- what it consumes
- what it emits
- which inputs are required
- which inputs are optional

### 5.2 Composition Contract

Prefer stable declarations such as:

- `requires`
- `provides`
- `mutates`
- `cache`

Do not treat scattered `hasattr(...)`, random context probing, and informal conventions as the main contract surface.

### 5.3 Persistence Contract

Keep these outputs distinct:

- `artifact`
- `trainer_state`
- `report`
- provider/plugin side products

Rules:

- artifact is the training-family primary product
- trainer_state is for resume / warm_start / incremental continuity
- report is observability and audit payload
- provider/plugin outputs must not impersonate the artifact

### 5.4 Compatibility Contract

When adding or changing family/preset logic, keep compatibility visible through:

- metadata
- signatures
- resume drift checks
- scaffold equivalence tests

## 6) Field Alignment Rules

Field alignment is part of the architecture, not just a UI concern.

### 6.1 Stable schema keys

Schema keys should remain stable and English.

Examples:

- `family`
- `preset`
- `head`
- `runtime_backend`
- `status`
- `supports_resume`
- `supports_warm_start`

### 6.2 User-facing labels

User-facing labels may be Chinese or bilingual.
Internal schema keys should not be freely translated.

### 6.3 No uncontrolled synonyms

If the same meaning appears in multiple surfaces, prefer one canonical field key.

If aliases are necessary:

- register them centrally
- document them
- do not let every caller invent its own synonym

## 7) Catalog Is A Formal Framework Surface

Catalog is not optional garnish.

It should represent formal framework objects in a structured way.

Current structured catalog kinds:

- `family`
- `preset`
- `head`
- `component`
- `provider`
- `plugin`

### 7.1 Minimum catalog requirements

If an object becomes part of the formal framework surface, add:

- catalog entry
- stable summary
- structured `fields`
- structured `relations`

### 7.2 Catalog is for more than docs

Catalog exists for:

- CLI discoverability
- UI filtering
- field-based lookup
- relation jump
- future service/API consumption

Formal objects must not live only in source code and README prose.

## 8) Catalog Must Be Materialized To Database

This is now an explicit development rule.

Catalog should not remain only an in-memory registry view.
It should support a materialized database surface, currently sqlite.

### 8.1 Current rule

- structured catalog is authored in the registry layer
- the catalog must be materializable into sqlite
- UI and service layers should be able to consume that sqlite surface

### 8.2 Current sqlite surface

Current first-step sqlite catalog includes:

- entry rows
- per-entry `fields` JSON
- per-entry `relations` JSON
- scalar index rows for future field/search acceleration
- per-profile summary/schema snapshot

### 8.3 Current entrypoints

```powershell
python -m mlblack catalog db materialize --db-path runs\catalog.sqlite3 --profile framework-core
python -m mlblack catalog db summary --db-path runs\catalog.sqlite3 --profile framework-core
python -m mlblack catalog db show preset:mlp_torch --db-path runs\catalog.sqlite3 --profile framework-core
```

### 8.4 Future direction

The intended next step is to let UI and service queries read directly from the catalog database,
instead of always rebuilding the registry in-process.

## 9) Database Rules: Runtime State vs Catalog

Do not mix these two concepts.

### 9.1 Runtime state storage

Use:

- `ContextStore` / `SQLiteContextStore`
- `SnapshotStore` / `SQLiteSnapshotStore`
- `ExperimentTrackerCapability`

This layer is for:

- runtime refs
- heavy payload snapshots
- experiment events and metrics

### 9.2 Catalog storage

Use the catalog sqlite surface for:

- structured discoverability
- UI lookup
- field alignment queries
- relation navigation

Runtime state database and catalog database are related in spirit, but they are not the same plane.

## 10) When Adding A New Family, Preset, Or Mechanism

### 10.1 New `family`

At minimum, add:

- formal family contract
- grouped `family_spec`
- preset assembly path
- artifact family metadata
- family signature
- trainer_state compatibility checks
- direct-vs-scaffold equivalence tests
- catalog coverage

### 10.2 New `preset`

Prefer attaching it to an existing family.
Differentiate it through:

- backend
- mechanism
- head
- policy

### 10.3 New mechanism

First decide whether it is:

- `component`
- `provider`
- `plugin`

Then decide its binding level in that family:

- `optional`
- `bound`
- `defining`

## 11) Testing Expectations

### 11.1 Training semantics

Validate:

- direct trainer vs scaffold assembly
- metrics
- predictions
- artifact metadata
- trainer_state signature

### 11.2 Continuation semantics

If supported, validate:

- `resume`
- `warm_start`

Prefer direct-vs-scaffold equivalence here too.

### 11.3 Catalog changes

Validate:

- schema
- fields
- relations
- snapshot payload
- CLI
- sqlite materialization

### 11.4 Storage changes

Validate:

- memory backend
- sqlite backend
- snapshot ref replay
- experiment tracker persistence

## 12) Hard AI Execution Checklist

This checklist is intentionally redundant and operational.
It should be used as a literal self-check, not as loose advice.

### 12.1 Before Editing

- [ ] Did I classify the change target as `family`, `component`, `head`, `provider`, or `plugin`?
- [ ] Did I confirm that I am not creating a new family when a preset/head/component change would be enough?
- [ ] Did I identify which architecture plane this belongs to?
- [ ] Did I decide whether the change affects runtime state storage, catalog storage, or both?

### 12.2 During Design

- [ ] Did I choose explicit modules/interfaces instead of burying logic inside trainer conditionals?
- [ ] Did I keep persistence concerns outside the core training semantics where possible?
- [ ] Did I preserve the difference between `artifact`, `trainer_state`, `report`, and provider/plugin byproducts?
- [ ] Did I keep schema field keys stable and aligned?
- [ ] If aliases are needed, did I add them centrally instead of inventing local synonyms?

### 12.3 Before Finishing The Change

- [ ] Did I preserve architecture boundaries?
- [ ] Did I keep contracts explicit (`requires/provides/mutates/cache` or equivalent stable surface)?
- [ ] If a formal object changed, did I update catalog entry, fields, relations, and summary text?
- [ ] If catalog changed, did I preserve or improve sqlite materialization?
- [ ] If continuation semantics changed, did I check `resume` / `warm_start` / `incremental` implications?
- [ ] If UI labels changed, did I avoid breaking internal schema keys?

### 12.4 Verification Checklist

- [ ] Did I add or update the right tests?
- [ ] Did I run the most relevant tests?
- [ ] If direct-vs-scaffold equivalence matters, did I validate it?
- [ ] If sqlite materialization matters, did I validate `catalog db materialize/summary/show`?
- [ ] If I could not verify an important risk, did I state that explicitly?

## 13) Hard PR Checklist

For any substantial change that would be reviewed like a PR, the following should be true:

- [ ] The change explains why the object belongs to its chosen classification.
- [ ] The change explains why the chosen module/plane is the correct landing place.
- [ ] New or changed formal objects have stable fields and stable relations.
- [ ] Catalog changes remain usable through structured CLI queries.
- [ ] Catalog changes remain materializable into sqlite.
- [ ] Database changes preserve the distinction between runtime-state storage and catalog storage.
- [ ] Training-semantic changes are backed by equivalence or contract tests.
- [ ] Any unverified risk or compatibility gap is made explicit.

## 14) Reading Order

1. `AGENTS.md`
2. this file
3. `docs/ARCHITECTURE_PURPOSE.md`
4. `docs/GETTING_STARTED.md`
5. `docs/mlblack_framework_logic.md`
6. relevant family code and tests
