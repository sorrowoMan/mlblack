# AGENTS.md

## 0) How To Use This File

This is the first-read execution contract for AI collaborators working in `mlblack`.

If you are an AI / coding agent entering this repo:

1. Read this file first.
2. Then read `docs/AI_DEVELOPMENT_GUIDELINES.md`.
3. Then read the family-specific code and tests relevant to the task.

Permission-first rule:

- **Before making any code change, running tests, launching benchmarks, or executing write-producing automation, you must first obtain explicit user permission.**
- Before permission is granted, you may only do: repo reading, architecture analysis, mechanism diagnosis, proposal drafting, and documentation rewrites.
- Do not treat “the root cause is obvious” or “the patch is small” as implicit authorization.
- If the user is discussing mechanisms, comparing designs, or asking for diagnosis, default to **analysis-only / no code changes**.

This file is intentionally short and normative.
If a more scattered document conflicts with this file, prefer this file.

## 1) First Principle

`mlblack` is a composable learning framework, not an algorithm-name dump.

Do not start by asking:

- "Should I add another algorithm directory?"

Start by asking:

- Is this a new `family`?
- Is this a cross-family `component`?
- Is this a new `head`?
- Is this a reusable `provider`?
- Is this a pure `plugin`?

## 2) The Five Kinds

### 2.1 `family`

Defines how a function family is trained.
It is the main training backbone.

Current formal families:

- `linear family`
- `tree family`
- `tree_boosting family`
- `neural family`
- `symbolic family`

### 2.2 `component`

Enhances an existing family without replacing its main training loop.

Typical examples:

- regularization
- dropout
- batch sampling policy
- warm-start policy
- router / gate / piecewise primitive
- gradient norm state signal
- layer activation signal

### 2.3 `head`

Defines output semantics.

Current formal heads:

- `point`
- `interval`

### 2.4 `provider`

Provides training/evaluation-side external power:

- bridge
- proxy
- cache
- short-circuit evaluation
- numerical solver

### 2.5 `plugin`

Owns side effects and observability:

- report
- checkpoint
- trace
- reproducibility
- resource audit

## 3) Non-Negotiable Architecture Boundaries

- `family` owns main training semantics.
- `component` can enhance, but must not take over the full training workflow.
- `head` changes output semantics, not the family identity.
- `provider` powers a path, but must not pretend to be the model artifact.
- `plugin` owns side effects, but must not rewrite task semantics.

Do not:

- mix `report/checkpoint/cache/trace` into the trainer body
- hard-code provider logic into trainer `if/else`
- mistake output-semantic changes for a new family
- create a new trainer family just for one mechanism
- fall back to algorithm-name scattering when a `family/preset/head` description is clearer

### 3.1 Symbolic Search Boundary

Architecture law:

- If the structure is fixed, use an `mlblack` trainer.
- If the structure must be searched, it must enter an `nsgablack` outer solver; `mlblack` should provide only the evaluation proxy, inner fitter, artifact builder, and audit surface.

For symbolic learning, structure means any searched expression topology, term set, basis set, source object, chart variant, realization head, branch/gate, threshold, or symbolic operator composition.

Consequences:

- Do not let a symbolic trainer grow into a second outer optimization framework.
- Do not hide symbolic structure search as private trainer beam/scoring logic when it should be represented as an `nsgablack` problem.
- `nsgablack` owns outer orchestration, population/frontier management, multi-objective search, solver/adaptor choice, and search trace.
- `mlblack` owns candidate evaluation: fitting, metrics, objective payloads, constraints, symbolic artifacts, interval heads, and audit reports.

## 4) Componentization Rules

When in doubt, prefer explicit modules over hidden trainer conditionals.

Good decomposition targets:

- family backbone
- preset assembly
- head semantics
- mechanism component
- provider capability
- plugin side effect

Bad decomposition:

- business routing hidden inside trainer internals
- provider behavior hidden as a trainer private helper
- persistence logic hidden inside fit loops

Execution guard:

- **No implementation, test run, or benchmark run without explicit user approval first.**

## 5) Contract Rules

Every formal object should expose stable contracts.

### 5.1 I/O Contract

At minimum, define:

- what it consumes
- what it produces
- what is optional
- what is required

### 5.2 Composition Contract

Prefer explicit declarations such as:

- `requires`
- `provides`
- `mutates`
- `cache`

Do not rely on ad hoc `hasattr(...)` and scattered context guessing as the main composition protocol.

### 5.3 Persistence Contract

Keep these outputs distinct:

- `artifact`
- `trainer_state`
- `report`
- provider/plugin byproducts

Do not let provider/plugin outputs masquerade as the main artifact.

## 6) Field Alignment Rules

Field alignment is a first-class rule in `mlblack`.

If two objects mean the same thing, they should use the same stable field key.

Examples:

- `family`
- `preset`
- `head`
- `runtime_backend`
- `status`
- `supports_resume`

Rules:

- schema keys stay stable and English
- user-facing labels may be Chinese
- do not create near-duplicate field names for the same meaning
- if aliases are needed, register them explicitly; do not scatter synonyms informally

## 7) Catalog Is Mandatory

Formal framework objects must not live only in code and README prose.

If something is a formal part of the framework surface, it should be represented in catalog.

Current structured catalog kinds:

- `family`
- `preset`
- `head`
- `component`
- `provider`
- `plugin`

For new formal objects, add:

- catalog entry
- structured `fields`
- structured `relations`
- stable summary text

## 8) Catalog Must Also Be Materialized To Database

Catalog is not just a transient in-memory helper.
It is a formal discoverability surface and should support database-backed persistence.

Current rule:

- catalog entries are authored in the structured registry surface
- catalog must be materializable into sqlite
- UI/service layers should be able to consume the materialized catalog database

Current sqlite entrypoints:

```powershell
python -m mlblack catalog db materialize --db-path runs\catalog.sqlite3 --profile framework-core
python -m mlblack catalog db summary --db-path runs\catalog.sqlite3 --profile framework-core
python -m mlblack catalog db show preset:mlp_torch --db-path runs\catalog.sqlite3 --profile framework-core
```

This is part of the formal framework surface, not an optional documentation trick.

## 9) Runtime State And Database Rules

`mlblack` distinguishes runtime state storage from catalog storage.

### 9.1 Runtime State

Use:

- `ContextStore` / `SQLiteContextStore` for light keys and refs
- `SnapshotStore` / `SQLiteSnapshotStore` for heavy payloads

Rules:

- small state goes to context
- heavy payload goes to snapshot
- context should prefer `*_ref`
- do not keep large payloads in context long-term

### 9.2 Catalog Database

Catalog database is a separate indexed surface.

It is for:

- discoverability
- UI queries
- field-aligned lookup
- relation jumps
- future service/API reads

Do not confuse runtime snapshot storage with catalog indexing.

## 10) When Adding A New Family Or Preset

### 10.1 New `family`

At minimum, add:

- formal family contract
- grouped `family_spec`
- preset assembly path
- artifact family metadata
- family signature
- trainer_state compatibility checks
- direct-vs-scaffold equivalence tests
- catalog fields and relations

### 10.2 New `preset`

Prefer reusing an existing family backbone.
Differentiate via:

- backend
- mechanism
- head
- policy

Do not invent a private one-off workflow too early.

## 11) Example Assembly Rule

If you add or change any `example`, `demo`, or benchmark runner, it must be assembled through the standard project scaffold/protocol path.

Here "standard scaffold" means the formal shape and responsibility split, not a mandatory literal directory name. It should look like the repo's `my_project` style even when it lives elsewhere:

- `problem/`: dataset, scenario, objective, target, and contract definitions
- `pipeline/`: feature flow, family/head assembly, evaluation chain, and artifact construction
- `config/`: declarative, reproducible assembly configuration
- `build_*` / `run_*`: thin official assembly and execution entrypoints
- `providers/` / `plugins/` / `reporting/`: external power, side effects, audit reports, and persistence
- `registry` / `catalog`: discoverability for formal framework objects

Example files themselves must remain thin entrypoints, compatibility wrappers, or teaching calls. Real assembly logic must live in the standard scaffold layers above.

Use the formal assembly surface, such as:

- `FlowAssemblySpec`
- `TrainerAssemblySpec`
- standard CLI / workflow entrypoints
- standard project scaffolds already used by the repo
- `nsgablack` outer-solver scaffold surface when symbolic structure search is involved

Do not:

- instantiate a pile of private components ad hoc inside the example when a formal scaffold path exists
- bypass the registry/assembly contract just to make the example shorter
- let examples drift into a second unofficial runtime architecture
- implement a private symbolic outer-search runner in an example when the search should be an `nsgablack` outer-solver problem
- leave all problem, pipeline, provider, head, plugin, reporting, and runtime wiring permanently inside one `examples/.../*.py` file

Cross-framework rule:

- If an example uses `mlblack`, the `mlblack` side must expose the evaluation proxy, inner fitter, artifact builder, head, and audit/report surface through the mlblack standard scaffold shape.
- If an example also uses `nsgablack`, the `nsgablack` side must expose the outer solver, adapter, representation, bias, plugin, and runtime surface through the nsgablack standard scaffold shape.
- Cross-framework examples may compose those two official scaffold surfaces, but must not bypass either side with private glue.

Examples are part of the framework teaching surface, so they must reflect the official product assembly path.

## 12) Testing Expectations

For training-semantics changes, validate:

- direct trainer vs scaffold assembly
- metrics
- predictions
- artifact metadata
- trainer_state signature

If `resume` / `warm_start` are supported, validate direct-vs-scaffold equivalence there too.

If catalog is changed, validate:

- schema
- snapshot
- relations
- CLI
- materialized sqlite catalog path

## 13) Hard AI Checklist

This checklist is meant to be used literally.
Before an AI collaborator finishes a change, it should be able to answer every item.

### 13.1 Before Editing

- [ ] Did I classify the object as `family` / `component` / `head` / `provider` / `plugin` before changing code?
- [ ] Did I check whether an existing family/preset/component already covers this need?
- [ ] Did I identify which plane this change belongs to?
- [ ] Did I verify whether this is a runtime-state change, a catalog change, or both?

### 13.2 Before Merging Code

- [ ] Did I preserve family/component/head/provider/plugin boundaries?
- [ ] Did I keep contracts explicit instead of hiding behavior in trainer internals?
- [ ] Did I reuse stable field keys instead of inventing new near-duplicates?
- [ ] If I changed a formal framework object, did I update catalog entry/fields/relations?
- [ ] If I changed catalog, did I preserve or improve sqlite materialization?
- [ ] If I added or changed an example/demo, did I keep it on the standard scaffold/assembly path?
- [ ] If I changed continuation semantics, did I verify `resume` / `warm_start` implications?
- [ ] If I changed persistence, did I keep `artifact`, `trainer_state`, `report`, and side products distinct?

### 13.3 Before Shipping A Result

- [ ] Did I add or update the right tests?
- [ ] Did I run the most relevant tests instead of assuming compatibility?
- [ ] If direct-vs-scaffold equivalence matters, did I validate it?
- [ ] If I could not validate something important, did I explicitly say so?

## 14) Hard PR Checklist

If a change is large enough to be treated like a PR, it should pass this checklist:

- [ ] The change explains its classification decision (`family/component/head/provider/plugin`).
- [ ] The change explains why the chosen module/plane is the correct landing place.
- [ ] Any new formal surface has stable fields, stable summary text, and catalog relations.
- [ ] Any catalog change remains queryable through CLI and materializable through sqlite.
- [ ] Any user-facing label changes do not break stable internal schema keys.
- [ ] Any persistence change preserves the runtime-state-vs-catalog-db distinction.
- [ ] Any new example or demo follows the standard scaffold/assembly path instead of a private shortcut.
- [ ] Any training-semantic change is covered by the right equivalence or contract tests.
- [ ] Any unverified risk is called out explicitly instead of being hidden.

## 15) Recommended Reading Order

1. `AGENTS.md`
2. `docs/AI_DEVELOPMENT_GUIDELINES.md`
3. `docs/ARCHITECTURE_PURPOSE.md`
4. `docs/GETTING_STARTED.md`
5. `docs/mlblack_framework_logic.md`
6. relevant family code and tests
