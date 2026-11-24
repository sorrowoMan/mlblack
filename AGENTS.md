# AGENTS.md

## 0. How To Use This File

Last verified against code: 2026-10-22 (ML pool startup and portable restore).

Runtime ownership and extension rules:

- ML context vocabulary lives in `mlblack.context_keys`; BlackBase owns registration and common keys. Project/Case extensions use scoped `context_keys.json` declarations shared by Doctor and runtime. Do not add ML-only vocabulary to BlackBase.
- Generated Cases use `mlblack.project.starter` for a runnable NumPy linear-regression baseline; keep data/pipeline and canonical assembly entries editable inside the Case. Scaffold creation is distinct from creating an unimplemented custom component.

- CandidateBatch preserves value and identity independently; Unit carries algorithm structure and an explicit codec, while Adapter owns search decisions.
- A model Codec interprets candidate parameters and model layout; an Adapter-owned Unit carries search relationships and history. They may share immutable layout data, never writable optimizer state or a live backend object graph. See `docs/architecture/ML_POOL_UNIT_INTEGRATION_CN.md`.
- Construct LearningSolver and its shared SolutionPool before generating candidates. Read declared `dimension` or resolve layout in Representation.setup; never sample init just to infer shape. Actual init goes through Solver.init_candidate. Adapters that propose directly or use geometry during setup (for example BO initial points) require the model dimension before Adapter.setup.
- LearningSolver state v7 includes the portable SolutionPool value closure. Restore it before Adapter state resolves exact SolutionRefs; never infer candidate identity from equal arrays or silently replace a missing legacy pool with an empty one.
- ML result projection uses token-matched model handles or LearningProblem.restore_model_for_evaluation over saved feedback. Recovery must not call prepare_model_for_evaluation, estimator.fit or an inner Case to recreate an already evaluated result. A Problem with preparation side effects must provide an explicit read-only restore rule.
- The formal checkpoint includes the required `ml_projection` component for best feedback, token/evaluation binding and Artifact references; shared Solver state keeps history. Ordinary build_context contains only backend descriptors. Inject the live session with enrich_ml_context only for local ML calls, never into Router/task checkpoint contexts.
- RunSession owns run identity/progress/restore/results; use `run_session.submit_restore` instead of writing private resume flags.
- PopulationAuthority owns the shared in-memory candidate projection; persistent authority still requires AuthorityStore validation.
- MLRepresentationBridge declares RepresentationCapabilities; initializer/decoder marker attributes are not capabilities.
- Joint repair is opt-in via LearningSolver(problem_validator=..., joint_repair_policy=...). The validator receives the final UnknownState and ML context; the existing ModelRepresentation remains the repair owner. Never infer feasibility or call model evaluation as an implicit validator. Unit admission, rejection policy and cancellation remain in the shared nsgablack control path.
- ProblemRepair-only and joint repair paths both restore rejected owners' local pre-repair state. Partial UnitRepair projection runs after Adapter disposition; its draft already uses admitted coordinates. The trajectory/poll/parameters/local_region Unit contracts are v2; never silently reinterpret v1 checkpoint feedback as current.
- Register ML step projections with SolverSemanticExtension and result publication with FinalizationHooks. Do not override step to repeat the shared transaction.
- Plugin effects use ParticipationContract; finalization veto and evaluation decision require explicit grants. Controller decisions stay within their owned domains.
- Bias adjustments retain Feedback.info.raw_evaluation and preference_deltas. A selection view is not the original measurement.

This is the collaboration contract for `mlblack`. It defines where new code, docs, and examples should land.

Language rule (added):

- Tutorial and guide docs should default to Chinese as the primary version, especially `docs/standard_scaffold_tutorial`.
- English content is allowed, but should be provided as a separate counterpart file (for example `*_EN.md`) instead of replacing Chinese primary pages.
- New tutorial chapters should be written in Chinese first, then optionally mirrored in English.

Current first principle:

- `nsgablack` and `mlblack` share one Project / Case / Scaffold / L0 substrate.
- Orchestration and resource grants belong to the shared substrate.
- `nsgablack` is the optimization/search semantic layer.
- `mlblack` is the machine-learning semantic layer.
- Adapter-driven ML uses `nsgablack.ComposableSolver` as the only optimization
  control plane. `LearningSolver.fit()` is an ML result facade, not another loop.
- CompletionPolicy consumes the Solver-owned, checkpointable `RunProgressState`;
  it must not keep a second private step/time counter or manufacture empty steps.
- Semantic model identity remains in the Solver-owned CandidateBatch population;
  numeric Adapters select rows by candidate token and may not reconstruct ML
  metadata from equal parameter vectors.
- `mlblack` has no private Trainer control class or optimization Adapter hierarchy.
  Closed-form and third-party `estimator.fit()` routes belong in ML Problem/Provider
  components and execute through `LearningSolver` + `nsgablack.ComposableSolver`.
- A `mlblack` Case can be outer or inner. It must not create a private project runner, scheduler, or resource allocator.
- LearningSolver restore requests are queued and applied after Solver/Adapter setup;
  ML components must not restore directly before setup or maintain private resume-order flags.
- A running LearningSolver may only apply state while the shared post-setup restore
  transaction lock is active; public runtime mutation must fail closed.
- Provider-backed Gradient Adapter owns exact optimizer-slot cleanup; LearningSolver
  trajectory release is an idempotent final safety net, not the primary slot owner.
- After Solver/Provider/backend teardown succeeds, transactional output first passes
  strict `on_solver_finalization_prepare`, then commits atomically. Final TrainerResult
  and save-on-finish checkpoint publication are observed through `on_solver_finalized`.

## 1. Project Positioning

`mlblack` turns ML tasks into explicit semantic components:

```text
DataView / Spec
  -> Representation / Codec / Head
  -> LearningProblem
  -> Provider / LearningSolver facade
  -> nsgablack ComposableSolver control
  -> Artifact / Report
```

It does not own cross-Case scheduling. When a task needs multiple Cases, fanout, nested optimization, or resource allocation, use the shared Project substrate. When the task needs search over structures, budgets, or tradeoffs, use `nsgablack` search semantics as a Case.

## 2. Boundary Map

| Layer | mlblack Responsibility | Not Responsible For |
| --- | --- | --- |
| DataView / Schema | stable ML data view | Project orchestration |
| Pipeline | feature and target transforms | global scheduling |
| Representation / Codec | encode/decode model state | objective aggregation |
| Head | point, interval, probability, symbolic output semantics | training order |
| LearningProblem | consume data and return feedback | resource allocation |
| LearningSolver facade / Provider | ML fit vocabulary and evaluation execution | owning a second optimization lifecycle |
| nsgablack ComposableSolver | one Adapter lifecycle/incumbent/budget/snapshot control | ML model/data/backend semantics |
| Plugin | checkpoint, tracking, audit, report | changing optimization semantics |
| Artifact | reproducible model/report payload | temporary runtime context |
| ResourceContext | consume and audit Project L0 grant | owning leases or global pools |

## 3. Standard Case Shape

Use the same Case template as `nsgablack`:

```text
case_name/
  __init__.py
  build_solver.py           # canonical assembly entry
  build_trainer.py          # alias only
  run_solver.py             # canonical CLI/debug entry
  run_trainer.py            # alias only
  config.py
  problem/
  pipeline/
  adapter/
  bias/
  plugins/
  evaluation/
  runtime/
  solver/
```

Rules:

- `build_solver.py` is canonical.
- `build_trainer.py` must stay a thin alias.
- `run_solver.py` is canonical.
- `run_trainer.py` must stay a thin alias.
- Case-level `capabilities/` is not a formal capability directory; use `plugins/`.
- Case-level `representation/` is not formal; model encoding belongs under `pipeline/` or framework semantic modules.
- `assembly/scaffold.json` is not a runtime source of truth.

## 4. L0 Resource Rule

`mlblack` consumes `ResourceContext`; it does not allocate global resources.

Allowed:

- read granted threads, device tokens, namespace, budget, backend hints
- clamp local compute backend settings to the grant
- report effective backend and fallback
- fail fast when a required capability is unavailable

Forbidden:

- creating a private resource allocator
- creating a private lease store
- hard-coding machine-local devices in examples or components
- silently expanding resources beyond the grant
- hiding backend selection inside Trainer/facade internals

## 5. Context / Snapshot / Artifact

Context is for lightweight fields and refs only:

- run name
- step
- small metrics
- resource refs
- artifact refs
- snapshot keys

Large objects go to Snapshot or Artifact:

- model object
- fitted estimator
- large arrays
- full history
- trace
- symbolic graph cache

## 6. Component Placement

Before adding a new capability, classify it:

- data preparation -> `pipeline/`
- model state encoding -> representation / codec
- output semantics -> head
- objective and metric semantics -> problem
- optimization step -> adapter
- optimization lifecycle / incumbent / budget / cancellation -> `nsgablack.ComposableSolver`
- lifecycle side effect -> plugin
- reproducible output -> artifact
- backend execution capability -> provider/backend surface
- cross-Case order or resource scheduling -> shared Project substrate
- search over structures/configurations -> `nsgablack` semantic Case

## 7. Cross-Framework Rules

- `mlblack.core`, `mlblack.pipeline`, `mlblack.problems`, and `mlblack.representations` should remain independent of `nsgablack`.
- Explicit `nsgablack` integration belongs under `integrations/nsgablack_*`.
- All training presets assemble through `LearningSolver`; a private Trainer loop
  is not a valid integration target.
- Cross-framework examples must compose standard Case surfaces.
- Data and model artifacts pass through Artifact/Snapshot refs.
- Nested calls pass structured request/result payloads and `ResourceContext`.

## 8. Inner Composition

Allowed inside one ML Case:

- ordered data pipeline
- model-conditioned target transform
- integrated prediction model
- sequential fitting that is part of one Trainer contract

When the execution becomes multiple independently runnable units, use multiple Cases under a Project.

## 9. Common Commands

```powershell
Set-Location "C:\Users\hp\Desktop\mlblack"

python -m compileall -q core project examples
python -c "from mlblack.project.doctor import run_project_doctor, format_doctor_report; print(format_doctor_report(run_project_doctor('.', strict=True)))"
```

## 10. Minimum Checklist

- [ ] Boundaries are clear: LearningSolver facade / ComposableSolver / Adapter / Representation / Problem / Plugin / Artifact.
- [ ] No adapter reads training data directly.
- [ ] No large object is written to context.
- [ ] `build_solver.py` remains canonical for Cases.
- [ ] `build_trainer.py` is only an alias when present.
- [ ] Resource use comes from injected `ResourceContext`.
- [ ] No private orchestration or L0 stack is introduced.
- [ ] Formal examples use Project / Case / Scaffold.
- [ ] Catalog, doctor, artifact, or report surfaces are updated where relevant.


## 11. Artifact local context and teardown boundary (2026-11-16)

- Problem.build_model_artifact is a local ML semantic call and must receive enrich_ml_context; never add backend.session to shared Router/task context.
- Case finalization prepares the derived model Artifact while backend services are alive, then serializes and publishes through the existing transaction only after successful teardown.
- Do not reopen a closed backend to regenerate parameter layout or model descriptors. The pending payload is process-local, reset per run, and excluded from checkpoint/context.
- Cross-worktree tests must explicitly select the target nsgablack configuration/import path; mlblack pytest's default pythonpath may select the neighboring desktop checkout.
