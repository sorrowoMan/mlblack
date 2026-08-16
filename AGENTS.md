# AGENTS.md

## 0. How To Use This File

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
- A `mlblack` Case can be outer or inner. It must not create a private project runner, scheduler, or resource allocator.

## 1. Project Positioning

`mlblack` turns ML tasks into explicit semantic components:

```text
DataView / Spec
  -> Representation / Codec / Head
  -> LearningProblem
  -> Trainer / Provider
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
| Trainer / Provider | fit/evaluate one ML task | multi-Case fanout |
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
- hiding backend selection inside Trainer internals

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
- lifecycle side effect -> plugin
- reproducible output -> artifact
- backend execution capability -> provider/backend surface
- cross-Case order or resource scheduling -> shared Project substrate
- search over structures/configurations -> `nsgablack` semantic Case

## 7. Cross-Framework Rules

- `mlblack.core`, `mlblack.pipeline`, `mlblack.problems`, and `mlblack.representations` should remain independent of `nsgablack`.
- Explicit `nsgablack` integration belongs under `integrations/nsgablack_*`.
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

- [ ] Boundaries are clear: Trainer / Adapter / Representation / Problem / Plugin / Artifact.
- [ ] No adapter reads training data directly.
- [ ] No large object is written to context.
- [ ] `build_solver.py` remains canonical for Cases.
- [ ] `build_trainer.py` is only an alias when present.
- [ ] Resource use comes from injected `ResourceContext`.
- [ ] No private orchestration or L0 stack is introduced.
- [ ] Formal examples use Project / Case / Scaffold.
- [ ] Catalog, doctor, artifact, or report surfaces are updated where relevant.
