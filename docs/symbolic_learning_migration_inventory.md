# Symbolic Learning Migration Inventory

Symbolic learning is now treated as standard Case composition under the shared Project / Case / Scaffold / L0 substrate.

## Current Decision

Symbolic learning should not be migrated as a private symbolic training stack. It should be decomposed into:

- mlblack symbolic semantics: expression DSL, codec, heads, parameter slots, metrics, artifact schema
- nsgablack search semantics: outer candidate generation, Pareto tradeoffs, structure search, search trace
- shared substrate: Project stages, nested Case calls, resource grants, run audit

## Standard Nested Shape

```text
Project
  stage 1:
    outer symbolic basis search Case
      -> inner mlblack fitting/evaluation Case
      -> basis artifact

  stage 2:
    outer task-expression search Case
      -> inner mlblack fitting/evaluation Case
      -> task artifact
```

Each runnable unit is a Case with `build_solver.py`, `run_solver.py`, local config, runtime request, artifact output, and audit payload.

## Mechanism Groups To Preserve

| Mechanism | mlblack Semantic Surface | Search/Substrate Surface |
| --- | --- | --- |
| symbolic expression DSL | model / codec | candidate payload |
| parameter slots | representation / problem | inner fit budget |
| primitive registry | pipeline symbolic component | outer search-space provider |
| function pool | symbolic pipeline | candidate pool payload |
| dynamic expansion | pool transform / signal component | Project stage or outer search policy |
| gradient guidance | metric / signal payload | bias or candidate scoring |
| orthogonal basis metrics | problem/evaluation metric | multi-objective search target |
| basis consensus | artifact/report | Project run surface |
| interval or probability head | Head semantics | objective projection |
| replay and lineage | artifact/report | run audit |

## Target Layout

```text
mlblack/
  models/symbolic.py
  models/symbolic_gradient.py
  pipeline/symbolic/
  representations/codecs/symbolic.py
  representations/heads/symbolic.py
  problems/symbolic.py
  integrations/nsgablack_symbolic/
    specs.py
    search_space.py
    dynamic_pool.py
    gradient_guidance.py
    structure_guard.py
    search_policy.py
    basis_consensus.py
    evaluation.py
    expression_audit.py
    replay.py
    orthogonal_problem.py
    task_symbolic_problem.py
    builders.py
    artifacts.py
```

`mlblack.core`, `mlblack.pipeline`, `mlblack.problems`, and `mlblack.representations` should remain independent of `nsgablack`. Explicit cross-framework imports belong under `integrations/nsgablack_symbolic/`.

## Migration Order

1. Move fixed symbolic expression and codec surfaces into `mlblack`.
2. Add symbolic heads and typed artifacts.
3. Add symbolic function-space and primitive pipeline components.
4. Add integration payloads for outer candidate search.
5. Build Stage 1 orthogonal basis Case.
6. Build Stage 2 basis-conditioned task Case.
7. Add Project-level examples that compose both stages through shared L0.

## Do Not Do

- Do not hide structure search inside a private trainer loop.
- Do not put Project stage order inside a Trainer.
- Do not hard-code device or thread settings inside symbolic components.
- Do not pass large expression tables through runtime context.
- Do not make `mlblack` require `nsgablack` for standalone ML Cases.

## Open Design Questions

1. What is the minimal symbolic expression spec for the codec?
2. How does the multi-symbol head split encoded state into expression blocks?
3. Which parameter slots are fitted by the inner Case?
4. Which objectives are returned to the outer Case?
5. How is the basis artifact passed into Stage 2?
6. Which primitive families are required for the first formal example?
7. Which runtime report fields are required for nested symbolic debugging?
