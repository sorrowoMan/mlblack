# nsgablack Symbolic Integration

This namespace is the only place where future symbolic structure search should
bind to `nsgablack`.

Boundary:

- `mlblack.models/representations/problems/pipeline` stay nsgablack-free.
- `pipeline.symbolic` materializes a `FunctionPool`.
- nsgablack outer solvers search symbolic structure over that pool.
- mlblack inner trainers fit parameters for each fixed decoded symbolic candidate.

Current status:

- JSON-compatible stage plans are defined for nsgablack-side orchestration.
- `OrthogonalBasisOuterProblem` provides Stage 1 basis search:
  outer candidate selects function-pool terms, inner mlblack run fits symbolic parameters, objectives return orthogonality/condition/complexity/rank signals.
- Stage 1 and Stage 2 problem surfaces exist. Serial/stage orchestration is owned by nsgablack examples/cases, not by mlblack.
