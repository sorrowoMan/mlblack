# Legacy Nowcasting Compatibility Layer

This project-local package keeps the original traffic symbolic nowcasting scripts runnable from the
standard `traffic_congestion` Project.

It intentionally lives under this Project instead of `mlblack.core`:

- the scripts still use historical imports such as `core.symbolic.*` and `examples.work_ci_reader`;
- the current mlblack architecture keeps Project/Case orchestration outside the ML semantic core;
- new work should move pieces from this layer into formal Case `problem/`, `pipeline/`,
  `adapter/`, `plugins/`, and artifact surfaces incrementally.

Formal Case entrypoints are:

- `cases/symbolic_mechanism_outer`
- `cases/symbolic_interval_outer`

