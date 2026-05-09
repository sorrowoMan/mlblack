# known_relation_symbolic

Standard scaffold for known-relation symbolic benchmark components.

## Responsibility Split

- `problem/`: scenario instances and registry (`known_relation`-specific only).
- `pipeline/`: thin project wrapper over framework benchmark bundle assembly.
- `config/`: scaffold build configuration.
- `orchestration/`: hint wrapper plus standard runner assembly helpers (`build_known_relation_semantic_flow_spec`).
- `evaluation/`: thin project wrapper over framework truth-contract helpers.
- `mlblack_side/`: known-relation project instance of framework symbolic evaluation proxy.
- `nsgablack_side/`: project-facing aliases of framework outer-proxy interfaces.
- `build_solver.py`: standard assembly surface.
- `run_solver.py`: thin scaffold entrypoint.

Framework-level reusable mechanisms now live in:

- `core/symbolic/benchmark/contracts.py`
- `core/symbolic/benchmark/bundle_pipeline.py`
- `core/symbolic/benchmark/hints.py`
- `core/symbolic/benchmark/outer_proxy.py`

The current benchmark runners may still call compatibility wrappers under `examples/`,
but new logic should land here instead of expanding example files.
