# Symbolic Orthogonal Nested Benchmark

This benchmark is a thin runner over the formal `examples/cases/symbolic_orthogonal_nested` scaffold.
It does not define a second workflow. Each benchmark row still uses `nsgablack` outer solvers and `mlblack` inner fitting.

Check only:

```powershell
python examples\cases\benchmarks\run_project.py --check --build-check
```

Small run:

```powershell
python examples\cases\benchmarks\run_project.py -- --variants point,interval --generations 1 --pop-size 4 --inner-steps 3
```

## Neural Graph Benchmark Matrix

This benchmark checks the unified `NeuralGraphSpec -> NeuralGraphCodec -> Problem -> Adapter`
route across representative neural families.

It covers:

- tiny CNN image classification
- tiny GNN graph classification
- tiny CNN contrastive/retrieval
- tiny Transformer language modeling

Run:

```powershell
python examples\cases\benchmarks\run_project.py -- --steps 2 --repeats 3
```

The output records per-repeat status, mean/std/min/max wall time, best score
summary, adapter, problem and representation route. This keeps it as a small
stable benchmark suite rather than a single smoke call.
