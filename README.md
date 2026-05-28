# mlblack

`mlblack` is an optimization-first machine learning layer aligned with `nsgablack`.
It is not a second orchestration framework. `nsgablack` owns solver groups,
stages, event routing, parallel/runtime backends, and L0 resource leases.
`mlblack` owns ML-specific components: representations, codecs, heads, problems,
inner fitting, artifacts, reports, and the symbolic engine.

```text
UnknownState
  -> ModelRepresentation / Codec / Head
  -> decoded model, function, estimator spec, or symbolic expression
  -> LearningProblem.evaluate(...)
  -> Feedback(objectives, constraints, gradients, residuals)
  -> OptimizerAdapter.update(...)
```

## Core Boundaries

| Layer | Responsibility | Not responsible for |
| --- | --- | --- |
| `Trainer` | single inner-training control plane | cross-trainer orchestration |
| `OptimizerAdapter` | propose/update strategy | reading business data directly |
| `Representation + Codec + Head` | decode unknown state into model/output semantics | computing objectives |
| `LearningProblem` | consume data and return feedback | scheduling resources |
| `Capability` | checkpoint/tracking/audit/report side effects | changing optimization semantics |
| `ResourceContext` | passive resource context injected from outside | resource allocation or leases |

## Quick Start: Single Inner Trainer

```python
import numpy as np

from mlblack.assembly import build_trainer
from mlblack.pipeline.data_views import train_valid_split

X = np.linspace(-1, 1, 64).reshape(-1, 1)
y = 1.5 + 2.0 * X[:, 0]
data = train_valid_split(X, y, feature_names=("x0",))

trainer = build_trainer(
    {
        "preset": "orthogonal_linear_point",
        "run_name": "linear_demo",
        "params": {"learning_rate": 0.05},
    },
    data=data,
)
result = trainer.fit(max_steps=50)
print(result.report["best_score"])
```

## Complex Orchestration

Use `nsgablack` for complex orchestration. `mlblack` exposes problem/proxy surfaces
that `nsgablack` can call as inner evaluation tasks.

Current formal cross-framework examples:

```powershell
python examples\cross_framework\nsgablack_outer_mlblack_inner\run_case.py
python examples\cases\symbolic_orthogonal_nested\run_solver.py --check
python examples\cases\symbolic_orthogonal_nested\run_solver.py --stage1-generations 1 --stage2-generations 1 --stage1-pop-size 4 --stage2-pop-size 4 --stage1-inner-steps 2 --stage2-inner-steps 2
```

## Symbolic Nested Learning

Symbolic learning is represented as nested optimization, not as an isolated
symbolic trainer family.

```text
Stage 1: nsgablack outer symbolic basis search
  -> mlblack inner parameter fitting
  -> orthogonality, stability, complexity, rank metrics

Stage 2: nsgablack outer basis-conditioned task expression search
  -> mlblack inner parameter fitting
  -> RMSE / interval / probability / classification metrics
```

The symbolic integration surface lives under `mlblack.integrations.nsgablack_symbolic`.
The core package remains nsgablack-free.

## Implemented Surface

- linear / orthogonal linear
- point / interval / probability / piecewise heads
- supervised regression, interval regression, classification metrics
- tree / xgboost estimator specs and estimator search
- numpy MLP and torch backprop adapter
- numericizer, feature-space, conditional primitives/composer
- nsgablack-style context contracts and doctor validation
- passive `ResourceContext` only; no mlblack-owned L0 allocator
- catalog, experiment query, dashboard export, artifact HTML viewer
- symbolic expression model, codec, multi-symbol head, symbolic gradients
- function pool, grammar, dynamic pool, graph cache, path memory
- Stage 1 orthogonal symbolic basis outer problem
- Stage 2 basis-conditioned symbolic task outer problem
- symbolic artifact schema with canonical expression, family recovery, phase-equivalence recovery

## Checks

```powershell
python -m compileall -q mlblack tests examples\cases\symbolic_orthogonal_nested examples\cross_framework
python -m pytest -q tests\test_symbolic_nsgablack_integration.py
python -c "from mlblack.project import run_project_doctor, format_doctor_report; print(format_doctor_report(run_project_doctor('.', strict=True)))"
python examples\orthogonal_point_demo.py
```
