# mlblack

`mlblack` is an optimization-first machine learning layer aligned with `nsgablack`.
Both frameworks share one Project / Case / Scaffold / L0 substrate. Orchestration
belongs to that substrate, not to either semantic layer privately. `nsgablack`
is responsible for optimization/search semantics; `mlblack` is responsible for
ML semantics: representations, codecs, heads, problems, inner fitting, artifacts,
reports, and the symbolic engine.

Shared substrate baseline: `blackbase>=0.3.3,<0.4.0`; optimization methods are
resolved from `nsgablack>=0.3.4,<0.4.0`, while ML semantics and compute Providers
remain in mlblack.
MLBlack imports Case orchestration, L0 grants, call binding, Context/Snapshot,
Catalog primitives and runtime projection envelopes directly from BlackBase;
the former `mlblack.core.resources`, Context and store forwarding modules no
longer exist.

```text
UnknownState
  -> ModelRepresentation / Codec / Head
  -> decoded model, function, estimator spec, or symbolic expression
  -> LearningProblem.evaluate(...)
  -> Feedback(objectives, constraints, gradients, residuals)
  -> nsgablack AlgorithmAdapter.update(...)
```

## Core Boundaries

| Layer | Responsibility | Not responsible for |
| --- | --- | --- |
| `LearningSolver` / Trainer facade | ML-friendly `fit` / artifact / report projection | owning another optimization lifecycle |
| `nsgablack.ComposableSolver` | the single Adapter lifecycle, incumbent, budget, cancellation and snapshot control plane | ML data/model/backend semantics |
| `nsgablack AlgorithmAdapter` | propose/update strategy | reading business data directly |
| `Representation + Codec + Head` | decode unknown state into model/output semantics | computing objectives |
| `LearningProblem` | consume data and return feedback | scheduling resources |
| `Capability` | checkpoint/tracking/audit/report side effects | changing optimization semantics |
| `ResourceContext` | project-level L0 grant consumed by a case | resource allocation or global leases |

## Quick Start: One ML View, One Optimization Control Plane

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

Gradient and neural presets now return an ML-friendly `LearningSolver`. Its
`fit()` method returns `TrainerResult`, but propose/evaluate/update, incumbent,
lifecycle, cancellation, budget and snapshot execution happens once inside
`nsgablack.ComposableSolver`. Closed-form and third-party `estimator.fit()`
routes are ML Problem/Provider implementations inside that same lifecycle;
MLBlack contains no private Trainer loop or optimization Adapter hierarchy.

## Complex Orchestration

Use the shared Project/Case substrate for complex orchestration. A `mlblack` case
can be outer or inner; it should not recreate a private orchestration or L0 stack.

Current formal cross-framework examples:

```powershell
python examples\cases\cross_framework\run_project.py --check --build-check
python examples\cases\symbolic_orthogonal_nested\run_project.py --check --build-check
python examples\cases\symbolic_orthogonal_nested\run_project.py -- --stage1-generations 1 --stage2-generations 1 --stage1-pop-size 4 --stage2-pop-size 4 --stage1-inner-steps 2 --stage2-inner-steps 2
```

## Symbolic Nested Learning

Symbolic learning is represented as nested optimization, not as an isolated
symbolic training stack with private orchestration.

```text
Stage 1: outer symbolic basis search Case
  -> mlblack inner parameter fitting
  -> orthogonality, stability, complexity, rank metrics

Stage 2: outer basis-conditioned task expression search Case
  -> mlblack inner parameter fitting
  -> RMSE / interval / probability / classification metrics
```

When these stages need optimization/search semantics, the outer Case is usually
implemented with `nsgablack`. The symbolic integration surface lives under
`mlblack.integrations.nsgablack_symbolic`; the core package remains nsgablack-free.

## Implemented Surface

- linear / orthogonal linear
- point / interval / probability / piecewise heads
- supervised regression, interval regression, classification metrics
- tree / xgboost estimator specs and estimator search
- 统一梯度路径：MLP、Transformer、CNN、GNN、时序 NeuralGraph 与 TabNet 共用 nsgablack SGD/Adam/AdamW Adapter + 唯一 ComposableSolver 控制面 + mlblack Torch Evaluation Provider
- 标准 MLP 路径：`NeuralGraphSpec.mlp` 唯一定义结构，`NeuralGraphCodec` 唯一定义参数布局，各 backend 只负责 lowering。
- 解析梯度路径：线性/符号 Problem 直接产出梯度，但仍通过同一个 nsgablack `gradient.*` Adapter；无需伪造 Backend Provider
- 统一黑盒路径：区间、分类、时序和 estimator spec 预设共用 nsgablack `search.random_gaussian`，模型解码与拟合仍归 mlblack
- BlackBase `StateRef` 版本栅栏更新与显式 materialization；设备参数/梯度/optimizer slot 不冒充 Artifact 或 checkpoint 状态
- numericizer, feature-space, conditional primitives/composer
- nsgablack-style context contracts and doctor validation
- Project L0 `ResourceContext` consumption and audit; no private L0 allocator
- catalog, experiment query, dashboard export, artifact HTML viewer
- symbolic expression model, codec, multi-symbol head, symbolic gradients
- function pool, grammar, dynamic pool, graph cache, path memory
- Stage 1 orthogonal symbolic basis outer problem
- Stage 2 basis-conditioned symbolic task outer problem
- complete Trainer-to-Solver Case invocation returning the full child Case envelope
- explicit best/Pareto projection with solve-status validation; no implicit default projection
- optional objective-to-Feedback mapping, separated from solution selection
- per-invocation timeout or `ExecutionControl`, bounded by the parent control lineage
- symbolic artifact schema with canonical expression, family recovery, phase-equivalence recovery

## Checks

```powershell
python -m compileall -q project core catalog examples
python -m pytest -q tests\test_symbolic_nsgablack_integration.py
python -c "from mlblack.project import run_project_doctor, format_doctor_report; print(format_doctor_report(run_project_doctor('.', strict=True)))"
python examples\cases\orthogonal_point_demo\run_project.py --check --build-check
```
