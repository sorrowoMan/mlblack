# Model Composition Boundary

Complex ML structures should be expressed as composable ML semantic components, then run through the shared Project / Case / Scaffold substrate.

This means:

- `mlblack` defines what a model, target transform, composition rule, artifact, and metric mean.
- the shared substrate defines stage order, parallelism, nested Case calls, resource grants, and run audit.
- `nsgablack` may provide the outer search semantics when the Project needs optimization over structures, budgets, or choices.

## Core Judgment

```text
complex model = fitted components + explicit composition semantics
multi-step training = Project runs multiple standard Cases or nested Case calls
residual learning = target transform plus additive model composition
multi-modal fusion = explicit per-component I/O contract plus integration model
```

Do not create a private mlblack orchestration stack to express those patterns. Add or reuse semantic components instead:

```text
IntegratedPredictionModel
PredictionIntegrationComponent
PredictionIOContract
ModelConditionedTargetComponent
ArtifactBuilder
```

## Why I/O Contracts Matter

Composition cannot assume every component consumes the same `X`.

Examples:

```text
residual:
  main_model.predict(X_numeric)
  residual_model.predict(X_numeric)

stacking:
  meta_model.predict([X_numeric, stage1_prediction])

multi-modal:
  text_model.predict(input_ids)
  image_model.predict(image_tensor)
  tabular_model.predict(tabular_features)
```

The integration layer must declare:

- which input key each component consumes
- expected kind, rank, and feature count
- output shape and row alignment
- final integration rule

## Current I/O Contract

Location:

```text
mlblack.models.composition
```

Key objects:

```text
PredictionInputSpec
PredictionOutputSpec
PredictionIOContract
PredictionIntegrationComponent
IntegratedPredictionModel
```

Current additive and mean integration consume point-vector outputs:

```text
allowed: shape = (n,) or (n, 1)
rejected: shape = (n, k), k > 1, or row-count mismatch
```

## Residual Example

```python
from mlblack.models import PredictionIntegrationComponent
from mlblack.pipeline import ModelConditionedTargetComponent

residual_data = ModelConditionedTargetComponent().build(
    data,
    reference_model=main_model,
)

residual_result = residual_trainer.fit(max_steps=120)

final_model = PredictionIntegrationComponent.additive(
    component_order=("main", "residual"),
).compose(
    {"main": main_model, "residual": residual_result.best_model},
)
```

The Project decides when these Cases run and what resources they receive. The model objects only define ML semantics.

## Multi-Input Example

```python
from mlblack.models import (
    PredictionIOContract,
    PredictionInputSpec,
    PredictionIntegrationComponent,
)

io_contract = PredictionIOContract.by_component(
    {
        "tabular": PredictionInputSpec(key="tabular", ndim=2, n_features=16),
        "image": PredictionInputSpec(key="image", ndim=4),
        "text": PredictionInputSpec(key="input_ids", ndim=2),
    }
)

final_model = PredictionIntegrationComponent.additive(
    component_order=("tabular", "image", "text"),
    weights={"tabular": 0.4, "image": 0.3, "text": 0.3},
    io_contract=io_contract,
).compose({...})
```

## Boundary Checklist

- Does this object define prediction, target transform, metric, or artifact semantics? Put it in `mlblack`.
- Does it choose stage order, fanout, or resource grants? Put it in the shared Project substrate.
- Does it search over structures, configurations, budgets, or tradeoffs? Use `nsgablack` search semantics as a Case.
- Does it need another runnable unit? Make another standard Case rather than adding a hidden runner.
