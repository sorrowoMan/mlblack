# 08. Complex Pattern Catalog

All complex ML patterns should be decomposed into ML semantic components plus shared substrate orchestration. `nsgablack` may search over choices, structures, and tradeoffs, but orchestration itself is Project substrate.

## Ownership Table

| Concern | Owner |
| --- | --- |
| stage order, parallelism, resource grant | shared Project substrate |
| outer candidate search, Pareto tradeoff | `nsgablack` semantic layer |
| data views, trainers, heads, artifacts | `mlblack` semantic layer |
| nested request/result payload | standard Case surface |

## Patterns

| Pattern | mlblack Semantics | Search/Substrate Role |
| --- | --- | --- |
| baseline plus residual | target transform, additive integration | Project stages; optional search over residual model |
| stacking | prediction-as-feature transform, meta trainer | Project runs base Cases then meta Case |
| weighted fusion | `PredictionIOContract`, integration weights | optional search over component subset and weights |
| multi-modal fusion | per-input contract and branch artifacts | Project runs branches; optional search over branch config |
| local corrector | region transform, additive correction | optional search over region and alpha |
| expert ensemble | expert artifacts and aggregation semantics | Project fanout and optional weight search |
| symbolic plus neural | symbolic artifact, neural artifact, integration | search symbolic structure and residual budget |
| pretrained plus adapter | frozen artifact, adapter artifact | Project controls fine-tune budget and resources |
| multi-head model | head specs and metrics | optional search over heads and loss weights |
| cascade | uncertainty and fallback semantics | search threshold and budget policy |
| distillation | teacher target transform and student trainer | Project runs teacher then student |
| active learning | uncertainty metric and data-view update | Project loop and acquisition schedule |
| architecture search | fixed NeuralGraphSpec training | outer search over graph/spec choices |
| symbolic grammar search | grammar and parameter fitting | outer search over primitives and expression structure |

## Design Rule

If a feature defines how a model predicts or how data is transformed, keep it in `mlblack`.

If a feature decides which Case runs when, with which resources, put it in the shared substrate.

If a feature searches a design space, represent it as a search Case using `nsgablack` semantics.
