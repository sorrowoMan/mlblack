# 04. NeuralGraph Decoder Design

This page keeps the current NeuralGraph architecture conclusion in the shared substrate era.

## Core Judgment

Neural networks are not optimizers. They are parameterized computation graphs.

In `mlblack`, a Transformer is a NeuralGraph preset:

```text
NeuralGraphSpec
  -> NeuralGraphCodec
  -> NeuralGraphRepresentation
  -> LearningProblem
  -> nsgablack AlgorithmAdapter
  -> Artifact
```

Project stages, parallelism, and resource grants are handled by the shared Project / Case / L0 substrate.

## Layer Mapping

| Transformer Concept | mlblack Layer | Meaning |
| --- | --- | --- |
| token embedding / attention / FFN / norm | `NeuralGraphSpec` + backend lowering | model structure |
| flat parameter vector | `UnknownState` | optimized state |
| parameter layout / decode | `NeuralGraphCodec` | state to model mapping |
| classification / LM / embedding / preference | head spec / problem head | output semantics |
| cross entropy / DPO / triplet | `LearningProblem` + backend loss | evaluation |
| backward / state transition | nsgablack Adapter + Evaluation Provider capability | Adapter 选择更新机制，Provider 执行后端原语 |
| attention map / activation / parameter summary | artifact/report | audit |

## Runtime Chain

```text
LearningSolver
  -> NeuralGraphRepresentation / NeuralGraphCodec
  -> LearningProblem builds EvaluationRequest
  -> Evaluation Provider capability preflight
  -> backend session executes loss / gradient / state transition
  -> OptimizationFeedbackBatch
  -> nsgablack Adapter.update
  -> ArtifactBuilder
```

## Backend Boundary

| Layer | Responsibility |
| --- | --- |
| Codec | defines how unknown state becomes a model |
| Backend / Provider | defines tensor lowering and executable loss, gradient, state-transition and artifact primitives |
| LearningSolver | projects ML semantics onto the canonical NSGABlack lifecycle |
| Project L0 | grants resources and records effective runtime |

Backends are not equivalent. A route that requires stateful module backward must fail fast on a backend that only supports functional gradients.

## Search Boundary

Neural architecture search should be expressed as:

```text
outer search Case:
  searches NeuralGraphSpec fields

inner mlblack Case:
  trains or evaluates one fixed spec
  returns loss, complexity, runtime, artifacts, audit
```

This is the same shape as symbolic nested search: outer searches structure, inner fits parameters.

## Do Not Add

```text
backend selection inside Codec
Problem directly choosing a backend
fake backend capability declarations
private mlblack orchestration stack
private resource allocator
```

Use `docs/neural_graph_backend_architecture/` for backend capability details.
