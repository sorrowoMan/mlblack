# ARCHITECTURE_PURPOSE

## 1) Core Purpose

`mlblack` is a **composable function-cluster learning framework**, not a single model.

The first principle is:

**do not restrict function families; compose them.**

That means:

- any hypothesis family (`linear`, `symbolic`, `piecewise`, `tree`, `nn`, hybrid) is valid
- any capability (`gradient diagnostics`, `path memory`, `gating`, `interval calibration`, robustness tooling) is pluggable
- training flow is orchestration-first, not script-first

## 2) Architectural Intent

The framework should answer three questions explicitly:

1. **What function family are we searching in?**  
   Representation / trainer choice.
2. **What capability stack is active?**  
   Capability composition and lifecycle hooks.
3. **What lifecycle stage are we in?**  
   Standardized flow stages with reproducible transitions.

## 3) Capability Lifecycle (Flow-Level)

`run_train_flow` now supports a capability lifecycle manager.

Supported hooks:

- `on_flow_start`
- `on_data_ready`
- `on_pre_fit`
- `on_post_fit`
- `on_pre_eval`
- `on_post_eval`
- `on_pre_persist`
- `on_post_persist`
- `on_flow_finish`

The flow report includes capability profile and context-contract metadata.

## 4) Non-Negotiable Constraints

- Keep model family and capability layer decoupled.
- Keep orchestration semantics stable when adding new trainers.
- Keep outputs unified (`artifact + report + replay`) regardless of internal combination.
- Keep capability failures observable (strict/fail-fast or soft/warn).

## 5) Example (Minimal)

```python
from dataclasses import dataclass
from workflow import FlowCapability, TrainFlowSpec, run_train_flow


@dataclass
class TraceCapability(FlowCapability):
    def on_post_eval(self, context):
        metrics = context.get("metrics", {})
        print("post-eval metrics keys:", sorted(metrics.keys()))
```

Attach it through:

- `TrainFlowSpec(capabilities=(TraceCapability(name="trace"),), capability_strict=False)`

