# 02. Component Configuration

## 1) Unified ownership first

- Orchestration/resource authority is shared substrate (Project/L0).
- `mlblack` owns ML semantics (DataView/Spec/Codec/Head/Problem/Trainer/Artifact).
- `nsgablack` owns search semantics.
- A standard Case can be outer or inner.

## 2) `mlblack` pipeline layering

- Case keeps one pipeline primary entry.
- Pipeline entry assembles ML-semantic operators.
- Operators can include transform/codec/head/feature/target logic.

## 3) Recommended shape

```text
pipeline/
  main.py
  operators/
    transform/
    codec/
    head/
    custom/
```

## 4) Rule

- Treat `pipeline` as flow orchestration.
- Treat operator files as internal units.
- Do not expose multiple case-level pipeline primaries.
