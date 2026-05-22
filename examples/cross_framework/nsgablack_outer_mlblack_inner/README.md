# nsgablack outer -> mlblack inner resource-context case

This case demonstrates the cross-framework boundary without importing nsgablack.

Expected production shape:

```text
nsgablack outer allocator
  -> acquire ResourceLease
  -> inject JSON-compatible ResourceContext
  -> mlblack TrainingProxy/build_trainer
  -> return objectives/constraints/report/artifact metadata
```

The example simulates the injected `ResourceContext` so the mlblack side can be tested standalone.
