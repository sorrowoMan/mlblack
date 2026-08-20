# cross_framework

This Project demonstrates a real nested cross-framework execution:

```text
cross_framework (nsgablack outer Solver Case)
  -> Problem.evaluate(log10_learning_rate)
     -> BlackBase CaseRunRequest
        -> inner_training (mlblack LearningSolver Case)
```

The outer Case receives one Project L0 grant. Every inner invocation receives a
child grant, lineage, cancellation chain and result envelope derived by
BlackBase; the example does not synthesize a resource context.

The inner linear model requires the ML pickle serializer, so this local example
explicitly opts its Project artifact authority into unsafe serializers. The
choice is visible in `project_config.py`; it is never enabled by a child Case.

Build-check both canonical builders:

```powershell
python examples/cases/cross_framework/run_project.py --check --build-check
```

Run the bounded nested search:

```powershell
python examples/cases/cross_framework/run_project.py
```
