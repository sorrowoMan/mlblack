# Tiny Transformer Smoke Case

This case proves that the current tiny Transformer scaffold can run end-to-end:

- classification trainer
- language-model trainer
- generation with KV cache
- DPO/preference trainer
- neural graph artifact viewer

Run from the repo root:

```powershell
python examples\cases\tiny_transformer_smoke\run_case.py --steps 2
```

Outputs:

```text
examples/cases/tiny_transformer_smoke/runs/latest/summary.json
examples/cases/tiny_transformer_smoke/runs/latest/classification_artifact.html
```

Scope:

- This is a tiny local smoke case, not production LLM training.
- It does not define a second workflow/runtime.
- nsgablack remains the outer orchestration/search/runtime owner.
