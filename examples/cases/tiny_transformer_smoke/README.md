# tiny_transformer_smoke

This directory is a Project containing three independent standard ML Cases:

- `tiny_transformer_classification`
- `tiny_transformer_language_model`
- `tiny_transformer_preference`

Each Case builds exactly one `LearningSolver`. The Project owns their ordering,
CLI isolation and L0 resource grants.

Build-check all three canonical builders:

```powershell
python examples/cases/tiny_transformer_smoke/run_project.py --check --build-check
```

Run the complete smoke Project:

```powershell
python examples/cases/tiny_transformer_smoke/run_project.py
```

Each Case writes its own bounded artifact/summary output below its Case-local
`runs/latest/` directory. No Case launches another Trainer internally.
