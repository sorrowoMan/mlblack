# symbolic_orthogonal_nested

Formal cross-framework scaffold:

- Stage 1: `nsgablack` outer NSGA-II searches symbolic basis structure.
- Stage 1 inner: `mlblack` fits parameters for each decoded basis set and returns orthogonality metrics.
- Stage 2: `nsgablack` outer NSGA-II searches task expressions over the Stage 1 basis artifact.
- Stage 2 inner: `mlblack` fits task parameters with the configured head/problem.

Run a scaffold check:

```powershell
python examples\cases\symbolic_orthogonal_nested\run_solver.py --check
```

Run a small smoke:

```powershell
python examples\cases\symbolic_orthogonal_nested\run_solver.py --stage1-generations 1 --stage2-generations 1 --stage1-pop-size 4 --stage2-pop-size 4 --stage1-inner-steps 2 --stage2-inner-steps 2
```

The case writes `summary.json`, Stage 1 records, Stage 2 records, and typed symbolic artifacts under `runs/<suite_id>/`.
