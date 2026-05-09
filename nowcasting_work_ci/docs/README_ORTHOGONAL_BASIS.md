# Orthogonal Basis Symbolic Track

This scenario track validates a two-stage idea on `work_ci`:

1. discover a small set of relatively orthogonal symbolic basis terms
2. fit a small-budget symbolic assembler on top of that basis set

Main entry:

- [run_nowcasting_orthogonal_symbolic_work_ci.py](/C:/Users/hp/Desktop/mlblack/nowcasting_work_ci/run_nowcasting_orthogonal_symbolic_work_ci.py)

Example:

```powershell
python C:\Users\hp\Desktop\mlblack\nowcasting_work_ci\run_nowcasting_orthogonal_symbolic_work_ci.py `
  --test-fold-col test_fold_10 `
  --candidate-limit 96 `
  --group-count 12 `
  --max-basis-count 6 `
  --selection-mode interval_first
```

What the report surfaces:

- screened candidate pool
- discovered orthogonal basis groups
- basis overlap / basis semantics payloads
- selected L2 budget for the final assembler
- test RMSE + interval metrics

Interpretation note:

- "orthogonal" here is relative to the current dataset distribution
- it is scored mainly through pairwise basis correlation, feature overlap, and condition number
- it is not yet a hard algebraic-orthogonality solver
