# symbolic_orthogonal_nested

Formal cross-framework scaffold:

- Stage 1: `nsgablack` outer NSGA-II searches symbolic basis structure.
- Stage 1 inner: `mlblack` fits parameters for each decoded basis set and returns orthogonality metrics.
- Stage 2: `nsgablack` outer NSGA-II searches task expressions over the Stage 1 basis artifact.
- Stage 2 inner: `mlblack` fits task parameters with the configured head/problem.

Run a scaffold check:

```powershell
python examples\cases\symbolic_orthogonal_nested\run_project.py --check --build-check
```

Run a small smoke:

```powershell
python examples\cases\symbolic_orthogonal_nested\run_project.py
```

Project 配置已经提供可执行的 smoke profile；需要扩大规模时，在
`project_config.py` 的正式 component overrides 中调整配置，而不是向 build-mode Case
转发旧 CLI 参数。

The case writes `summary.json`, Stage 1 records, Stage 2 records, and typed symbolic artifacts under `runs/<suite_id>/`.
