# STANDARD_LAYOUT

This file is the hard layout contract for `nowcasting_work_ci`.

## 1. Top-level goal

Package root is **public surface only**.

Top-level should answer just three questions:

1. What is the preferred entry?
2. What public compatibility shims still exist?
3. Where do I find internal docs and scenario outputs?

Anything else should move down one layer.

## 2. Formal public surface

The long-term public code surface is:

- `run.py`: preferred CLI entry
- `build_solver.py`: stable solver-build import surface
- `assembly.py`: stable runtime-assembly forwarder

The long-term public documentation surface is:

- `README.md`: short package homepage
- `STANDARD_LAYOUT.md`: hard top-level contract

These files may forward into deeper modules, but their filenames and roles are
part of the stable package contract.

## 3. Deprecated surface

The following top-level files remain only as compatibility shims:

- `run_solver.py`
- `aggregate_and_plot_results.py`
- `run_deterministic_smoke_regression.py`
- `run_nowcasting_symbolic_subset_bridge_work_ci.py`
- `run_nowcasting_symbolic_subset_bridge_work_ci_native_interval.py`

Rules:

- deprecated files must stay thin
- no new business logic goes into deprecated files
- real implementations belong in `compat/` or `tools/`
- new callers should prefer the formal public surface

## 4. Allowed top-level contents

Allowed directories:

- `_internal/`: package-private support helpers
- `compat/`: legacy CLI implementations
- `docs/`: scenario-specific documentation
- `mlblack_side/`: scenario runtime/problem/config side
- `nsgablack_side/`: outer solver scaffold side
- `tests/`: scaffold-facing tests
- `tools/`: smoke/report helper tools

Allowed files:

- `README.md`: short package overview
- `STANDARD_LAYOUT.md`: this contract
- `AGENTS.md`: collaboration notes
- `__init__.py`: package bootstrap only
- formal public surface files listed in section 2
- deprecated surface files listed in section 3

`_internal/` is package-private support only, not a public import contract.

## 5. Forbidden top-level contents

The following do **not** belong in package root:

- generated run artifacts
- long-form design docs
- one-off experiment dumps
- heavy implementation modules
- direct report outputs
- scenario caches

Concretely:

- runtime output must not return to `nowcasting_work_ci/out/`
- doc files like `README_*` other than the main `README.md` should live in `docs/`
- real compatibility implementations should live in `compat/`, not package root
- real operational helper implementations should live in `tools/`, not package root

## 6. Public surface rules

Top-level Python files should be one of:

1. preferred entry
2. stable import forwarder
3. deprecated compatibility shim
4. package bootstrap

They should not:

- own business logic
- own report generation logic
- own runtime assembly logic
- decide output directory policy

## 7. Docs rules

`docs/` holds scenario-specific explanation and architecture notes.

Examples:

- runtime contracts
- reporting docs
- architecture split notes
- interval method notes
- problem-model extraction notes

Top-level `README.md` should link into `docs/`, not duplicate all of it.

## 8. Output rules

Generated artifacts belong under:

- `C:\Users\hp\Desktop\mlblack\_scenario_runs\nowcasting_work_ci\`

Optional override:

- env `MLBLACK_SCENARIO_RUNS_ROOT`

Package root must stay clean even after repeated runs.

## 9. Import precedence rules

When both repos are present:

1. `mlblack` repo root must appear before `nsgablack` repo root on `sys.path`
2. `nowcasting_work_ci` should normalize that order on import
3. tests should enforce the same order before importing ambiguous top-level packages such as `bias`, `pipeline`, `plugins`, or `workflow`

This rule exists to prevent cross-repo name drift.
