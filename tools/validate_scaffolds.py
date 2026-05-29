#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Validate generated example scaffolds under examples/cases.

Checks performed:
- `config.py` exists and defines `get_project_config`
- `run_solver.py` exists and can be imported (does not execute heavy compute)

This script is safe: it imports the generated files but does not execute original heavy workloads.
"""
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
CASES_DIR = ROOT / 'examples' / 'cases'

results = {}
for case in sorted(CASES_DIR.iterdir()):
    if not case.is_dir():
        continue
    project_config = case / 'config.py'
    run_entry = case / 'run_solver.py'
    ok = True
    msgs = []
    if not project_config.exists():
        ok = False
        msgs.append('config.py missing')
    else:
        try:
            sys.path.insert(0, str(ROOT))
            mod = runpy.run_path(str(project_config))
            if 'get_project_config' not in mod:
                msgs.append('get_project_config not found in generated config.py')
        except Exception as e:
            ok = False
            msgs.append(f'exception importing config.py: {e}')
    if not run_entry.exists():
        msgs.append('run_solver.py missing')
    else:
        try:
            # import; this may execute the small wrapper but original heavy work resides in original/*
            runpy.run_path(str(run_entry))
        except Exception as e:
            msgs.append(f'run_solver import error: {e}')
    results[case.name] = (ok, msgs)

for case, (ok, msgs) in results.items():
    print(f'{case}:', 'OK' if ok and not msgs else 'WARN', msgs)

# Exit code non-zero if any failures
failed = any((not ok) or msgs for ok, msgs in results.values())
if failed:
    raise SystemExit(2)
