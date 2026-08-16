#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Migrate top-level examples into standardized scaffolds under examples/cases.

Behavior:
- For each entry in mlblack/examples (files and directories), except 'cases',
  create a standardized scaffold under examples/cases/<name> using
  the mlblack trainer case template over the shared blackbase substrate.
- Move original files/directories into the new scaffold's 'original' subdir.
- Create a lightweight run entry `run_solver.py` in the scaffold root that
  imports and invokes a main() from the original script if available, otherwise
  runs the script with runpy.

Run from the mlblack project root.
"""

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = ROOT / "examples"
CASES_DIR = EXAMPLES_DIR / "cases"

SKIP_NAMES = {"cases", "__pycache__"}

_CASE_TEMPLATE = ROOT / "project" / "scaffold" / "trainer_case_template"


def init_case_scaffold(target_dir: Path):
    """Create a unified case scaffold at target_dir using the mlblack template."""
    import shutil as _shutil
    target_dir.mkdir(parents=True, exist_ok=True)
    _shutil.copytree(_CASE_TEMPLATE, target_dir, dirs_exist_ok=True)
    # Rename .template files
    for f in target_dir.glob("**/*.template"):
        f.rename(f.with_suffix(""))


def make_run_entry(target_dir: Path, orig_rel: Path):
    """Generate a run_solver.py that attempts to import main() from original file.
    orig_rel is the relative path from target_dir to the original file/directory.
    """
    run_py = target_dir / "run_solver.py"
    content = f'''# Auto-generated run entry for migrated example
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
# insert project root to sys.path
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

orig = Path(__file__).resolve().parent / "{orig_rel.as_posix()}"

# If original is a file and defines main(), import and call it.
if orig.is_file():
    try:
        # Try to import as module by path
        import importlib.util
        spec = importlib.util.spec_from_file_location("migrated_example", str(orig))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        if hasattr(mod, 'main'):
            mod.main()
        else:
            # fallback: run as script
            runpy.run_path(str(orig), run_name='__main__')
    except Exception as e:
        print('Error running migrated example:', e)
        raise
else:
    # If original is a directory, attempt to run its __main__.py or run_case.py
    if (orig / '__main__.py').exists():
        runpy.run_path(str(orig / '__main__.py'), run_name='__main__')
    elif (orig / 'run_case.py').exists():
        runpy.run_path(str(orig / 'run_case.py'), run_name='__main__')
    else:
        print('No obvious entrypoint in original example:', orig)
'''
    run_py.write_text(content, encoding='utf-8')


def migrate_one(src: Path):
    name = src.stem if src.is_file() else src.name
    target = CASES_DIR / name
    if target.exists():
        print(f"Target {target} exists; skipping migration for {name}")
        return
    print(f"[MIGRATE] Creating scaffold for {name} -> {target}")
    init_case_scaffold(target)
    orig_dst = target / 'original'
    orig_dst.mkdir(exist_ok=True)
    # Move the source into orig_dst
    dst_path = orig_dst / src.name
    print(f"[MIGRATE] Moving {src} -> {dst_path}")
    try:
        shutil.move(str(src), str(dst_path))
    except Exception as e:
        print(f"Failed to move {src}: {e}")
        return
    # Create run entry referencing the moved original
    rel_path = Path('original') / src.name
    make_run_entry(target, rel_path)
    print(f"[MIGRATE] Completed {name}")


def main():
    EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    CASES_DIR.mkdir(parents=True, exist_ok=True)
    entries = [p for p in EXAMPLES_DIR.iterdir() if p.name not in SKIP_NAMES]
    print('Found entries to consider for migration:')
    for p in entries:
        print('  -', p.name)
    for p in entries:
        migrate_one(p)
    print('\nMigration finished.')


if __name__ == '__main__':
    main()
