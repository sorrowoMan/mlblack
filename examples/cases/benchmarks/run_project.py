from __future__ import annotations

import sys
from pathlib import Path


def _ensure_repo_importable() -> None:
    here = Path(__file__).resolve()
    for parent in (here.parent, *here.parents):
        if (parent / "pyproject.toml").is_file() and (parent / "mlblack.py").is_file():
            repo_parent = str(parent.parent)
            if repo_parent not in sys.path:
                sys.path.insert(0, repo_parent)
            return


def main(argv: list[str] | None = None) -> int:
    _ensure_repo_importable()
    from mlblack.project.project_runner import main as project_main

    return int(project_main(project_root=Path(__file__).resolve().parent, argv=argv))


if __name__ == "__main__":
    raise SystemExit(main())
