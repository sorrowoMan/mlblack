"""blackbase substrate 演示项目的顶层入口。"""

from __future__ import annotations

from pathlib import Path


def main(argv=None) -> int:
    from mlblack.project.project_runner import main as project_main

    return int(project_main(project_root=Path(__file__).resolve().parent, argv=argv))


if __name__ == "__main__":
    raise SystemExit(main())
