"""共享 Project runner；没有另写训练、调度或资源授权循环。"""
from pathlib import Path
from mlblack.project.project_runner import main as project_main


def main(argv=None):
    return project_main(project_root=Path(__file__).resolve().parent, argv=argv)


if __name__ == "__main__":
    raise SystemExit(main())
