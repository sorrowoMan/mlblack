"""正式 Project CLI：由共享 Project/Case/L0 运行器执行。"""
from pathlib import Path
from mlblack.project.project_runner import main

if __name__ == "__main__":
    raise SystemExit(main(Path(__file__).resolve().parent))
