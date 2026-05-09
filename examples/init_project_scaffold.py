from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from project import init_project


def main() -> None:
    parser = argparse.ArgumentParser(description="Initialize mlblack standard scaffold project")
    parser.add_argument(
        "--path",
        type=str,
        default=str(ROOT / "my_mlblack_project"),
        help="Target directory for scaffold project",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow scaffold initialization on non-empty directory",
    )
    args = parser.parse_args()

    root = init_project(args.path, force=bool(args.force))
    print("MLBLACK PROJECT SCAFFOLD INITIALIZED")
    print(f"root={root}")
    print(f"config={root / 'configs' / 'train_config.json'}")
    print(f"runner={root / 'run_train.py'}")


if __name__ == "__main__":
    main()
