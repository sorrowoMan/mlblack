from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.path_defaults import apply_env_defaults
from project import load_scaffold_spec, run_project_scaffold


def main() -> None:
    apply_env_defaults()

    parser = argparse.ArgumentParser(description="Run mlblack standard project scaffold")
    parser.add_argument(
        "--config",
        type=str,
        default=str(ROOT / "examples" / "configs" / "work_ci_xgboost_portable.json"),
        help="Path to scaffold config json",
    )
    args = parser.parse_args()

    spec = load_scaffold_spec(args.config)
    result = run_project_scaffold(spec)

    print("SCAFFOLD RUN RESULT")
    print(f"trainer={result.report['trainer_name']}")
    print(f"artifact_id={result.report['artifact']['artifact_id']}")
    for split, m in result.metrics.items():
        print(f"{split:5s} rmse={m['rmse']:.4f}  mae={m['mae']:.4f}  r2={m['r2']:.4f}")
    print(f"output_dir={result.output_dir}")


if __name__ == "__main__":
    main()
