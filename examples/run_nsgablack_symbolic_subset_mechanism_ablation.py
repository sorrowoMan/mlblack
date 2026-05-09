from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.components.mechanism_ablation_component import run_mechanism_ablation


def main() -> None:
    parser = argparse.ArgumentParser(description="One-shot mechanism ablation runner for symbolic subset bridge example.")
    parser.add_argument(
        "--target-script",
        type=str,
        default=str(Path(__file__).resolve().parent / "run_nsgablack_symbolic_subset_bridge_work_ci.py"),
    )
    parser.add_argument("--out-dir", type=str, default=str(Path(__file__).resolve().parent / "out" / "mechanism_ablation"))
    parser.add_argument("--base-args", type=str, default="--pop-size 32 --generations 25 --rolling-folds 3 --seed 42")
    args = parser.parse_args()

    target = Path(args.target_script).resolve()
    out_dir = Path(args.out_dir).resolve()
    base_args = [v for v in str(args.base_args).split(" ") if str(v).strip()]
    result_json = run_mechanism_ablation(
        target_script=target,
        out_dir=out_dir,
        base_args=base_args,
    )
    print("MECHANISM_ABLATION_DONE")
    print(f"result_json={result_json}")


if __name__ == "__main__":
    main()
