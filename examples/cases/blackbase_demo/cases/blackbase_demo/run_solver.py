"""blackbase substrate 演示 Case 的唯一正式 CLI 入口。"""

from __future__ import annotations

import argparse
import json

try:
    from .build_solver import build_solver
except ImportError:  # direct script execution
    from build_solver import build_solver


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="运行 blackbase substrate 演示。")
    parser.add_argument("--check", action="store_true", help="只检查正式装配，不执行训练。")
    parser.add_argument("--steps", type=int, default=3)
    args = parser.parse_args(argv)

    trainer = build_solver()
    if args.check:
        print(
            "[check] blackbase_demo assembly ok | "
            f"problem={type(trainer.problem).__name__} | "
            f"pipeline={type(trainer.representation_pipeline).__name__} | "
            f"adapter={type(trainer.adapter).__name__} | "
            f"resource_context={json.dumps(trainer.get_resource_context().as_dict(), ensure_ascii=False)}"
        )
        return 0
    result = trainer.fit(max_steps=max(1, int(args.steps)))
    print(json.dumps(result.report, ensure_ascii=False, default=str, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
