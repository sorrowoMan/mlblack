from __future__ import annotations

from typing import Mapping, Sequence

from nowcasting_work_ci.mlblack_side.runtime.stages import run_stage_sequence


def main(argv: Sequence[str] | None = None) -> Mapping[str, object]:
    return run_stage_sequence(argv)


__all__ = ["main"]


if __name__ == "__main__":
    main()
