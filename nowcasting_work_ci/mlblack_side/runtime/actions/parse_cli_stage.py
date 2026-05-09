from __future__ import annotations

from typing import Any, Mapping, Sequence

from ..contracts import RuntimeContextKey, ctx_require, ctx_set
from ..config import parse_runtime_args


def run_parse_cli_stage(context: dict[str, Any], argv: Sequence[str] | None = None) -> Mapping[str, Any]:
    runtime_argv = list(argv) if argv is not None else list(ctx_require(context, RuntimeContextKey.ARGV))
    args = parse_runtime_args(runtime_argv)
    ctx_set(context, RuntimeContextKey.ARGS, args)
    ctx_set(context, RuntimeContextKey.RUNTIME_SEED, int(args.seed))
    return {
        "seed": int(args.seed),
        "interval_method": str(args.interval_method),
        "outer_strategy": str(args.outer_strategy),
    }


__all__ = ["run_parse_cli_stage"]
