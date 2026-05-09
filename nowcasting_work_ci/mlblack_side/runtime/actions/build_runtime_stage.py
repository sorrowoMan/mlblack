from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from ..build_runtime import build_runtime
from ..contracts import RuntimeContextKey, ctx_require, ctx_set


def run_build_runtime_stage(context: dict[str, Any]) -> Mapping[str, Any]:
    args = ctx_require(context, RuntimeContextKey.ARGS)
    prepared = build_runtime(args)
    ctx_set(context, RuntimeContextKey.PREPARED, prepared)
    ctx_set(context, RuntimeContextKey.OUTPUT_ROOT, str(prepared["out_root"]))
    ctx_set(context, RuntimeContextKey.GRAPH_CACHE_RESOURCE, prepared.get("graph_cache"))
    return {
        "out_root": str(prepared["out_root"]),
        "n_train": int(np.asarray(prepared["X_train"]).shape[0]),
        "n_test": int(np.asarray(prepared["X_test"]).shape[0]),
        "n_features": int(np.asarray(prepared["X_train"]).shape[1]),
        "candidate_pool_size": int(len(prepared["candidates"])),
    }


__all__ = ["run_build_runtime_stage"]
