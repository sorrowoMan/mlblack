from __future__ import annotations

from typing import Any, Mapping


def default_inner_training_config(*, preset: str = "orthogonal_linear_point", run_name: str = "mlblack_run") -> dict[str, Any]:
    return {
        "name": run_name,
        "pipeline": {"components": ["identity"]},
        "trainer": {
            "preset": preset,
            "run_name": run_name,
            "params": {},
            "capabilities": ["resource_audit"],
            "biases": [],
        },
        "resource_context": {},
        "report": {"format": "dict"},
    }


def default_scaffold_config(*, name: str = "mlblack_project", features: tuple[str, ...] = ("x0",), target: str = "target") -> dict[str, Any]:
    return {
        "name": name,
        "version": "0.1",
        "schema": {
            "features": [{"key": item, "dtype": "numeric", "encoder": "numeric"} for item in features],
            "targets": [{"key": target, "dtype": "numeric"}],
            "strict": True,
        },
        "inner_training": default_inner_training_config(run_name=name),
        "metadata": {"created_by": "mlblack.project.scaffold"},
    }


def merge_config(base: Mapping[str, Any], override: Mapping[str, Any] | None = None) -> dict[str, Any]:
    out = dict(base)
    for key, value in dict(override or {}).items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = merge_config(out[key], value)
        else:
            out[key] = value
    return out

