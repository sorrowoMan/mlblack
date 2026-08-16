"""
Project-level orchestration configuration.
"""
from typing import Dict, Any, List

# Project-level L0 grant. Cases declare/consume requirements; only the project
# allocator grants ResourceContext for a run.
L0: Dict[str, Any] = {
    "namespace": "default_project",
    "offer": {
        "threads": 4,
        "gpus": 0,
        "backend": "local",
        "device_tokens": [],
    },
    "policy": {
        "mode": "strict",
        "gpu_sharing": "exclusive",
        "cpu_oversubscribe": False,
    },
    "default_request": {
        "threads": 1,
        "gpus": 0,
        "backend": "local",
    },
    "compute_backend": "auto",
    "execution_backend": "local",
}

# Define stages of execution. Each stage is a group of standard cases that can
# be run together by the unified Project/Case/L0 substrate.
STAGES: List[Dict[str, Any]] = [
    {
        "name": "stage_1",
        "cases": ["case_a"],
        "policy": "serial",
        # Optional per-case override. The project L0 allocator is still the
        # only place that grants a ResourceContext.
        "resource_requests": {
            "case_a": {"threads": 1, "gpus": 0, "backend": "local"},
        },
    },
]

# Define groups of solvers. This can be used for more complex orchestration.
GROUPS: Dict[str, Any] = {
    "default": {
        "stages": ["stage_1"]
    }
}
