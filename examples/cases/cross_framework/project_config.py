# -*- coding: utf-8 -*-
"""Project-level orchestration config for the cross_framework example."""

from __future__ import annotations

PROJECT_NAME = "cross_framework"

L0 = {
    "namespace": "examples.cases.cross_framework",
    "offer": {"threads": 4, "device": "cpu", "backend": "local"},
    "policy": {"mode": "auto", "cpu_oversubscribe": True},
    "default_request": {"threads": 1, "device": "cpu", "backend": "local"},
    "artifacts": {
        "path": ".blackbase/artifacts",
        "allow_unsafe_serializers": True,
    },
    "compute_backend": "auto",
    "execution_backend": "local",
}

STAGES = [
    {
        "name": "outer_search",
        "cases": ["cross_framework"],
        "resource_requests": {
            "cross_framework": {
                "workers": 2,
                "threads": 2,
                "memory_mb": 1024,
                "device": "cpu",
                "backend": "local",
            }
        },
    },
]

GROUPS = {
    "default": {"stages": ["outer_search"]},
}
