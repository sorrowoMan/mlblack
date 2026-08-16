# -*- coding: utf-8 -*-
"""Project-level orchestration config for the orthogonal_point_demo example."""

from __future__ import annotations

PROJECT_NAME = "orthogonal_point_demo"

L0 = {
    "namespace": "examples.cases.orthogonal_point_demo",
    "offer": {"threads": 4, "device": "cpu", "backend": "local"},
    "policy": {"mode": "auto", "cpu_oversubscribe": True},
    "default_request": {"threads": 1, "device": "cpu", "backend": "local"},
    "compute_backend": "auto",
    "execution_backend": "local",
}

STAGES = [
    {
        "name": "main",
        "cases": ['orthogonal_point_demo'],
        "resource_requests": {"orthogonal_point_demo": {"threads": 1, "device": "cpu", "backend": "local"}},
    },
]

GROUPS = {
    "default": {"stages": ["main"]},
}
