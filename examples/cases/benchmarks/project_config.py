# -*- coding: utf-8 -*-
"""Project-level orchestration config for the benchmarks example."""

from __future__ import annotations

PROJECT_NAME = "benchmarks"

L0 = {
    "namespace": "examples.cases.benchmarks",
    "offer": {"threads": 4, "device": "cpu", "backend": "local"},
    "policy": {"mode": "auto", "cpu_oversubscribe": True},
    "default_request": {"threads": 1, "device": "cpu", "backend": "local"},
    "compute_backend": "auto",
    "execution_backend": "local",
}

STAGES = [
    {
        "name": "main",
        "cases": ['benchmarks'],
        "resource_requests": {"benchmarks": {"threads": 1, "device": "cpu", "backend": "local"}},
    },
]

GROUPS = {
    "default": {"stages": ["main"]},
}
