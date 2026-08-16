# -*- coding: utf-8 -*-
"""Project-level orchestration config for the granger_causality example."""

from __future__ import annotations

PROJECT_NAME = "granger_causality"

L0 = {
    "namespace": "examples.cases.granger_causality",
    "offer": {"threads": 4, "device": "cpu", "backend": "local"},
    "policy": {"mode": "auto", "cpu_oversubscribe": True},
    "default_request": {"threads": 1, "device": "cpu", "backend": "local"},
    "compute_backend": "auto",
    "execution_backend": "local",
}

STAGES = [
    {
        "name": "main",
        "cases": ['granger_causality'],
        "resource_requests": {"granger_causality": {"threads": 1, "device": "cpu", "backend": "local"}},
    },
]

GROUPS = {
    "default": {"stages": ["main"]},
}
