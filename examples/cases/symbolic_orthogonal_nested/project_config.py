# -*- coding: utf-8 -*-
"""Project-level orchestration config for the symbolic_orthogonal_nested example."""

from __future__ import annotations

PROJECT_NAME = "symbolic_orthogonal_nested"

L0 = {
    "namespace": "examples.cases.symbolic_orthogonal_nested",
    "offer": {"threads": 4, "device": "cpu", "backend": "local"},
    "policy": {"mode": "auto", "cpu_oversubscribe": True},
    "default_request": {"threads": 1, "device": "cpu", "backend": "local"},
    "compute_backend": "auto",
    "execution_backend": "local",
}

STAGES = [
    {
        "name": "main",
        "cases": ['symbolic_orthogonal_nested'],
        "resource_requests": {"symbolic_orthogonal_nested": {"threads": 1, "device": "cpu", "backend": "local"}},
        "case_modes": {"symbolic_orthogonal_nested": "cli"},
    },
]

GROUPS = {
    "default": {"stages": ["main"]},
}
