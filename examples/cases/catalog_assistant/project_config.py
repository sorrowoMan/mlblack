# -*- coding: utf-8 -*-
"""Project-level orchestration config for the catalog_assistant example."""

from __future__ import annotations

PROJECT_NAME = "catalog_assistant"

L0 = {
    "namespace": "examples.cases.catalog_assistant",
    "offer": {"threads": 4, "device": "cpu", "backend": "local"},
    "policy": {"mode": "auto", "cpu_oversubscribe": True},
    "default_request": {"threads": 1, "device": "cpu", "backend": "local"},
    "compute_backend": "auto",
    "execution_backend": "local",
}

STAGES = [
    {
        "name": "main",
        "cases": ['catalog_assistant'],
        "resource_requests": {"catalog_assistant": {"threads": 1, "device": "cpu", "backend": "local"}},
    },
]

GROUPS = {
    "default": {"stages": ["main"]},
}
