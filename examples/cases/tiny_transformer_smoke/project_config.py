# -*- coding: utf-8 -*-
"""Project-level orchestration config for the tiny_transformer_smoke example."""

from __future__ import annotations

PROJECT_NAME = "tiny_transformer_smoke"

L0 = {
    "namespace": "examples.cases.tiny_transformer_smoke",
    "offer": {"threads": 4, "device": "cpu", "backend": "local"},
    "policy": {"mode": "auto", "cpu_oversubscribe": True},
    "default_request": {"threads": 1, "device": "cpu", "backend": "local"},
    "compute_backend": "auto",
    "execution_backend": "local",
}

STAGES = [
    {
        "name": "main",
        "cases": ['tiny_transformer_smoke'],
        "resource_requests": {"tiny_transformer_smoke": {"threads": 1, "device": "cpu", "backend": "local"}},
        "case_modes": {"tiny_transformer_smoke": "cli"},
    },
]

GROUPS = {
    "default": {"stages": ["main"]},
}
