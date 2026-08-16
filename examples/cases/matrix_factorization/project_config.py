# -*- coding: utf-8 -*-
"""Project-level orchestration config for the matrix_factorization example."""

from __future__ import annotations

PROJECT_NAME = "matrix_factorization"

L0 = {
    "namespace": "examples.cases.matrix_factorization",
    "offer": {"threads": 4, "device": "cpu", "backend": "local"},
    "policy": {"mode": "auto", "cpu_oversubscribe": True},
    "default_request": {"threads": 1, "device": "cpu", "backend": "local"},
    "compute_backend": "auto",
    "execution_backend": "local",
}

STAGES = [
    {
        "name": "main",
        "cases": ['matrix_factorization'],
        "resource_requests": {"matrix_factorization": {"threads": 1, "device": "cpu", "backend": "local"}},
    },
]

GROUPS = {
    "default": {"stages": ["main"]},
}
