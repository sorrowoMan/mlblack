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
        "name": "training",
        "cases": [
            "tiny_transformer_classification",
            "tiny_transformer_language_model",
            "tiny_transformer_preference",
        ],
        "resource_requests": {
            "tiny_transformer_classification": {
                "threads": 1,
                "device": "cpu",
                "backend": "local",
            },
            "tiny_transformer_language_model": {
                "threads": 1,
                "device": "cpu",
                "backend": "local",
            },
            "tiny_transformer_preference": {
                "threads": 1,
                "device": "cpu",
                "backend": "local",
            },
        },
        "case_modes": {
            "tiny_transformer_classification": "cli",
            "tiny_transformer_language_model": "cli",
            "tiny_transformer_preference": "cli",
        },
        "case_args": {
            "tiny_transformer_classification": ["--steps", "2"],
            "tiny_transformer_language_model": ["--steps", "2"],
            "tiny_transformer_preference": ["--steps", "1"],
        },
    },
]

GROUPS = {
    "default": {"stages": ["training"]},
}
