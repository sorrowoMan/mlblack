# -*- coding: utf-8 -*-
"""Project orchestration for independent neural benchmark Cases."""

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
        "name": "neural_training",
        "policy": "parallel",
        "max_workers": 4,
        "cases": [
            "benchmark_tiny_cnn_classification",
            "benchmark_tiny_gnn_classification",
            "benchmark_tiny_cnn_contrastive",
            "benchmark_tiny_transformer_lm",
        ],
        "resource_requests": {
            "benchmark_tiny_cnn_classification": {"threads": 1, "device": "cpu", "backend": "local"},
            "benchmark_tiny_gnn_classification": {"threads": 1, "device": "cpu", "backend": "local"},
            "benchmark_tiny_cnn_contrastive": {"threads": 1, "device": "cpu", "backend": "local"},
            "benchmark_tiny_transformer_lm": {"threads": 1, "device": "cpu", "backend": "local"},
        },
        "component_overrides": {
            "benchmark_tiny_cnn_classification": {"max_steps": 2},
            "benchmark_tiny_gnn_classification": {"max_steps": 2},
            "benchmark_tiny_cnn_contrastive": {"max_steps": 2},
            "benchmark_tiny_transformer_lm": {"max_steps": 2},
        },
    },
]

GROUPS = {"default": {"stages": ["neural_training"]}}
