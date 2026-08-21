# -*- coding: utf-8 -*-
"""Project-level orchestration config for the temporal_neural_compare example."""

from __future__ import annotations

PROJECT_NAME = "temporal_neural_compare"

L0 = {
    "namespace": "examples.cases.temporal_neural_compare",
    "offer": {"threads": 4, "device": "cpu", "backend": "local"},
    "policy": {"mode": "auto", "cpu_oversubscribe": True},
    "default_request": {"threads": 1, "device": "cpu", "backend": "local"},
    "compute_backend": "auto",
    "execution_backend": "local",
}

STAGES = [
    {
        "name": "model_comparison",
        "cases": [
            "temporal_lstm",
            "temporal_tcn",
            "temporal_transformer",
            "temporal_nbeats",
            "temporal_deepar",
            "temporal_patchtst",
            "temporal_tft",
        ],
        "resource_requests": {
            case_name: {"threads": 1, "device": "cpu", "backend": "local"}
            for case_name in (
                "temporal_lstm",
                "temporal_tcn",
                "temporal_transformer",
                "temporal_nbeats",
                "temporal_deepar",
                "temporal_patchtst",
                "temporal_tft",
            )
        },
        "component_overrides": {
            case_name: {
                "config": {
                    "n_train": 32,
                    "n_valid": 8,
                    "sequence_length": 12,
                    "noise_std": 0.1,
                    "random_seed": 42,
                },
                "max_steps": 2,
            }
            for case_name in (
                "temporal_lstm",
                "temporal_tcn",
                "temporal_transformer",
                "temporal_nbeats",
                "temporal_deepar",
                "temporal_patchtst",
                "temporal_tft",
            )
        },
    },
]

GROUPS = {
    "default": {"stages": ["model_comparison"]},
}
