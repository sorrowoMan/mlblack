# -*- coding: utf-8 -*-
"""Project-level orchestration config for the etf_temporal_forecast example."""

from __future__ import annotations

PROJECT_NAME = "etf_temporal_forecast"

L0 = {
    "namespace": "examples.cases.etf_temporal_forecast",
    "offer": {"threads": 4, "device": "cpu", "backend": "local"},
    "policy": {"mode": "auto", "cpu_oversubscribe": True},
    "default_request": {"threads": 1, "device": "cpu", "backend": "local"},
    "compute_backend": "auto",
    "execution_backend": "local",
}

STAGES = [
    {
        "name": "main",
        "cases": ['etf_temporal_forecast'],
        "resource_requests": {"etf_temporal_forecast": {"threads": 1, "device": "cpu", "backend": "local"}},
    },
]

GROUPS = {
    "default": {"stages": ["main"]},
}
