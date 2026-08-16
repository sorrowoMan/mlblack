# -*- coding: utf-8 -*-
"""Project-level orchestration config for the tsne_umap example."""

from __future__ import annotations

PROJECT_NAME = "tsne_umap"

L0 = {
    "namespace": "examples.cases.tsne_umap",
    "offer": {"threads": 4, "device": "cpu", "backend": "local"},
    "policy": {"mode": "auto", "cpu_oversubscribe": True},
    "default_request": {"threads": 1, "device": "cpu", "backend": "local"},
    "compute_backend": "auto",
    "execution_backend": "local",
}

STAGES = [
    {
        "name": "main",
        "cases": ['tsne_umap'],
        "resource_requests": {"tsne_umap": {"threads": 1, "device": "cpu", "backend": "local"}},
    },
]

GROUPS = {
    "default": {"stages": ["main"]},
}
