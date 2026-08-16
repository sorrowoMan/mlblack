# -*- coding: utf-8 -*-
"""Project-level orchestration config for the traffic_congestion example."""

from __future__ import annotations

PROJECT_NAME = "traffic_congestion"

L0 = {
    "namespace": "examples.cases.traffic_congestion",
    "offer": {"threads": 4, "device": "cpu", "backend": "local"},
    "policy": {"mode": "auto", "cpu_oversubscribe": True},
    "default_request": {"threads": 1, "device": "cpu", "backend": "local"},
    "compute_backend": "auto",
    "execution_backend": "local",
}

STAGES = [
    {
        "name": "diagnostics",
        "cases": [
            "arimax_factor_attribution",
            "gam_linearity_check",
            "granger_causality_check",
            "shap_contribution_check",
            "symbolic_regression",
            "xgboost_baseline",
        ],
        "resource_requests": {
            "arimax_factor_attribution": {"threads": 1, "device": "cpu", "backend": "local"},
            "gam_linearity_check": {"threads": 1, "device": "cpu", "backend": "local"},
            "granger_causality_check": {"threads": 1, "device": "cpu", "backend": "local"},
            "shap_contribution_check": {"threads": 1, "device": "cpu", "backend": "local"},
            "symbolic_regression": {"threads": 1, "device": "cpu", "backend": "local"},
            "xgboost_baseline": {"threads": 1, "device": "cpu", "backend": "local"},
        },
        "case_modes": {
            "arimax_factor_attribution": "cli",
            "gam_linearity_check": "cli",
            "granger_causality_check": "cli",
            "shap_contribution_check": "cli",
            "symbolic_regression": "cli",
            "xgboost_baseline": "cli",
        },
    },
    {
        "name": "symbolic_outer_search",
        "cases": [
            "symbolic_mechanism_outer",
            "symbolic_interval_outer",
        ],
        "resource_requests": {
            "symbolic_mechanism_outer": {"threads": 2, "device": "cpu", "backend": "local"},
            "symbolic_interval_outer": {"threads": 2, "device": "cpu", "backend": "local"},
        },
        "case_modes": {
            "symbolic_mechanism_outer": "cli",
            "symbolic_interval_outer": "cli",
        },
        "case_args": {
            "symbolic_mechanism_outer": [
                "--pop-size",
                "32",
                "--generations",
                "25",
                "--rolling-folds",
                "3",
            ],
            "symbolic_interval_outer": [
                "--pop-size",
                "32",
                "--generations",
                "25",
                "--rolling-folds",
                "3",
            ],
        },
    },
]

GROUPS = {
    "default": {"stages": ["diagnostics"]},
    "symbolic": {"stages": ["symbolic_outer_search"]},
    "all": {"stages": ["diagnostics", "symbolic_outer_search"]},
}
