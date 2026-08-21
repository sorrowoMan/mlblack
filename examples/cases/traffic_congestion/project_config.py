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
            "xgboost_baseline",
        ],
        "resource_requests": {
            "arimax_factor_attribution": {"threads": 1, "device": "cpu", "backend": "local"},
            "gam_linearity_check": {"threads": 1, "device": "cpu", "backend": "local"},
            "granger_causality_check": {"threads": 1, "device": "cpu", "backend": "local"},
            "shap_contribution_check": {"threads": 1, "device": "cpu", "backend": "local"},
            "xgboost_baseline": {"threads": 1, "device": "cpu", "backend": "local"},
        },
        "component_overrides": {
            "arimax_factor_attribution": {
                "config": {
                    "ar_order": 1,
                    "ma_order": 0,
                    "diff": 0,
                    "max_rows": 120,
                    "max_factor_groups": 2,
                }
            },
            "gam_linearity_check": {
                "config": {"n_knots": 4, "degree": 2, "top_k": 4, "max_rows": 160}
            },
            "granger_causality_check": {
                "config": {"maxlag": 2, "alpha": 0.05, "max_rows": 160, "max_factors": 3}
            },
            "shap_contribution_check": {
                "config": {"n_estimators": 20, "top_k": 5, "max_rows": 160}
            },
            "xgboost_baseline": {
                "config": {"population_size": 4, "mutation_scale": 0.2}
            },
        },
    },
    {
        "name": "symbolic_learning",
        "cases": ["symbolic_regression"],
        "resource_requests": {
            "symbolic_regression": {"threads": 1, "device": "cpu", "backend": "local"},
        },
        "component_overrides": {
            "symbolic_regression": {
                "config": {"learning_rate": 0.01},
                "max_steps": 2,
            },
        },
    },
]

GROUPS = {
    "default": {"stages": ["diagnostics", "symbolic_learning"]},
    "symbolic": {"stages": ["symbolic_learning"]},
    "all": {"stages": ["diagnostics", "symbolic_learning"]},
}
