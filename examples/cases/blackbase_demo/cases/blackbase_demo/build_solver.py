"""blackbase substrate 演示 Case 的唯一正式装配入口。"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from mlblack.core import ResourceContext, Trainer

try:
    from .adapter import DemoRandomSearchAdapter
    from .pipeline import build_pipeline
    from .problem import SimpleRegressionProblem
except ImportError:  # 允许直接执行同目录下的 run_solver.py
    from adapter import DemoRandomSearchAdapter
    from pipeline import build_pipeline
    from problem import SimpleRegressionProblem


def build_solver(
    config: Mapping[str, Any] | None = None,
    *,
    resource_context: Mapping[str, Any] | ResourceContext | None = None,
    component_overrides: Mapping[str, Any] | None = None,
) -> Trainer:
    """装配一个可运行的 Trainer，并消费 Project 发放的 L0 grant。"""

    settings = dict(config or {})
    overrides = dict(component_overrides or {})
    seed = int(settings.get("seed", 17))
    n_samples = int(settings.get("n_samples", 96))
    n_features = int(settings.get("n_features", 3))
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples, n_features))
    weights = np.arange(1, n_features + 1, dtype=float) / n_features
    y = X @ weights + rng.normal(0.0, 0.02, size=n_samples)

    problem = overrides.get("problem") or SimpleRegressionProblem(X, y)
    representation = build_pipeline(
        n_features,
        seed=seed,
        resource_context=resource_context,
        component_overrides=overrides,
    )
    adapter = overrides.get("adapter") or DemoRandomSearchAdapter(
        batch_size=int(settings.get("batch_size", 4))
    )
    grant = resource_context or ResourceContext(
        scope="training",
        execution_backend="local",
        compute_backend="auto",
        device="cpu",
        threads=1,
        namespace="blackbase_demo.case",
    )
    return Trainer(
        problem=problem,
        representation=representation,
        adapter=adapter,
        run_name="blackbase_substrate_demo",
        resource_context=grant,
    )
