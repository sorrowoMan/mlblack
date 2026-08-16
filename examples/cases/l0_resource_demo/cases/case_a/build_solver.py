"""
Build a solver that demonstrates L0 resource capabilities.

This example shows how mlblack can leverage blackbase's L0 resource layer
for resource detection, context management, and audit capabilities.
"""

from __future__ import annotations

import numpy as np
from typing import Any, Mapping

from mlblack.core import Trainer, ResourceContext, LearningProblem, OptimizerAdapter
from mlblack.core.types import Feedback

try:
    from .pipeline.main import build_pipeline
except ImportError:  # direct execution from the Case directory
    from pipeline.main import build_pipeline


class L0ResourceDemoProblem(LearningProblem):
    """
    Demonstration problem that showcases L0 resource capabilities.
    
    This problem evaluates candidate solutions while demonstrating:
    - Resource context consumption
    - Resource audit capabilities
    - System resource detection
    """
    
    def evaluate(self, model, state, context=None):
        """Evaluate decoded model and return objectives/constraints/signals."""
        x = np.asarray(state.values)
        objectives = np.array([np.sum(x**2)])
        constraints = np.array([])
        
        metrics = {}
        resource_ctx = context.get("resource_context") if context else None
        
        if resource_ctx:
            metrics["resource_namespace"] = resource_ctx.get("namespace", "unknown")
            metrics["resource_threads"] = resource_ctx.get("resources", {}).get("threads", 1)
            metrics["resource_gpus"] = resource_ctx.get("resources", {}).get("gpus", 0)
            metrics["resource_backend"] = resource_ctx.get("resources", {}).get("backend", "local")
        
        print(f"  [L0 Demo] Evaluated candidate with metrics: {metrics}")
        
        return Feedback(
            objectives=objectives,
            constraints=constraints,
            metrics=metrics,
        )


class L0ResourceDemoAdapter(OptimizerAdapter):
    """
    Simple adapter for L0 resource demonstration.
    
    Uses random search to propose candidates while respecting
    the resource constraints from the ResourceContext.
    """
    
    name = "l0_random_search"

    def propose(self, solver, context=None):
        """Propose new candidates."""
        resource_context = context.get("resource_context") if context else None
        
        if resource_context:
            seed = resource_context.get("resources", {}).get("threads", 1)
            np.random.seed(seed)
        
        representation = solver.representation_pipeline
        candidates = [representation.init(context or {}) for _ in range(3)]
        return candidates

    def update(self, solver, candidates, feedback, context=None):
        """Random-search demo has no persistent strategy state to update."""

        del solver, candidates, feedback, context
        return None


def build_solver(config=None, *, resource_context=None, component_overrides=None):
    """Assemble the L0 resource demo through the canonical Case entry."""

    del config
    overrides = dict(component_overrides or {})
    problem = overrides.get("problem") or L0ResourceDemoProblem()
    representation = build_pipeline(
        resource_context=resource_context,
        component_overrides=overrides,
    )
    adapter = overrides.get("adapter") or L0ResourceDemoAdapter()
    grant = resource_context or ResourceContext(
        scope="training",
        execution_backend="local",
        compute_backend="auto",
        device="cpu",
        threads=1,
        namespace="l0_resource_demo.case_a",
    )
    return Trainer(
        problem=problem,
        representation=representation,
        adapter=adapter,
        run_name="l0_resource_demo",
        resource_context=grant,
    )
