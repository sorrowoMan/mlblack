"""唯一规范装配入口：ML 语义组件加共享优化控制面。"""
from mlblack.integrations.nsgablack_control import build_learning_solver
from mlblack.representations import LinearPointConfig, LinearPointRepresentation
from nsgablack.project import apply_solver_component_overrides

from .adapter import build_adapter
from .config import DEFAULTS
from .pipeline import build_pipeline
from .problem import build_problem


def build_solver(config=None, *, resource_context=None, component_overrides=None):
    settings = {**DEFAULTS, **dict(config or {})}
    data = build_pipeline(seed=settings["seed"], n_samples=settings["n_samples"])
    representation = LinearPointRepresentation(LinearPointConfig(
        n_features=data.n_features,
        feature_names=data.effective_feature_names,
        random_seed=int(settings["seed"]),
    ))
    solver = build_learning_solver(
        problem=build_problem(data),
        representation=representation,
        adapter=build_adapter(learning_rate=settings["learning_rate"]),
        resource_context=resource_context,
        run_name="catalog_linear_point",
    )
    solver.set_random_seed(int(settings["seed"]))
    solver.set_max_steps(int(settings["max_steps"]))
    apply_solver_component_overrides(solver, dict(component_overrides or {}))
    return solver
