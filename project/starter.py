"""Small NumPy-only ML Case defaults with real gradient updates."""
import numpy as np
from mlblack.pipeline.data_views import NumericDataView
from mlblack.problems import SupervisedRegressionProblem
from mlblack.representations import OrthogonalPointLinearRepresentation
from mlblack.integrations import build_learning_solver, build_optimization_adapter


def build_starter_data():
    x = np.linspace(-1.0, 1.0, 12).reshape(-1, 1)
    return NumericDataView(X_train=x, y_train=1.0 + 2.5 * x[:, 0], feature_names=("x0",))


def build_starter_trainer(data, *, resource_context=None, component_overrides=None):
    from nsgablack.project import apply_solver_component_overrides
    options = dict(component_overrides or {})
    representation = OrthogonalPointLinearRepresentation.from_data(
        data.X_train, feature_names=data.effective_feature_names)
    solver = build_learning_solver(
        problem=SupervisedRegressionProblem(data), representation=representation,
        adapter=build_optimization_adapter("sgd", learning_rate=0.05),
        resource_context=resource_context, run_name="starter_linear_regression",
    )
    solver.set_max_steps(2)
    apply_solver_component_overrides(solver, options)
    return solver
