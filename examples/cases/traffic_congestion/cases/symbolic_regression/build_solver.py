import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent))
from typing import Mapping
import numpy as np
from mlblack.core.trainer import ComposableTrainer
from mlblack.adapters.gradient_descent import GradientDescentAdapter, GradientDescentConfig
from mlblack.core.problem import LearningProblem
from mlblack.core.types import Feedback

try:
    from .pipeline import add_intercept as _add_intercept
    from .pipeline import build_pipeline, build_representation
except ImportError:
    from pipeline import add_intercept as _add_intercept
    from pipeline import build_pipeline, build_representation

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
CSV_PATH = DATA_DIR / "ci_interval_opt_table_no_flow_speed_occ_lag.csv"


class CIDirectRegressionProblem(LearningProblem):
    """Direct point regression on CI target using simple linear model with intercept."""
    context_requires = ()
    context_provides = ("feedback.objectives", "feedback.gradients")
    context_mutates = ()
    context_cache = ()
    context_notes = "MSE regression on traffic CI target with intercept bias column."

    def __init__(self, data, *, name="ci_regression"):
        self.name = str(name)
        self._X = _add_intercept(data.X_train)
        self._y = np.asarray(data.y_train, dtype=float).ravel()
        self._X_val = _add_intercept(data.X_valid) if data.X_valid is not None else None
        self._y_val = (
            np.asarray(data.y_valid, dtype=float).ravel()
            if data.y_valid is not None
            else None
        )

    def evaluate(self, model, state, context):
        coef = np.asarray(model, dtype=float).ravel()
        pred = self._X @ coef
        residuals = pred - self._y
        mse = float(np.mean(residuals ** 2))
        grad = (2.0 / len(self._y)) * (self._X.T @ residuals)
        return Feedback(objectives=np.array([mse]), gradients=grad, constraints=np.zeros(0))


def build_solver(config=None, *, resource_context=None, component_overrides=None):
    """Canonical unified scaffold entry for this analysis/training case."""

    payload = dict(config or {}) if isinstance(config, Mapping) else {}
    overrides = dict(component_overrides or {})
    data = overrides.get("data") or build_pipeline(CSV_PATH)
    problem = overrides.get("problem") or CIDirectRegressionProblem(data)
    representation = overrides.get("representation") or build_representation(data)
    adapter = overrides.get("adapter") or GradientDescentAdapter(
        config=GradientDescentConfig(learning_rate=float(payload.get("learning_rate", 0.01)))
    )
    trainer = ComposableTrainer(
        problem=problem,
        adapter=adapter,
        representation=representation,
        run_name="traffic_symbolic_regression",
        resource_context=resource_context,
    )
    trainer.traffic_data = data
    return trainer
