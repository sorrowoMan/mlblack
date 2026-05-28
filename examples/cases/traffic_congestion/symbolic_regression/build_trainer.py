import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent))
import argparse, time, json
import pandas as pd
import numpy as np
from mlblack.pipeline.data_views import NumericDataView, train_valid_split
from mlblack.pipeline.components import ZScoreNormalizeComponent
from mlblack.pipeline.base import DataPipeline
from mlblack.core.trainer import ComposableTrainer
from mlblack.adapters.gradient_descent import GradientDescentAdapter, GradientDescentConfig
from mlblack.core.problem import LearningProblem
from mlblack.core.types import Feedback, UnknownState
from mlblack.core.representation import ModelRepresentation

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CSV_PATH = DATA_DIR / "ci_interval_opt_table_no_flow_speed_occ_lag.csv"


def _add_intercept(X):
    return np.column_stack([np.ones(X.shape[0], dtype=float), np.asarray(X, dtype=float)])


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


class CIDirectRepresentation(ModelRepresentation):
    """Simple coefficient vector representation (includes intercept)."""
    context_requires = ()
    context_provides = ("model.coefficients",)
    context_mutates = ()
    context_cache = ()
    context_notes = "Linear coefficient vector for CI regression (bias + feature weights)."

    def __init__(self, n_features, *, name="ci_linear"):
        self.name = str(name)
        self.n_features = int(n_features) + 1  # +1 for intercept

    def init(self, context):
        rng = np.random.default_rng()
        values = rng.normal(0.0, 0.01, size=(self.n_features,))
        return UnknownState(values=values)

    def encode(self, coeffs, context=None):
        return UnknownState(values=np.asarray(coeffs, dtype=float).ravel())

    def decode(self, state, context=None):
        return np.asarray(state.as_array(), dtype=float).ravel()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.01)
    args = parser.parse_args()

    df = pd.read_csv(CSV_PATH)
    feature_cols = [c for c in df.columns if c not in ("date", "ci", "Unnamed: 0") and not c.startswith("test_fold_")]
    X = df[feature_cols].values.astype(float)
    y = df["ci"].values.astype(float)

    raw_data = train_valid_split(
        X, y,
        feature_names=feature_cols,
        target_name="ci",
        valid_ratio=0.2,
        seed=42,
    )

    pipeline = DataPipeline([ZScoreNormalizeComponent()])
    data = pipeline.fit_transform(raw_data)

    problem = CIDirectRegressionProblem(data)
    rep = CIDirectRepresentation(data.n_features)

    gd_config = GradientDescentConfig(learning_rate=args.lr)
    adapter = GradientDescentAdapter(config=gd_config)

    trainer = ComposableTrainer(problem=problem, adapter=adapter, representation=rep)
    trainer.context_store["step"] = 0

    if args.check:
        print(f"[check] assembly ok | problem={type(problem).__name__} | adapter={type(adapter).__name__}")
        return

    t0 = time.time()
    result = trainer.fit(max_steps=args.steps)
    elapsed = time.time() - t0

    best = result.best_state.as_array()
    pred_train = _add_intercept(data.X_train) @ best
    pred_val = _add_intercept(data.X_valid) @ best if data.X_valid is not None else None
    train_rmse = np.sqrt(np.mean((pred_train - data.y_train) ** 2))
    val_rmse = np.sqrt(np.mean((pred_val - data.y_valid) ** 2)) if pred_val is not None else float("nan")

    print(f"Symbolic-compatible Linear Regression on Traffic CI")
    print(f"  Features: {data.n_features} (+ intercept), Steps: {args.steps}, LR: {args.lr}")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Train RMSE: {train_rmse:.4f}")
    print(f"  Valid RMSE: {val_rmse:.4f}")
    print(f"  Intercept: {best[0]:+.4f}")
    print(f"  Top 5 feature coefficients (abs):")
    top_idx = 1 + np.argsort(np.abs(best[1:]))[-5:][::-1]
    for i in top_idx:
        print(f"    {feature_cols[i - 1]:30s}: {best[i]:+.6f}")

if __name__ == "__main__":
    main()
