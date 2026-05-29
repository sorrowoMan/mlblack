import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent))
import argparse
import time
import pandas as pd
import numpy as np
from mlblack.pipeline.data_views import NumericDataView, train_valid_split
from mlblack.assembly import build_trainer

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CSV_PATH = DATA_DIR / "ci_interval_opt_table_no_flow_speed_occ_lag.csv"


def load_data():
    df = pd.read_csv(CSV_PATH)
    feature_cols = [c for c in df.columns if c not in ("date", "ci", "Unnamed: 0") and not c.startswith("test_fold_")]
    X = df[feature_cols].values.astype(float)
    y = df["ci"].values.astype(float)
    return train_valid_split(X, y, feature_names=tuple(feature_cols), target_name="ci", valid_ratio=0.2, seed=42)


def build_solver():
    """Canonical unified scaffold entry; returns the assembled Trainer."""

    data = load_data()
    spec = {
        "preset": "xgboost",
        "resource_context": {"device": "cpu", "threads": 4},
        "params": {"population_size": 8, "mutation_scale": 0.2},
    }
    return build_trainer(spec, data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    data = load_data()

    spec = {
        "preset": "xgboost",
        "resource_context": {"device": "cpu", "threads": 4},
        "params": {"population_size": 8, "mutation_scale": 0.2},
    }

    trainer = build_trainer(spec, data)
    if args.check:
        print(f"[check] assembly ok | problem={type(trainer.problem).__name__} | adapter={type(trainer.adapter).__name__}")
        return

    t0 = time.time()
    result = trainer.fit(max_steps=50)
    elapsed = time.time() - t0

    report = trainer.build_report()
    metrics = report.get("best_metrics", {})
    print(f"XGBoost on Traffic CI ({len(data.X_train)} train, {len(data.X_valid)} valid)")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Train RMSE: {metrics.get('train.rmse', 'N/A')}")
    print(f"  Valid RMSE: {metrics.get('valid.rmse', 'N/A')}")
    print(f"  Valid R2:  {metrics.get('valid.r2', 'N/A')}")


if __name__ == "__main__":
    main()
