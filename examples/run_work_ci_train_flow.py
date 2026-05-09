from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import TrainerAssemblySpec
from core.orchestration.workflow import TrainFlowSpec, run_train_flow
from examples.path_defaults import default_work_ci_csv
from examples.work_ci_reader import WorkCiIntervalReader


def main() -> None:
    reader = WorkCiIntervalReader(
        csv_path=default_work_ci_csv(),
        target_col="ci",
        test_fold_col="test_fold_10",
    )

    flow_spec = TrainFlowSpec(
        assembly=TrainerAssemblySpec(
            trainer_key="xgboost",
            pipeline_key="identity",
            trainer_params={
                "n_estimators": 360,
                "max_depth": 6,
                "learning_rate": 0.05,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
                "tree_method": "hist",
                "random_seed": 42,
            },
        ),
        eval_splits=("train", "test"),
        output_dir=str(ROOT / "examples" / "out" / "work_ci_fold10_xgboost"),
        save_artifact=True,
        save_report=True,
        run_name="work_ci_fold10_xgboost",
    )

    result = run_train_flow(reader, spec=flow_spec)

    print("WORK FLOW RESULT")
    print(f"trainer={result.report['trainer_name']}")
    print(f"feature_dim={result.report['data']['train']['feature_dim']}")
    for split, m in result.metrics.items():
        print(f"{split:5s} rmse={m['rmse']:.4f}  mae={m['mae']:.4f}  r2={m['r2']:.4f}")
    print(f"output_dir={result.output_dir}")


if __name__ == "__main__":
    main()
