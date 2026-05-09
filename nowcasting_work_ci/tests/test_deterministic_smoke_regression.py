from __future__ import annotations

from nowcasting_work_ci.run_deterministic_smoke_regression import build_summary_snapshot, diff_summary_snapshots


def _sample_summary(*, symbolic_rmse: float = 1.2) -> dict:
    return {
        "config": {
            "seed": 42,
            "interval_method": "symmetric_residual",
            "interval_alpha": 0.2,
            "pop_size": 4,
            "generations": 1,
        },
        "best_solution": {
            "subset_idx": [1, 3, 5],
            "subset_names": ["a", "b", "c"],
            "obj_coverage_error": 0.03,
            "obj_pinaw": 0.28,
            "obj_interval_score": 31.0,
        },
        "test_compare": {
            "symbolic_subset_rmse": symbolic_rmse,
            "symbolic_subset_mae": 0.8,
            "xgboost_rmse": 1.1,
            "xgboost_mae": 0.7,
            "interval_metrics": {
                "symbolic": {
                    "picp": 0.82,
                    "pinaw": 0.31,
                    "interval_score": 25.0,
                    "mean_width": 10.0,
                },
                "xgboost": {
                    "picp": 0.75,
                    "pinaw": 0.29,
                    "interval_score": 24.0,
                    "mean_width": 9.0,
                },
            },
        },
    }


def test_build_summary_snapshot_extracts_stable_fields() -> None:
    snapshot = build_summary_snapshot(_sample_summary())

    assert snapshot["config"]["seed"] == 42
    assert snapshot["best_solution"]["subset_idx"] == [1, 3, 5]
    assert snapshot["test_compare"]["symbolic_interval"]["pinaw"] == 0.31


def test_diff_summary_snapshots_detects_changed_metric() -> None:
    left = build_summary_snapshot(_sample_summary(symbolic_rmse=1.2))
    right = build_summary_snapshot(_sample_summary(symbolic_rmse=1.25))

    diffs = diff_summary_snapshots(left, right)

    assert any("symbolic_subset_rmse" in diff for diff in diffs)
