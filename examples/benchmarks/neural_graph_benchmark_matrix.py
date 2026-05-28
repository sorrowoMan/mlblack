from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter
from typing import Any, Callable

import numpy as np


def _bootstrap() -> None:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "mlblack").is_dir():
            sys.path.insert(0, str(parent))
            return


_bootstrap()

from mlblack.pipeline.data_views import GraphDataView, ImageContrastivePairDataView, ImageDataView, NumericDataView  # noqa: E402
from mlblack.presets import (  # noqa: E402
    build_tiny_cnn_image_classification_trainer,
    build_tiny_cnn_image_contrastive_trainer,
    build_tiny_gnn_graph_classification_trainer,
    build_tiny_transformer_lm_trainer,
)


def run_matrix(*, steps: int, output_path: Path, repeats: int = 1, fail_fast: bool = False) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    cases: tuple[tuple[str, Callable[[], Any]], ...] = (
        ("tiny_cnn_image_classification", lambda: build_tiny_cnn_image_classification_trainer(_image_data(), conv_channels=(4,), learning_rate=1e-2, random_seed=51)),
        ("tiny_gnn_graph_classification", lambda: build_tiny_gnn_graph_classification_trainer(_graph_data(), hidden_dim=6, num_layers=2, learning_rate=1e-2, random_seed=53)),
        ("tiny_cnn_image_contrastive", lambda: build_tiny_cnn_image_contrastive_trainer(_image_pair_data(), conv_channels=(4,), embedding_dim=4, learning_rate=1e-2, random_seed=55)),
        ("tiny_transformer_lm", lambda: build_tiny_transformer_lm_trainer(_lm_data(), vocab_size=12, max_length=5, hidden_dim=8, num_layers=1, num_heads=2, learning_rate=1e-2, random_seed=57)),
    )
    for name, builder in cases:
        runs: list[dict[str, Any]] = []
        for repeat_index in range(max(1, int(repeats))):
            start = perf_counter()
            try:
                trainer = builder()
                result = trainer.fit(max_steps=int(steps))
                elapsed = perf_counter() - start
                runs.append(
                    {
                        "repeat": int(repeat_index),
                        "status": "ok",
                        "seconds": round(float(elapsed), 6),
                        "best_score": result.report.get("best_score"),
                        "best_metrics": result.report.get("best_metrics", {}),
                        "adapter": result.report.get("adapter", {}).get("name"),
                        "representation_route": result.report.get("representation", {}).get("codec", {}).get("route"),
                        "problem": result.report.get("problem", {}).get("name"),
                    }
                )
            except BaseException as exc:
                if fail_fast:
                    raise
                elapsed = perf_counter() - start
                runs.append(
                    {
                        "repeat": int(repeat_index),
                        "status": "error",
                        "seconds": round(float(elapsed), 6),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
        ok_runs = [run for run in runs if run.get("status") == "ok"]
        seconds = np.asarray([float(run.get("seconds", 0.0)) for run in ok_runs], dtype=float)
        scores = np.asarray([float(run["best_score"]) for run in ok_runs if run.get("best_score") is not None], dtype=float)
        last_ok = ok_runs[-1] if ok_runs else {}
        rows.append(
            {
                "name": name,
                "steps": int(steps),
                "repeats": max(1, int(repeats)),
                "status": "ok" if len(ok_runs) == len(runs) else "partial_error" if ok_runs else "error",
                "ok_runs": int(len(ok_runs)),
                "seconds": None if seconds.size == 0 else round(float(np.mean(seconds)), 6),
                "seconds_mean": None if seconds.size == 0 else round(float(np.mean(seconds)), 6),
                "seconds_std": None if seconds.size == 0 else round(float(np.std(seconds)), 6),
                "seconds_min": None if seconds.size == 0 else round(float(np.min(seconds)), 6),
                "seconds_max": None if seconds.size == 0 else round(float(np.max(seconds)), 6),
                "best_score": None if scores.size == 0 else float(np.min(scores)),
                "best_score_mean": None if scores.size == 0 else float(np.mean(scores)),
                "best_metrics": last_ok.get("best_metrics", {}),
                "adapter": last_ok.get("adapter"),
                "representation_route": last_ok.get("representation_route"),
                "problem": last_ok.get("problem"),
                "runs": runs,
            }
        )
    summary = {"benchmark": "neural_graph_benchmark_matrix", "steps": int(steps), "repeats": max(1, int(repeats)), "rows": rows}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def _image_data() -> ImageDataView:
    X = np.zeros((6, 1, 4, 4), dtype=float)
    X[:3, :, :2, :2] = 1.0
    X[3:, :, 2:, 2:] = 1.0
    y = np.asarray([0, 0, 0, 1, 1, 1], dtype=float)
    return ImageDataView(X_train=X, y_train=y)


def _graph_data() -> GraphDataView:
    node_features = np.zeros((6, 4, 3), dtype=float)
    adjacency = np.zeros((6, 4, 4), dtype=float)
    y = np.asarray([0, 0, 0, 1, 1, 1], dtype=float)
    for idx in range(6):
        adjacency[idx] = np.eye(4)
        if idx < 3:
            adjacency[idx, 0, 1] = adjacency[idx, 1, 0] = 1.0
            node_features[idx, :, 0] = 1.0
        else:
            adjacency[idx, 2, 3] = adjacency[idx, 3, 2] = 1.0
            node_features[idx, :, 1] = 1.0
    return GraphDataView(node_features_train=node_features, adjacency_train=adjacency, y_train=y)


def _image_pair_data() -> ImageContrastivePairDataView:
    anchors = np.zeros((4, 1, 4, 4), dtype=float)
    positives = np.zeros_like(anchors)
    negatives = np.zeros_like(anchors)
    anchors[:, :, :2, :2] = 1.0
    positives[:, :, :2, :2] = 0.9
    negatives[:, :, 2:, 2:] = 1.0
    return ImageContrastivePairDataView(anchor_train=anchors, positive_train=positives, negative_train=negatives)


def _lm_data() -> NumericDataView:
    X = np.asarray(
        [
            [1, 2, 3, 4, 5],
            [1, 2, 2, 3, 4],
            [5, 4, 3, 2, 1],
            [6, 6, 4, 4, 2],
        ],
        dtype=float,
    )
    return NumericDataView(X_train=X, y_train=np.zeros(X.shape[0], dtype=float))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the mlblack neural graph benchmark matrix.")
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "runs" / "neural_graph_matrix_latest.json",
    )
    args = parser.parse_args()
    summary = run_matrix(
        steps=max(1, int(args.steps)),
        repeats=max(1, int(args.repeats)),
        fail_fast=bool(args.fail_fast),
        output_path=args.output,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
