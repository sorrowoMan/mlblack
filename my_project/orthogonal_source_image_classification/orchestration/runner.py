from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

from my_project.orthogonal_source_image_classification.config import ImageClassificationConfig
from my_project.orthogonal_source_image_classification.pipeline import (
    build_orthogonal_image_sources,
    fit_classification_baselines,
    search_image_representation_formulas,
    summarize_classification_winners,
)
from my_project.orthogonal_source_image_classification.problem import load_image_classification_dataset
from my_project.orthogonal_source_image_classification.reporting import write_suite_outputs


def _with_reference_deltas(rows: tuple[dict[str, Any], ...]) -> tuple[dict[str, Any], ...]:
    raw_by_model = {
        str(row.get("model")): float(row.get("test_accuracy"))
        for row in rows
        if str(row.get("feature_space")) == "raw_pixels"
    }
    representation_by_model = {
        str(row.get("model")): float(row.get("test_accuracy"))
        for row in rows
        if str(row.get("feature_space")) == "image_representation"
    }
    out: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        raw_acc = raw_by_model.get(str(item.get("model")))
        representation_acc = representation_by_model.get(str(item.get("model")))
        item["accuracy_delta_vs_raw_pixels"] = "" if raw_acc is None else float(float(item.get("test_accuracy")) - raw_acc)
        item["accuracy_delta_vs_image_representation"] = (
            "" if representation_acc is None else float(float(item.get("test_accuracy")) - representation_acc)
        )
        out.append(item)
    return tuple(out)


def run_one_dataset(*, dataset_key: str, cfg: ImageClassificationConfig) -> dict[str, Any]:
    dataset = load_image_classification_dataset(
        dataset_key=str(dataset_key),
        train_ratio=float(cfg.train_ratio),
        max_rows=int(cfg.max_rows),
        seed=int(cfg.seed),
    )
    representation_result = search_image_representation_formulas(dataset, cfg)
    source_result = build_orthogonal_image_sources(dataset, cfg, representation_result)
    metric_rows = _with_reference_deltas(
        tuple(
            dict(row)
            for row in fit_classification_baselines(
                raw_train=dataset.X_train,
                raw_test=dataset.X_test,
                formula_pool_train=representation_result.pool_train,
                formula_pool_test=representation_result.pool_test,
                representation_train=representation_result.selected_train,
                representation_test=representation_result.selected_test,
                basis_train=source_result.train_basis,
                basis_test=source_result.test_basis,
                y_train=dataset.y_train,
                y_test=dataset.y_test,
                seed=int(cfg.seed),
            )
        )
    )
    source_report = dict(source_result.report)
    rows = []
    for row in metric_rows:
        item = dict(row)
        item["scenario"] = str(dataset_key)
        item["selected_source_count"] = int(source_report.get("selected_source_count", 0))
        item["mean_source_stability"] = float(source_report.get("mean_source_stability", 0.0))
        item["pair_abs_corr_max"] = float(source_report.get("pair_abs_corr_max", 0.0))
        rows.append(item)
    source_rows = []
    for row in source_result.source_rows:
        item = dict(row)
        item["scenario"] = str(dataset_key)
        source_rows.append(item)
    return {
        "scenario": str(dataset_key),
        "dataset_metadata": dict(dataset.metadata),
        "representation_formula_rows": representation_result.formula_rows,
        "representation_report": dict(representation_result.report),
        "rows": rows,
        "source_rows": source_rows,
        "source_report": source_report,
        "winners": summarize_classification_winners(tuple(rows)),
    }


def run_suite(
    cfg: ImageClassificationConfig,
    *,
    suite_id: str | None = None,
    dataset_keys: Sequence[str] | None = None,
) -> dict[str, Any]:
    resolved_suite_id = str(suite_id or datetime.now().strftime("%Y%m%d_%H%M%S"))
    keys = tuple(str(key) for key in (tuple(dataset_keys) if dataset_keys is not None else tuple(cfg.dataset_keys)))
    output_dir = Path(cfg.output_dir).expanduser().resolve() / resolved_suite_id
    scenario_reports = []
    metric_rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    representation_rows: list[dict[str, Any]] = []
    for key in keys:
        report = run_one_dataset(dataset_key=key, cfg=cfg)
        scenario_reports.append(report)
        metric_rows.extend(tuple(report["rows"]))
        source_rows.extend(tuple(report["source_rows"]))
        for row in tuple(report["representation_formula_rows"]):
            item = dict(row)
            item["scenario"] = str(key)
            representation_rows.append(item)

    summary = {
        "protocol": "searched_symbolic_representation_then_orthogonal_source_classification_v2",
        "layer_order": [
            "raw_pixels",
            "searched_symbolic_representation_formula_pool",
            "selected_objectified_image_representation",
            "orthogonal_source_governance",
            "downstream_classification_family",
        ],
        "suite_id": resolved_suite_id,
        "config": {
            "dataset_keys": keys,
            "train_ratio": float(cfg.train_ratio),
            "seed": int(cfg.seed),
            "representation_max_features": int(cfg.representation_max_features),
            "representation_candidate_keep_top": int(cfg.representation_candidate_keep_top),
            "representation_max_pair_abs_corr": float(cfg.representation_max_pair_abs_corr),
            "max_sources": int(cfg.max_sources),
            "candidate_keep_top": int(cfg.candidate_keep_top),
            "max_pair_abs_corr": float(cfg.max_pair_abs_corr),
            "max_rows": int(cfg.max_rows),
        },
        "scenario_reports": scenario_reports,
    }
    artifacts = write_suite_outputs(
        output_dir=output_dir,
        summary=summary,
        rows=metric_rows,
        source_rows=source_rows,
        representation_rows=representation_rows,
    )
    summary["artifacts"] = artifacts
    return {
        "suite_id": resolved_suite_id,
        "output_dir": str(output_dir),
        "artifacts": artifacts,
        "rows": metric_rows,
        "source_rows": source_rows,
        "representation_rows": representation_rows,
        "summary": summary,
    }


__all__ = ["run_one_dataset", "run_suite"]
